"""
services/feature_engineering.py
Generates advanced ML features from raw fixture data.
Rolling averages, form, attack/defense strength, momentum, fatigue.
"""
import numpy as np
import pandas as pd
from typing import Tuple


class FeatureEngineer:
    """Transforms raw football data into ML-ready features."""

    ROLLING_WINDOWS = [3, 5, 10]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature engineering pipeline."""
        df = df.copy()
        df = self._clean(df)
        df = self._add_targets(df)
        df = self._add_form_features(df)
        df = self._add_strength_features(df)
        df = self._add_odds_features(df)
        df = self._add_fatigue(df)
        df = self._normalize(df)
        return df.dropna(subset=self._feature_cols())

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and sort data."""
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["league_id", "date"]).reset_index(drop=True)
        for col in ["home_goals","away_goals","total_goals","home_xg","away_xg",
                    "home_shots","away_shots","home_corners","away_corners",
                    "home_cards","away_cards","home_possession"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary target variables for each market."""
        df["target_over15"]  = (df["total_goals"] >= 2).astype(int)
        df["target_over25"]  = (df["total_goals"] >= 3).astype(int)
        df["target_over35"]  = (df["total_goals"] >= 4).astype(int)
        df["target_under25"] = (df["total_goals"] <  3).astype(int)
        df["target_btts"]    = (
            (df["home_goals"] > 0) & (df["away_goals"] > 0)
        ).astype(int)
        df["target_over95c"] = (
            (df["home_corners"] + df["away_corners"]) >= 10
        ).astype(int)
        df["target_over35k"] = (
            (df["home_cards"] + df["away_cards"]) >= 4
        ).astype(int)
        return df

    def _rolling_team_stats(self, df: pd.DataFrame, team_col: str,
                             goal_scored_col: str, goal_conceded_col: str,
                             prefix: str, window: int) -> pd.DataFrame:
        """Calculate rolling stats for home or away teams."""
        for team in df[team_col].unique():
            mask = df[team_col] == team
            idx  = df[mask].index

            df.loc[idx, f"{prefix}_goals_scored_avg{window}"] = (
                df.loc[mask, goal_scored_col]
                  .rolling(window, min_periods=1).mean().shift(1).values
            )
            df.loc[idx, f"{prefix}_goals_conceded_avg{window}"] = (
                df.loc[mask, goal_conceded_col]
                  .rolling(window, min_periods=1).mean().shift(1).values
            )
        return df

    def _add_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling averages for goals, xG, shots, corners, cards."""
        for w in self.ROLLING_WINDOWS:
            # Home team rolling stats
            df = self._rolling_team_stats(
                df, "home_team", "home_goals", "away_goals", "home", w
            )
            # Away team rolling stats
            df = self._rolling_team_stats(
                df, "away_team", "away_goals", "home_goals", "away", w
            )

        # xG rolling averages
        for team_col, xg_for, xg_ag, prefix in [
            ("home_team", "home_xg", "away_xg", "home"),
            ("away_team", "away_xg", "home_xg", "away"),
        ]:
            for team in df[team_col].unique():
                mask = df[team_col] == team
                idx  = df[mask].index
                df.loc[idx, f"{prefix}_xg_avg5"] = (
                    df.loc[mask, xg_for].rolling(5, min_periods=1).mean().shift(1).values
                )
                df.loc[idx, f"{prefix}_xga_avg5"] = (
                    df.loc[mask, xg_ag].rolling(5, min_periods=1).mean().shift(1).values
                )

        # Corners rolling
        for team_col, c_for, prefix in [
            ("home_team", "home_corners", "home"),
            ("away_team", "away_corners", "away"),
        ]:
            for team in df[team_col].unique():
                mask = df[team_col] == team
                idx  = df[mask].index
                df.loc[idx, f"{prefix}_corners_avg5"] = (
                    df.loc[mask, c_for].rolling(5, min_periods=1).mean().shift(1).values
                )

        # Last 5 form (points: W=3, D=1, L=0)
        for team_col, goals_for, goals_ag, prefix in [
            ("home_team", "home_goals", "away_goals", "home"),
            ("away_team", "away_goals", "home_goals", "away"),
        ]:
            for team in df[team_col].unique():
                mask = df[team_col] == team
                idx  = df[mask].index
                gf   = df.loc[mask, goals_for].values
                ga   = df.loc[mask, goals_ag].values
                pts  = [3 if f>a else 1 if f==a else 0 for f,a in zip(gf,ga)]
                form = pd.Series(pts, index=idx).rolling(5, min_periods=1).mean().shift(1)
                df.loc[idx, f"{prefix}_form5"] = form.values

        return df

    def _add_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attack/defense strength relative to league average."""
        for league in df["league_id"].unique():
            lmask = df["league_id"] == league
            league_avg_goals = df.loc[lmask, "total_goals"].mean() or 2.5

            # Home attack strength = home goals / league avg
            df.loc[lmask, "home_attack_strength"] = (
                df.loc[lmask, "home_goals_scored_avg5"] / league_avg_goals * 2
            ).clip(0, 3)

            # Away defense weakness = goals conceded away / league avg
            df.loc[lmask, "away_defense_weakness"] = (
                df.loc[lmask, "away_goals_conceded_avg5"] / league_avg_goals * 2
            ).clip(0, 3)

            df.loc[lmask, "league_avg_goals"] = league_avg_goals

        # xG differential
        df["xg_diff"] = (
            df.get("home_xg_avg5", df["home_xg"]) -
            df.get("away_xg_avg5", df["away_xg"])
        )

        # Shot conversion rate
        for prefix, shots, goals in [
            ("home", "home_shots", "home_goals"),
            ("away", "away_shots", "away_goals"),
        ]:
            df[f"{prefix}_conversion"] = (
                df[goals] / df[shots].replace(0, 1)
            ).clip(0, 1)

        # Momentum (weighted recent form: last 3 weighted more)
        df["home_momentum"] = (
            df.get("home_form5", 1.5) * 0.6 +
            df.get("home_goals_scored_avg3", 1.2) * 0.4
        ).fillna(1.5)

        df["away_momentum"] = (
            df.get("away_form5", 1.5) * 0.6 +
            df.get("away_goals_scored_avg3", 1.2) * 0.4
        ).fillna(1.5)

        return df

    def _add_odds_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implied probabilities and value from bookmaker odds."""
        for odds_col, prob_col in [
            ("home_odds",    "implied_home"),
            ("draw_odds",    "implied_draw"),
            ("away_odds",    "implied_away"),
            ("over25_odds",  "implied_over25"),
            ("btts_yes_odds","implied_btts"),
        ]:
            if odds_col in df.columns:
                df[prob_col] = (1 / df[odds_col].replace(0, np.inf)).clip(0, 1)

        # Overround (bookmaker margin)
        if all(c in df.columns for c in ["home_odds","draw_odds","away_odds"]):
            df["overround"] = (
                df["implied_home"] + df["implied_draw"] + df["implied_away"]
            ).clip(1, 2)

        # Log odds (better for ML)
        for col in ["home_odds","away_odds","over25_odds"]:
            if col in df.columns:
                df[f"log_{col}"] = np.log(df[col].replace(0, np.nan)).fillna(0)

        return df

    def _add_fatigue(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fatigue score based on days since last match."""
        for team_col, prefix in [("home_team","home"),("away_team","away")]:
            for team in df[team_col].unique():
                mask = df[team_col] == team
                idx  = df[mask].index
                dates = df.loc[mask, "date"]
                days_rest = dates.diff().dt.days.fillna(7)
                df.loc[idx, f"{prefix}_days_rest"] = days_rest.values

        # Fatigue score: < 4 days = high fatigue
        df["home_fatigue"] = (df.get("home_days_rest", 7) < 4).astype(int)
        df["away_fatigue"] = (df.get("away_days_rest", 7) < 4).astype(int)

        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Min-max normalize continuous features."""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols  = [c for c in num_cols if c.startswith("target_")]
        feature_cols = [c for c in num_cols if not c.startswith("target_")
                        and c not in ["fixture_id","league_id","season"]]
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols].fillna(0))
        return df

    def _feature_cols(self) -> List[str]:
        return [
            "home_goals_scored_avg5","home_goals_conceded_avg5",
            "away_goals_scored_avg5","away_goals_conceded_avg5",
            "home_xg_avg5","away_xg_avg5","home_corners_avg5","away_corners_avg5",
            "home_form5","away_form5","home_attack_strength","away_defense_weakness",
            "xg_diff","home_conversion","away_conversion",
            "home_momentum","away_momentum",
            "home_fatigue","away_fatigue","league_avg_goals",
        ]

    def get_feature_cols(self) -> List[str]:
        return self._feature_cols()

    def prepare_for_training(self, df: pd.DataFrame, target: str) -> Tuple:
        """Return X, y ready for sklearn."""
        feature_cols = [c for c in self._feature_cols() if c in df.columns]
        valid = df[feature_cols + [target]].dropna()
        X = valid[feature_cols].values
        y = valid[target].values
        return X, y, feature_cols


from typing import List
