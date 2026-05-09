"""
services/dataset_builder.py
Pulls historical fixtures from API-Football and builds ML training dataset.
Stores goals, xG, shots, possession, corners, cards, form, odds, league strength.
"""
import os
import json
import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path


DB_PATH = os.getenv("DB_PATH", "database/football.db")
API_KEY  = os.getenv("API_FOOTBALL_KEY", "")

# Leagues to collect historical data for
TARGET_LEAGUES = [
    2, 3, 848,          # UCL, Europa, Conference
    39, 40, 41, 42,     # EPL, Championship, L1, L2
    140, 135, 78, 61,   # La Liga, Serie A, Bundesliga, Ligue 1
    88, 94, 144, 203,   # Eredivisie, Portugal, Belgium, Turkey
    172, 307, 371,      # Bulgaria, Saudi, Armenia
    71, 128, 253,       # Brazil, Argentina, MLS
]


class DatasetBuilder:
    """Builds and maintains the historical football dataset."""

    def __init__(self):
        self.api_key = API_KEY
        self.db_path = DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create SQLite tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS fixtures (
                fixture_id    INTEGER PRIMARY KEY,
                league_id     INTEGER,
                league_name   TEXT,
                country       TEXT,
                season        INTEGER,
                date          TEXT,
                home_team     TEXT,
                away_team     TEXT,
                home_goals    INTEGER,
                away_goals    INTEGER,
                total_goals   INTEGER,
                home_xg       REAL,
                away_xg       REAL,
                home_shots    INTEGER,
                away_shots    INTEGER,
                home_shots_ot INTEGER,
                away_shots_ot INTEGER,
                home_possession REAL,
                away_possession REAL,
                home_corners  INTEGER,
                away_corners  INTEGER,
                home_cards    INTEGER,
                away_cards    INTEGER,
                home_fouls    INTEGER,
                away_fouls    INTEGER,
                home_odds     REAL,
                draw_odds     REAL,
                away_odds     REAL,
                over25_odds   REAL,
                btts_yes_odds REAL,
                status        TEXT,
                created_at    TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS team_form (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name     TEXT,
                league_id     INTEGER,
                season        INTEGER,
                fixture_id    INTEGER,
                date          TEXT,
                is_home       INTEGER,
                goals_scored  INTEGER,
                goals_conceded INTEGER,
                result        TEXT,
                xg_for        REAL,
                xg_against    REAL,
                shots_for     INTEGER,
                shots_against INTEGER,
                corners_for   INTEGER,
                corners_against INTEGER,
                cards         INTEGER
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id    INTEGER,
                market        TEXT,
                ml_probability REAL,
                ai_probability REAL,
                final_probability REAL,
                bookmaker_odds REAL,
                value         REAL,
                is_value_bet  INTEGER,
                strategy      TEXT,
                result        TEXT,
                profit_loss   REAL,
                created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
                settled_at    TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name    TEXT,
                market        TEXT,
                accuracy      REAL,
                log_loss      REAL,
                roc_auc       REAL,
                brier_score   REAL,
                calibration   REAL,
                n_predictions INTEGER,
                trained_at    TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        print(f"[DatasetBuilder] Database initialized at {self.db_path}")

    def _api_get(self, endpoint: str, params: dict) -> dict:
        """Make API-Football request."""
        try:
            r = requests.get(
                f"https://v3.football.api-sports.io/{endpoint}",
                headers={"x-apisports-key": self.api_key},
                params=params,
                timeout=15,
            )
            return r.json()
        except Exception as e:
            print(f"[DatasetBuilder] API error: {e}")
            return {}

    def fetch_season_fixtures(self, league_id: int, season: int) -> List[Dict]:
        """Fetch all finished fixtures for a league season."""
        data = self._api_get("fixtures", {
            "league": league_id,
            "season": season,
            "status": "FT",
        })
        return data.get("response", [])

    def fetch_fixture_stats(self, fixture_id: int) -> Dict:
        """Fetch detailed stats for a fixture."""
        data = self._api_get("fixtures/statistics", {"fixture": fixture_id})
        stats = {}
        for team_data in data.get("response", []):
            team_name = team_data.get("team", {}).get("name", "")
            for stat in team_data.get("statistics", []):
                key = f"{team_name}_{stat['type'].lower().replace(' ', '_')}"
                stats[key] = stat.get("value")
        return stats

    def fetch_odds(self, fixture_id: int) -> Dict:
        """Fetch pre-match odds for a fixture."""
        data = self._api_get("odds", {
            "fixture": fixture_id,
            "bookmaker": 6,  # Bwin - widely available
        })
        odds = {}
        for response in data.get("response", [])[:1]:
            for bookmaker in response.get("bookmakers", [])[:1]:
                for bet in bookmaker.get("bets", []):
                    if bet.get("name") == "Match Winner":
                        for v in bet.get("values", []):
                            odds[f"match_{v['value'].lower()}"] = float(v.get("odd", 0))
                    elif bet.get("name") == "Goals Over/Under":
                        for v in bet.get("values", []):
                            key = v["value"].replace(" ", "_").lower()
                            odds[f"goals_{key}"] = float(v.get("odd", 0))
                    elif bet.get("name") == "Both Teams Score":
                        for v in bet.get("values", []):
                            odds[f"btts_{v['value'].lower()}"] = float(v.get("odd", 0))
        return odds

    def parse_fixture(self, f: Dict, stats: Dict, odds: Dict) -> Optional[Dict]:
        """Parse raw fixture data into clean row."""
        try:
            fix   = f.get("fixture", {})
            teams = f.get("teams", {})
            goals = f.get("goals", {})
            league = f.get("league", {})

            home = teams.get("home", {}).get("name", "")
            away = teams.get("away", {}).get("name", "")

            def get_stat(team, key, default=0):
                val = stats.get(f"{team}_{key}", default)
                try: return float(val) if val is not None else default
                except: return default

            return {
                "fixture_id":       fix.get("id"),
                "league_id":        league.get("id"),
                "league_name":      league.get("name"),
                "country":          league.get("country"),
                "season":           league.get("season"),
                "date":             fix.get("date", "")[:10],
                "home_team":        home,
                "away_team":        away,
                "home_goals":       goals.get("home", 0) or 0,
                "away_goals":       goals.get("away", 0) or 0,
                "total_goals":      (goals.get("home") or 0) + (goals.get("away") or 0),
                "home_xg":          get_stat(home, "expected_goals"),
                "away_xg":          get_stat(away, "expected_goals"),
                "home_shots":       get_stat(home, "total_shots"),
                "away_shots":       get_stat(away, "total_shots"),
                "home_shots_ot":    get_stat(home, "shots_on_goal"),
                "away_shots_ot":    get_stat(away, "shots_on_goal"),
                "home_possession":  get_stat(home, "ball_possession"),
                "away_possession":  get_stat(away, "ball_possession"),
                "home_corners":     get_stat(home, "corner_kicks"),
                "away_corners":     get_stat(away, "corner_kicks"),
                "home_cards":       get_stat(home, "yellow_cards") + get_stat(home, "red_cards"),
                "away_cards":       get_stat(away, "yellow_cards") + get_stat(away, "red_cards"),
                "home_fouls":       get_stat(home, "fouls"),
                "away_fouls":       get_stat(away, "fouls"),
                "home_odds":        odds.get("match_home", 0),
                "draw_odds":        odds.get("match_draw", 0),
                "away_odds":        odds.get("match_away", 0),
                "over25_odds":      odds.get("goals_over_2.5", 0),
                "btts_yes_odds":    odds.get("btts_yes", 0),
                "status":           "FT",
            }
        except Exception as e:
            print(f"[DatasetBuilder] Parse error: {e}")
            return None

    def save_fixture(self, row: Dict):
        """Upsert fixture row into database."""
        conn = sqlite3.connect(self.db_path)
        cols = ", ".join(row.keys())
        vals = ", ".join(["?" for _ in row])
        conn.execute(
            f"INSERT OR REPLACE INTO fixtures ({cols}) VALUES ({vals})",
            list(row.values())
        )
        conn.commit()
        conn.close()

    def build_dataset(self, league_ids: List[int] = None, seasons: List[int] = None):
        """Main entry: fetch and store historical data for given leagues/seasons."""
        if not self.api_key:
            print("[DatasetBuilder] No API_FOOTBALL_KEY set")
            return

        leagues  = league_ids or TARGET_LEAGUES[:5]  # Start small
        seasons  = seasons or [2023, 2024]

        total = 0
        for league_id in leagues:
            for season in seasons:
                print(f"[DatasetBuilder] Fetching league {league_id} season {season}...")
                fixtures = self.fetch_season_fixtures(league_id, season)
                print(f"[DatasetBuilder] Found {len(fixtures)} fixtures")

                for f in fixtures[:50]:  # Limit to save API quota
                    fid = f.get("fixture", {}).get("id")
                    if not fid:
                        continue

                    stats = self.fetch_fixture_stats(fid)
                    odds  = self.fetch_odds(fid)
                    row   = self.parse_fixture(f, stats, odds)

                    if row:
                        self.save_fixture(row)
                        total += 1

        print(f"[DatasetBuilder] Done. Saved {total} fixtures.")
        return total

    def export_csv(self, path: str = "database/training_data.csv") -> str:
        """Export fixtures table to CSV for ML training."""
        conn  = sqlite3.connect(self.db_path)
        df    = pd.read_sql("SELECT * FROM fixtures WHERE status='FT'", conn)
        conn.close()
        df.to_csv(path, index=False)
        print(f"[DatasetBuilder] Exported {len(df)} rows to {path}")
        return path

    def get_dataframe(self) -> pd.DataFrame:
        """Return fixtures as DataFrame."""
        conn = sqlite3.connect(self.db_path)
        df   = pd.read_sql("SELECT * FROM fixtures WHERE status='FT'", conn)
        conn.close()
        return df


if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build_dataset(league_ids=[39], seasons=[2024])
    builder.export_csv()
