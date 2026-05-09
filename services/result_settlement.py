"""
services/result_settlement.py
Checks finished matches, grades predictions, updates ML statistics.
Run as a scheduled job (e.g. every 3 hours via APScheduler).
"""
import os
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import Dict, List


DB_PATH = os.getenv("DB_PATH", "database/football.db")
API_KEY = os.getenv("API_FOOTBALL_KEY", "")


class ResultSettlement:
    """Automatically settles predictions after matches complete."""

    def __init__(self):
        self.api_key = API_KEY
        self.db_path = DB_PATH

    def _api_get(self, endpoint: str, params: dict) -> dict:
        try:
            r = requests.get(
                f"https://v3.football.api-sports.io/{endpoint}",
                headers={"x-apisports-key": self.api_key},
                params=params, timeout=15,
            )
            return r.json()
        except Exception as e:
            print(f"[Settlement] API error: {e}")
            return {}

    def fetch_finished_fixtures(self, date: str) -> List[Dict]:
        """Get finished fixtures for a date."""
        data = self._api_get("fixtures", {"date": date, "status": "FT"})
        return data.get("response", [])

    def grade_prediction(self, market: str, pick: str,
                         home_goals: int, away_goals: int,
                         home_corners: int = 0, away_corners: int = 0,
                         home_cards: int = 0, away_cards: int = 0) -> str:
        """
        Grade a prediction as WIN or LOSS.
        Returns: "WIN", "LOSS", or "VOID"
        """
        total_goals   = home_goals + away_goals
        total_corners = home_corners + away_corners
        total_cards   = home_cards + away_cards
        btts          = home_goals > 0 and away_goals > 0

        pick_lower = pick.lower()

        if "over 1.5" in pick_lower:
            return "WIN" if total_goals >= 2 else "LOSS"
        elif "over 2.5" in pick_lower:
            return "WIN" if total_goals >= 3 else "LOSS"
        elif "over 3.5" in pick_lower:
            return "WIN" if total_goals >= 4 else "LOSS"
        elif "over 4.5" in pick_lower:
            return "WIN" if total_goals >= 5 else "LOSS"
        elif "under 1.5" in pick_lower:
            return "WIN" if total_goals <= 1 else "LOSS"
        elif "under 2.5" in pick_lower:
            return "WIN" if total_goals <= 2 else "LOSS"
        elif "under 3.5" in pick_lower:
            return "WIN" if total_goals <= 3 else "LOSS"
        elif "btts yes" in pick_lower:
            return "WIN" if btts else "LOSS"
        elif "btts no" in pick_lower:
            return "WIN" if not btts else "LOSS"
        elif "over 8.5 corners" in pick_lower:
            return "WIN" if total_corners >= 9 else "LOSS"
        elif "over 9.5 corners" in pick_lower:
            return "WIN" if total_corners >= 10 else "LOSS"
        elif "over 10.5 corners" in pick_lower:
            return "WIN" if total_corners >= 11 else "LOSS"
        elif "over 3.5 cards" in pick_lower:
            return "WIN" if total_cards >= 4 else "LOSS"
        elif "over 4.5 cards" in pick_lower:
            return "WIN" if total_cards >= 5 else "LOSS"
        elif "first half over 0.5" in pick_lower:
            return "VOID"  # Need half-time score
        else:
            return "VOID"

    def calculate_profit(self, result: str, stake: float, odds: float) -> float:
        """Calculate profit/loss from a bet result."""
        if result == "WIN":
            return round(stake * (odds - 1), 2)
        elif result == "LOSS":
            return round(-stake, 2)
        return 0.0

    def settle_predictions(self, date: str = None) -> Dict:
        """
        Main settlement run for a given date.
        Grades all unsettled predictions for finished matches.
        """
        date = date or (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"[Settlement] Settling predictions for {date}...")

        fixtures = self.fetch_finished_fixtures(date)
        if not fixtures:
            print(f"[Settlement] No finished fixtures found for {date}")
            return {"settled": 0, "date": date}

        conn = sqlite3.connect(self.db_path)
        settled = 0
        wins = 0

        for f in fixtures:
            fid         = f.get("fixture", {}).get("id")
            home_goals  = f.get("goals", {}).get("home", 0) or 0
            away_goals  = f.get("goals", {}).get("away", 0) or 0

            # Get stats for corners/cards
            stats_raw = requests.get(
                f"https://v3.football.api-sports.io/fixtures/statistics",
                headers={"x-apisports-key": self.api_key},
                params={"fixture": fid}, timeout=10,
            ).json().get("response", [])

            home_corners = away_corners = home_cards = away_cards = 0
            for team_data in stats_raw:
                for stat in team_data.get("statistics", []):
                    val = int(stat.get("value") or 0)
                    t   = stat.get("type", "")
                    if stat == stats_raw[0]:  # home
                        if t == "Corner Kicks":   home_corners = val
                        if t == "Yellow Cards":   home_cards += val
                        if t == "Red Cards":      home_cards += val
                    else:  # away
                        if t == "Corner Kicks":   away_corners = val
                        if t == "Yellow Cards":   away_cards += val
                        if t == "Red Cards":      away_cards += val

            # Find unsettled predictions for this fixture
            cursor = conn.execute(
                "SELECT id, market, pick, bookmaker_odds, ml_probability FROM predictions "
                "WHERE fixture_id=? AND result IS NULL",
                (fid,)
            )
            rows = cursor.fetchall()

            for row in rows:
                pred_id, market, pick, odds, ml_prob = row
                result = self.grade_prediction(
                    market, pick or market,
                    home_goals, away_goals,
                    home_corners, away_corners,
                    home_cards, away_cards,
                )

                if result == "VOID":
                    continue

                profit = self.calculate_profit(result, 1.0, odds or 1.90)

                conn.execute(
                    "UPDATE predictions SET result=?, profit_loss=?, settled_at=? WHERE id=?",
                    (result, profit, datetime.utcnow().isoformat(), pred_id)
                )

                # Update model_performance
                if ml_prob:
                    actual = 1 if result == "WIN" else 0
                    brier_contrib = (ml_prob - actual) ** 2
                    conn.execute(
                        "INSERT INTO model_performance (model_name, market, brier_score, n_predictions, trained_at) "
                        "VALUES (?, ?, ?, 1, ?) "
                        "ON CONFLICT DO NOTHING",
                        ("ml_predictor", market, brier_contrib, datetime.utcnow().isoformat())
                    )

                settled += 1
                if result == "WIN":
                    wins += 1

        conn.commit()
        conn.close()

        summary = {
            "date":    date,
            "settled": settled,
            "wins":    wins,
            "losses":  settled - wins,
            "win_rate": round(wins / settled * 100, 1) if settled > 0 else 0,
        }
        print(f"[Settlement] Done: {summary}")
        return summary

    def get_performance_stats(self) -> Dict:
        """Get overall prediction performance from DB."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END), "
            "SUM(profit_loss), AVG(profit_loss) "
            "FROM predictions WHERE result IS NOT NULL"
        )
        row = cursor.fetchone()
        conn.close()

        total, wins, total_profit, avg_profit = row
        total = total or 0
        wins  = wins or 0

        return {
            "total_predictions": total,
            "wins":              wins,
            "losses":            total - wins,
            "win_rate":          round(wins / total * 100, 1) if total > 0 else 0,
            "total_profit":      round(total_profit or 0, 2),
            "avg_profit":        round(avg_profit or 0, 4),
            "roi":               round((total_profit or 0) / total * 100, 2) if total > 0 else 0,
        }
