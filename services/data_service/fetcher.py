"""
SERVICE 1: DATA SERVICE
Fetches real fixtures, team stats, and odds from external APIs.
Single source of truth for all match data.
"""
import httpx
import json
import os
import time
from typing import List, Dict, Optional
from datetime import datetime


class DataService:
    """Fetches and normalises football data from multiple sources."""

    # Class-level cache shared across all instances
    _fixture_cache: Dict[str, Dict] = {}
    CACHE_TTL_SECONDS = 1800  # 30 minutes

    def __init__(self):
        self.api_football_key = os.getenv("API_FOOTBALL_KEY", "")
        self.odds_api_key = os.getenv("ODDS_API_KEY", "")
        self._load_allowed_leagues()

    def _load_allowed_leagues(self):
        try:
            path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.json")
            with open(path) as f:
                self.allowed_leagues = set(json.load(f).get("leagues", []))
        except Exception:
            self.allowed_leagues = set([
                1,2,3,848,39,40,41,42,43,78,79,81,82,83,84,
                140,141,135,136,61,62,88,94,179,144,145,172,
                203,204,307,371,253,254,71,72,128,197,793,
                45,46,80,66,90,96,143,137,146,181,73,129,
                802,804,806,116,794,807,799,783,752,851,12,20,29,
                1,4,5,6,9,11,13,531,528
            ])

    # ── FIXTURES ─────────────────────────────────────────────────
    async def get_fixtures(self, date: str) -> List[Dict]:
        """Fetch today's fixtures from API-Football, filtered by allowed leagues. Cached for 30 min."""
        # Check cache first to save API quota
        if date in self._fixture_cache:
            cached = self._fixture_cache[date]
            if time.time() - cached["timestamp"] < self.CACHE_TTL_SECONDS:
                print(f"[DataService] Using cached fixtures for {date}")
                return cached["data"]

        if not self.api_football_key:
            return []

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.get(
                    f"https://v3.football.api-sports.io/fixtures?date={date}",
                    headers={"x-apisports-key": self.api_football_key}
                )
                data = r.json()
                # Log quota info if present
                remaining = r.headers.get("x-ratelimit-requests-remaining")
                if remaining:
                    print(f"[DataService] API quota remaining: {remaining}")
            except Exception as e:
                print(f"[DataService] Fixtures fetch error: {e}")
                return []

        fixtures = []
        for f in data.get("response", []):
            league_id = f.get("league", {}).get("id", 0)
            status = f.get("fixture", {}).get("status", {}).get("short", "")

            if league_id not in self.allowed_leagues:
                continue
            if status not in ("NS", "TBD", "PST", "1H", "2H", "HT"):
                continue

            home = f.get("teams", {}).get("home", {}).get("name", "")
            away = f.get("teams", {}).get("away", {}).get("name", "")
            if not home or not away:
                continue

            fixture_date = f.get("fixture", {}).get("date", "")
            try:
                dt = datetime.fromisoformat(fixture_date.replace("Z", "+00:00"))
                time_str = dt.strftime("%H:%M") + " GMT"
            except Exception:
                time_str = "TBD"

            fixtures.append({
                "fixture_id": f.get("fixture", {}).get("id"),
                "home": home,
                "away": away,
                "league": f.get("league", {}).get("name", "Unknown"),
                "country": f.get("league", {}).get("country", ""),
                "league_id": league_id,
                "time": time_str,
                "date": date,
                "venue": f.get("fixture", {}).get("venue", {}).get("name", ""),
                "status": status,
            })

        # Save to cache
        self._fixture_cache[date] = {
            "data": fixtures,
            "timestamp": time.time(),
        }

        return fixtures

    # ── TEAM STATS ───────────────────────────────────────────────
    async def get_team_stats(self, team_id: int, league_id: int, season: int = 2025) -> Dict:
        """Fetch team statistics for a specific league/season."""
        if not self.api_football_key:
            return {}

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.get(
                    f"https://v3.football.api-sports.io/teams/statistics"
                    f"?team={team_id}&league={league_id}&season={season}",
                    headers={"x-apisports-key": self.api_football_key}
                )
                data = r.json()
                return data.get("response", {})
            except Exception:
                return {}

    # ── HEAD TO HEAD ─────────────────────────────────────────────
    async def get_h2h(self, team1_id: int, team2_id: int, last: int = 6) -> List[Dict]:
        """Fetch head-to-head history between two teams."""
        if not self.api_football_key:
            return []

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.get(
                    f"https://v3.football.api-sports.io/fixtures/headtohead"
                    f"?h2h={team1_id}-{team2_id}&last={last}",
                    headers={"x-apisports-key": self.api_football_key}
                )
                data = r.json()
                return data.get("response", [])
            except Exception:
                return []

    # ── LIVE ODDS ────────────────────────────────────────────────
    async def get_odds(self, fixture_id: int) -> Dict:
        """Fetch odds for a specific fixture from API-Football."""
        if not self.api_football_key:
            return {}

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.get(
                    f"https://v3.football.api-sports.io/odds?fixture={fixture_id}",
                    headers={"x-apisports-key": self.api_football_key}
                )
                data = r.json()
                response = data.get("response", [])
                if response:
                    return self._parse_odds(response[0])
                return {}
            except Exception:
                return {}

    def _parse_odds(self, raw: Dict) -> Dict:
        """Parse raw odds into clean format."""
        odds = {}
        for bookmaker in raw.get("bookmakers", []):
            for bet in bookmaker.get("bets", []):
                market = bet.get("name", "")
                for value in bet.get("values", []):
                    key = f"{market}_{value.get('value', '')}"
                    try:
                        odds[key] = float(value.get("odd", 0))
                    except (ValueError, TypeError):
                        pass
        return odds

    # ── FORMAT FOR AI PROMPT ─────────────────────────────────────
    def format_for_prompt(self, fixtures: List[Dict]) -> str:
        """Format ALL fixtures into text for AI analysis prompt."""
        lines = []
        for i, f in enumerate(fixtures, 1):
            lines.append(f"{i}. {f['home']} vs {f['away']} | "
                         f"{f['league']} ({f['country']}) | {f['time']}")
        return "\n".join(lines)
