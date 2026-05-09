"""
odds_fetcher.py - Fetches real bookmaker odds from The Odds API
Free tier: 500 requests/month
Covers: EPL, La Liga, Serie A, Bundesliga, Ligue 1, UCL, MLS, etc.
"""
import requests
import os
from typing import Dict, List, Optional


# Map our league names to The Odds API sport keys
LEAGUE_MAP = {
    "Premier League":       "soccer_epl",
    "Championship":         "soccer_england_championship",
    "League One":           "soccer_england_league1",
    "League Two":           "soccer_england_league2",
    "La Liga":              "soccer_spain_la_liga",
    "La Liga 2":            "soccer_spain_segunda_division",
    "Serie A":              "soccer_italy_serie_a",
    "Serie B":              "soccer_italy_serie_b",
    "Bundesliga":           "soccer_germany_bundesliga",
    "2. Bundesliga":        "soccer_germany_bundesliga2",
    "Ligue 1":              "soccer_france_ligue_one",
    "Ligue 2":              "soccer_france_ligue_two",
    "Eredivisie":           "soccer_netherlands_eredivisie",
    "Primeira Liga":        "soccer_portugal_primeira_liga",
    "Scottish Premiership": "soccer_scotland_premiership",
    "Belgian Pro League":   "soccer_belgium_first_div",
    "Super Lig":            "soccer_turkey_super_league",
    "UEFA Champions League":"soccer_uefa_champs_league",
    "UEFA Europa League":   "soccer_uefa_europa_league",
    "UEFA Conference League":"soccer_uefa_europa_conference_league",
    "MLS":                  "soccer_usa_mls",
    "Major League Soccer":  "soccer_usa_mls",
    "Brazil Serie A":       "soccer_brazil_campeonato",
    "Argentina Primera Division": "soccer_argentina_primera_division",
}

# Markets we care about and their Odds API equivalents
MARKET_MAP = {
    "Over/Under 1.5 Goals":  ("totals", "1.5"),
    "Over/Under 2.5 Goals":  ("totals", "2.5"),
    "Over/Under 3.5 Goals":  ("totals", "3.5"),
    "Over/Under 4.5 Goals":  ("totals", "4.5"),
    "BTTS Yes":              ("btts",   "Yes"),
    "BTTS No":               ("btts",   "No"),
}

# Preferred bookmakers (highest reliability)
PREFERRED_BOOKMAKERS = [
    "bet365", "williamhill", "betway", "unibet",
    "pinnacle", "draftkings", "betfair_ex_eu",
    "1xbet", "betano", "marathonbet",
]


class OddsFetcher:
    """Fetches real-time odds from The Odds API."""

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self):
        self.api_key = os.getenv("ODDS_API_KEY", "")
        self._cache: Dict[str, Dict] = {}

    def _get_sport_key(self, league_name: str) -> Optional[str]:
        """Find the Odds API sport key for a given league name."""
        league_lower = league_name.lower()
        for our_name, sport_key in LEAGUE_MAP.items():
            if our_name.lower() in league_lower or league_lower in our_name.lower():
                return sport_key
        return None

    def get_odds_for_match(self, home: str, away: str, league: str) -> Dict:
        """
        Fetch real odds for a specific match.
        Returns odds for all allowed markets.
        """
        if not self.api_key:
            return {}

        sport_key = self._get_sport_key(league)
        if not sport_key:
            return {}

        # Check cache first
        cache_key = f"{sport_key}_{home}_{away}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            r = requests.get(
                f"{self.BASE_URL}/sports/{sport_key}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": "uk,eu,us",
                    "markets": "totals,btts",
                    "oddsFormat": "decimal",
                    "bookmakers": ",".join(PREFERRED_BOOKMAKERS[:5]),
                },
                timeout=8,
            )

            if r.status_code != 200:
                print(f"[OddsFetcher] HTTP {r.status_code} for {sport_key}")
                return {}

            # Log remaining requests
            remaining = r.headers.get("x-requests-remaining", "?")
            print(f"[OddsFetcher] Odds API requests remaining: {remaining}")

            events = r.json()
            result = self._find_match_odds(events, home, away)
            if result:
                self._cache[cache_key] = result
            return result

        except Exception as e:
            print(f"[OddsFetcher] Error: {e}")
            return {}

    def _norm(self, s: str) -> str:
        """Normalise team name for fuzzy comparison."""
        return (s.lower()
            .replace("fc", "").replace("sc", "").replace("ac", "")
            .replace("afc", "").replace("cf", "").replace("city", "")
            .replace("united", "utd").replace("hotspur", "")
            .replace("&", "and").replace(".", "")
            .replace("-", " ").strip()
            .replace("  ", " ").replace(" ", ""))

    def _teams_match(self, name1: str, name2: str) -> bool:
        """Check if two team names refer to the same team."""
        n1 = self._norm(name1)
        n2 = self._norm(name2)
        if n1 == n2:
            return True
        # One contains the other
        if n1 in n2 or n2 in n1:
            return True
        # First 5 chars match
        if len(n1) >= 5 and len(n2) >= 5 and n1[:5] == n2[:5]:
            return True
        # Check word overlap
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())
        # Remove common words
        common_words = {"fc", "sc", "ac", "cf", "city", "united", "utd", "the", "de", "of"}
        words1 -= common_words
        words2 -= common_words
        if words1 and words2:
            overlap = words1 & words2
            if len(overlap) >= 1 and max(len(w) for w in overlap) >= 4:
                return True
        return False

    def _find_match_odds(self, events: List[Dict], home: str, away: str) -> Dict:
        """Find matching event using flexible name matching."""
        for event in events:
            ev_home = event.get("home_team", "")
            ev_away = event.get("away_team", "")

            home_match = self._teams_match(home, ev_home)
            away_match = self._teams_match(away, ev_away)

            if not (home_match and away_match):
                continue

            # Extract odds from bookmakers
            odds_data = {}
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    market_key = market.get("key", "")

                    if market_key == "totals":
                        for outcome in market.get("outcomes", []):
                            point = str(outcome.get("point", ""))
                            name = outcome.get("name", "")
                            price = outcome.get("price", 0)
                            key = f"{name} {point} Goals"
                            # Keep best (lowest) odds across bookmakers
                            if key not in odds_data or price < odds_data[key]:
                                odds_data[key] = round(price, 2)

                    elif market_key == "btts":
                        for outcome in market.get("outcomes", []):
                            name = outcome.get("name", "")
                            price = outcome.get("price", 0)
                            key = f"BTTS {name}"
                            if key not in odds_data or price < odds_data[key]:
                                odds_data[key] = round(price, 2)

            if odds_data:
                print(f"[OddsFetcher] Matched: {home} vs {away} -> {ev_home} vs {ev_away}")
                print(f"[OddsFetcher] Markets found: {list(odds_data.keys())}")
                return {
                    "home": ev_home,
                    "away": ev_away,
                    "commence_time": event.get("commence_time"),
                    "odds": odds_data,
                }

        return {}

    def get_best_odds(self, pick: str, odds_data: Dict) -> Optional[float]:
        """
        Find real bookmaker odds for a given pick.
        e.g. "Over 2.5 Goals" -> looks for "Over 2.5 Goals" in odds_data
        """
        if not odds_data or not odds_data.get("odds"):
            return None

        pick_lower = pick.lower().strip()
        available = odds_data.get("odds", {})

        # Try exact match first
        for market_name, price in available.items():
            if market_name.lower() == pick_lower:
                return price

        # Try: "Over 2.5 Goals" matches "Over 2.5 Goals"
        for market_name, price in available.items():
            m_lower = market_name.lower()
            if pick_lower in m_lower or m_lower in pick_lower:
                return price

        # Try extracting number: "Over 2.5" from "Over 2.5 Goals"
        import re
        pick_num = re.search(r'(\d+\.?\d*)', pick)
        pick_dir = "over" if "over" in pick_lower else "under" if "under" in pick_lower else ""
        if pick_num and pick_dir:
            num = pick_num.group(1)
            for market_name, price in available.items():
                m_lower = market_name.lower()
                if pick_dir in m_lower and num in m_lower:
                    return price

        # BTTS matching
        if "btts" in pick_lower or "both teams" in pick_lower:
            yes_no = "yes" if "yes" in pick_lower else "no" if "no" in pick_lower else ""
            for market_name, price in available.items():
                if "btts" in market_name.lower() and yes_no in market_name.lower():
                    return price

        return None

    def enrich_tips_with_odds(self, tips: List[Dict]) -> List[Dict]:
        """
        Add real bookmaker odds to each tip.
        Replaces estimated odds_range with real odds.
        """
        for tip in tips:
            home, away = self._split_match(tip.get("match", ""))
            league = tip.get("league", "")

            if not home or not away:
                continue

            odds_data = self.get_odds_for_match(home, away, league)
            if not odds_data:
                tip["real_odds_available"] = False
                continue

            real_odds = self.get_best_odds(tip.get("pick", ""), odds_data)
            if real_odds:
                tip["bookmaker_odds"] = real_odds
                tip["real_odds_available"] = True
                tip["odds_source"] = "The Odds API (Live)"
                # Recalculate value with real odds
                prob = tip.get("predicted_probability", 0)
                if prob and real_odds:
                    value = (prob * real_odds) - 1
                    tip["value"] = round(value, 4)
                    tip["edge_pct"] = f"+{value*100:.1f}%" if value > 0 else f"{value*100:.1f}%"
                    tip["is_value_bet"] = value >= 0.05
            else:
                tip["real_odds_available"] = False

        return tips

    def _split_match(self, match_str: str):
        """Split 'Team A vs Team B' into (home, away)."""
        parts = match_str.split(" vs ")
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        return None, None
