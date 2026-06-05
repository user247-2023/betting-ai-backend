"""
odds_fetcher.py — Fetches real bookmaker odds from The Odds API
Free tier: 500 requests/month

FIXES (v2):
  1. markets="totals" only on the bulk endpoint (btts caused HTTP 422 —
     it is an "additional market" available only on the per-event endpoint).
  2. League-level caching — one API call per league per cycle covers ALL
     matches in that league, instead of one call per match (≈10-15x quota saving).
  3. Detailed error logging — captures the response body on non-200 so any
     future issue is diagnosable from the logs.
  4. Optional per-event BTTS enrichment (off by default to protect quota;
     enable with ODDS_FETCH_BTTS=true).
"""
import os
import time
import requests
from typing import Dict, List, Optional


# Map our league names to The Odds API sport keys
LEAGUE_MAP = {
    "Premier League":        "soccer_epl",
    "Championship":          "soccer_england_championship",
    "League One":            "soccer_england_league1",
    "League Two":            "soccer_england_league2",
    "La Liga":               "soccer_spain_la_liga",
    "La Liga 2":             "soccer_spain_segunda_division",
    "Serie A":               "soccer_italy_serie_a",
    "Serie B":               "soccer_italy_serie_b",
    "Bundesliga":            "soccer_germany_bundesliga",
    "2. Bundesliga":         "soccer_germany_bundesliga2",
    "Ligue 1":               "soccer_france_ligue_one",
    "Ligue 2":               "soccer_france_ligue_two",
    "Eredivisie":            "soccer_netherlands_eredivisie",
    "Primeira Liga":         "soccer_portugal_primeira_liga",
    "Scottish Premiership":  "soccer_scotland_premiership",
    "Belgian Pro League":    "soccer_belgium_first_div",
    "Super Lig":             "soccer_turkey_super_league",
    "UEFA Champions League": "soccer_uefa_champs_league",
    "UEFA Europa League":    "soccer_uefa_europa_league",
    "UEFA Conference League":"soccer_uefa_europa_conference_league",
    "MLS":                   "soccer_usa_mls",
    "Major League Soccer":   "soccer_usa_mls",
    "Brazil Serie A":        "soccer_brazil_campeonato",
    "Argentina Primera Division": "soccer_argentina_primera_division",
    "World Cup":             "soccer_fifa_world_cup",
    "FIFA World Cup":        "soccer_fifa_world_cup",
}

# Core bulk market — ALWAYS valid on /sports/{key}/odds.
# (h2h + totals are the only markets guaranteed on the bulk endpoint for soccer.)
BULK_MARKETS = "totals"

# Preferred bookmakers (highest reliability)
PREFERRED_BOOKMAKERS = [
    "bet365", "williamhill", "betway", "unibet",
    "pinnacle", "draftkings", "betfair_ex_eu",
    "1xbet", "betano", "marathonbet",
]

# Cache lifetime for a league's odds snapshot (seconds)
LEAGUE_CACHE_TTL = int(os.getenv("ODDS_CACHE_TTL", "900"))  # 15 min


class OddsFetcher:
    """Fetches real-time odds from The Odds API (quota-efficient)."""

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self):
        self.api_key = os.getenv("ODDS_API_KEY", "")
        self.fetch_btts = os.getenv("ODDS_FETCH_BTTS", "false").lower() == "true"
        # Cache keyed by sport_key -> {"ts": epoch, "events": [...]}
        self._league_cache: Dict[str, Dict] = {}
        # Track quota from the last response
        self.requests_remaining: Optional[str] = None

    # ── League key resolution ──────────────────────────────────────
    def _get_sport_key(self, league_name: str) -> Optional[str]:
        league_lower = (league_name or "").lower()
        for our_name, sport_key in LEAGUE_MAP.items():
            if our_name.lower() in league_lower or league_lower in our_name.lower():
                return sport_key
        return None

    # ── Bulk league fetch (cached) ─────────────────────────────────
    def _fetch_league_events(self, sport_key: str) -> List[Dict]:
        """
        Fetch ALL events+odds for a league in ONE request, cached for
        LEAGUE_CACHE_TTL. This is the core quota-saving change: 10 matches
        in the same league now cost 1 request, not 10.
        """
        now = time.time()
        cached = self._league_cache.get(sport_key)
        if cached and (now - cached["ts"]) < LEAGUE_CACHE_TTL:
            return cached["events"]

        try:
            r = requests.get(
                f"{self.BASE_URL}/sports/{sport_key}/odds",
                params={
                    "apiKey":     self.api_key,
                    "regions":    "uk,eu",
                    "markets":    BULK_MARKETS,        # ← FIX: totals only, no btts
                    "oddsFormat": "decimal",
                },
                timeout=10,
            )

            self.requests_remaining = r.headers.get("x-requests-remaining", "?")

            if r.status_code != 200:
                # FIX: log the actual body so any future error is diagnosable
                print(f"[OddsFetcher] HTTP {r.status_code} for {sport_key} "
                      f"| body: {r.text[:200]} | remaining: {self.requests_remaining}")
                # Cache an empty result briefly to avoid hammering on errors
                self._league_cache[sport_key] = {"ts": now, "events": []}
                return []

            events = r.json()
            print(f"[OddsFetcher] {sport_key}: {len(events)} events "
                  f"| quota remaining: {self.requests_remaining}")
            self._league_cache[sport_key] = {"ts": now, "events": events}
            return events

        except Exception as e:
            print(f"[OddsFetcher] Error fetching {sport_key}: {type(e).__name__}: {e}")
            return []

    # ── Per-event BTTS (optional, quota-expensive) ─────────────────
    def _fetch_event_btts(self, sport_key: str, event_id: str) -> Dict[str, float]:
        """
        Fetch BTTS for a single event via the per-event endpoint.
        Only used when ODDS_FETCH_BTTS=true (1 request per event — costly).
        """
        if not self.fetch_btts or not event_id:
            return {}
        try:
            r = requests.get(
                f"{self.BASE_URL}/sports/{sport_key}/events/{event_id}/odds",
                params={
                    "apiKey":     self.api_key,
                    "regions":    "uk,eu",
                    "markets":    "btts",
                    "oddsFormat": "decimal",
                },
                timeout=10,
            )
            self.requests_remaining = r.headers.get("x-requests-remaining", self.requests_remaining)
            if r.status_code != 200:
                return {}
            data = r.json()
            out: Dict[str, float] = {}
            for bk in data.get("bookmakers", []):
                for m in bk.get("markets", []):
                    if m.get("key") == "btts":
                        for o in m.get("outcomes", []):
                            name = o.get("name", "")      # "Yes" / "No"
                            price = o.get("price", 0)
                            key = f"BTTS {name}"
                            if key not in out or price < out[key]:
                                out[key] = round(price, 2)
            return out
        except Exception:
            return {}

    # ── Team-name matching ─────────────────────────────────────────
    GENERIC_WORDS = {
        "city", "united", "town", "real", "club", "sporting", "athletic",
        "county", "rovers", "wanderers", "albion", "fc", "afc", "sc", "cf",
        "calcio", "deportivo", "atletico",
    }

    def _norm(self, s: str) -> str:
        """Lowercase and strip common trailing club suffixes (whole-word only,
        so names like 'Ascoli'/'Schalke' are NOT corrupted)."""
        s = (s or "").lower().replace(".", "").strip()
        for suffix in (" fc", " afc", " sc", " cf", " sd", " ac", " cd"):
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
        return s

    def _side_match(self, q: str, e: str) -> bool:
        """True if query team name `q` plausibly refers to event team name `e`."""
        nq, ne = self._norm(q), self._norm(e)
        qc, ec = nq.replace(" ", ""), ne.replace(" ", "")
        if not qc or not ec:
            return False
        # 1) direct containment ("Arsenal" ⊂ "Arsenal FC")
        if qc in ec or ec in qc:
            return True
        # 2) first-token prefix ("Man" → "Manchester")
        qt = nq.split()[0] if nq.split() else ""
        et = ne.split()[0] if ne.split() else ""
        if qt and et and min(len(qt), len(et)) >= 3 and (qt.startswith(et) or et.startswith(qt)):
            return True
        # 3) shared distinctive (non-generic) word ("Tottenham Hotspur" ↔ "Tottenham")
        wq = {w for w in nq.split() if len(w) >= 4 and w not in self.GENERIC_WORDS}
        we = {w for w in ne.split() if len(w) >= 4 and w not in self.GENERIC_WORDS}
        if wq & we:
            return True
        return False

    def _teams_match(self, qhome, qaway, ehome, eaway) -> bool:
        return self._side_match(qhome, ehome) and self._side_match(qaway, eaway)

    # ── Extract odds for one match from the cached league events ───
    def get_odds_for_match(self, home: str, away: str, league: str) -> Dict:
        if not self.api_key:
            return {}
        sport_key = self._get_sport_key(league)
        if not sport_key:
            return {}

        events = self._fetch_league_events(sport_key)
        if not events:
            return {}

        for event in events:
            if not self._teams_match(home, away,
                                     event.get("home_team", ""),
                                     event.get("away_team", "")):
                continue

            odds_data: Dict[str, float] = {}
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "totals":
                        for outcome in market.get("outcomes", []):
                            point = str(outcome.get("point", ""))
                            name  = outcome.get("name", "")   # "Over" / "Under"
                            price = outcome.get("price", 0)
                            key = f"{name} {point} Goals"
                            # keep best (lowest) price = most conservative
                            if key not in odds_data or price < odds_data[key]:
                                odds_data[key] = round(price, 2)

            # Optional BTTS enrichment (per-event, only if explicitly enabled)
            if self.fetch_btts:
                odds_data.update(self._fetch_event_btts(sport_key, event.get("id", "")))

            if odds_data:
                return {
                    "home":          event.get("home_team"),
                    "away":          event.get("away_team"),
                    "commence_time": event.get("commence_time"),
                    "odds":          odds_data,
                }

        return {}

    # ── Pick → odds resolution ─────────────────────────────────────
    def get_best_odds(self, pick: str, odds_data: Dict) -> Optional[float]:
        if not odds_data:
            return None
        pick_lower = (pick or "").lower()
        for market_name, price in odds_data.get("odds", {}).items():
            if market_name.lower() in pick_lower or pick_lower in market_name.lower():
                return price
        # partial token match (e.g. "over" + "2.5")
        for market_name, price in odds_data.get("odds", {}).items():
            parts = [p for p in pick_lower.split() if len(p) > 2]
            if parts and all(p in market_name.lower() for p in parts):
                return price
        return None

    # ── Enrich a batch of tips with real odds ──────────────────────
    def enrich_tips_with_odds(self, tips: List[Dict]) -> List[Dict]:
        for tip in tips:
            home, away = self._split_match(tip.get("match", ""))
            league = tip.get("league", "")
            if not home or not away:
                tip["real_odds_available"] = False
                continue

            odds_data = self.get_odds_for_match(home, away, league)
            if not odds_data:
                tip["real_odds_available"] = False
                continue

            real_odds = self.get_best_odds(tip.get("pick", ""), odds_data)
            if real_odds:
                tip["bookmaker_odds"]      = real_odds
                tip["real_odds_available"] = True
                tip["odds_source"]         = "The Odds API (Live)"
                prob = tip.get("predicted_probability", 0) or tip.get("predicted_prob", 0)
                if prob and real_odds:
                    value = (prob * real_odds) - 1
                    tip["value"]       = round(value, 4)
                    tip["edge_pct"]    = (f"+{value*100:.1f}%" if value > 0
                                          else f"{value*100:.1f}%")
                    tip["is_value_bet"] = value >= 0.05
            else:
                tip["real_odds_available"] = False

        return tips

    def _split_match(self, match_str: str):
        parts = (match_str or "").split(" vs ")
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        return None, None
