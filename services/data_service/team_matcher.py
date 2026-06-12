"""
services/data_service/team_matcher.py

The live odds feed and the prediction model speak slightly different dialects
of the same names:

    Feed (The Odds API)      Model (openfootball / martj42)
    ----------------------   ------------------------------
    Manchester United        Manchester United FC
    Wolves                   Wolverhampton Wanderers FC
    South Korea              South Korea          (sometimes "Korea Republic")
    Türkiye / Turkiye        Turkey
    Czechia                  Czech Republic

This module bridges them. Matching is always done WITHIN a league, so an
English club is only ever compared to other English clubs, and a national
team only to national teams.

Resolution chain, cheapest and safest first:
  1. exact        — the feed name is already a model name
  2. alias        — we matched this name before; answer saved in team_aliases
  3. norm         — equal after normalising (accents, punctuation, FC/AFC...)
  4. seed         — hand-checked international spellings (USA, Czechia, ...)
  5. subset       — feed tokens are a subset of exactly ONE candidate
  6. fuzzy        — difflib ratio >= 0.86, best candidate, unique winner

Every non-exact hit is written to the team_aliases table, so the next lookup
for that name is a single indexed read. Misses return (None, "miss") and are
logged so they can be inspected at /api/live/unmatched.
"""

from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from difflib import SequenceMatcher

DB_PATH = os.getenv("FOOTBALL_DB", "database/football.db")

# --------------------------------------------------------------------------- #
#  Normalisation (proven on real Premier League names in earlier testing)     #
# --------------------------------------------------------------------------- #

# Club-type tokens to drop (English + common European forms)
_SUFFIX = re.compile(
    r"\b(fc|afc|cf|sc|ac|as|us|usl|sd|cd|ca|rc|sv|ssc|ssd|fk|bk|if|"
    r"cfc|rcd|ud|club|calcio|spvgg|tsg|vfb|vfl|bv|sg|sk)\b"
)

# Token-level rewrites applied during normalisation
_TOKEN_ALIASES = {
    "utd": "united",
    "munich": "munchen",            # Bayern Munich -> Bayern Munchen
    "munchengladbach": "monchengladbach",
    "st": "saint",
}

# Whole-name aliases (normalised feed name -> normalised target) for club
# shortenings that token logic can't reach.
_WHOLE_ALIASES = {
    # England
    "wolves": "wolverhampton wanderers",
    "spurs": "tottenham hotspur",
    "man city": "manchester city",
    "man united": "manchester united",
    "west brom": "west bromwich albion",
    "nottm forest": "nottingham forest",
    # Spain
    "atletico madrid": "atletico de madrid",
    "athletic club": "athletic bilbao",
    "betis": "real betis",
    # Italy
    "inter": "internazionale milano",
    "inter milan": "internazionale milano",
    "ac milan": "milan",
    # Germany
    "bayern munich": "bayern munchen",
    "leverkusen": "bayer leverkusen",
    "dortmund": "borussia dortmund",
    # France
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
}

# International seeds: normalised feed spelling -> candidate model names, in
# preference order. The first one actually present in the league's team list
# wins, so these stay safe even if the dataset's spelling shifts one day.
_INTL_SEEDS: dict[str, list[str]] = {
    "usa": ["United States"],
    "united states": ["United States"],
    "korea republic": ["South Korea", "Korea Republic"],
    "south korea": ["South Korea", "Korea Republic"],
    "korea dpr": ["North Korea", "Korea DPR"],
    "north korea": ["North Korea", "Korea DPR"],
    "czechia": ["Czech Republic", "Czechia"],
    "czech republic": ["Czech Republic", "Czechia"],
    "turkiye": ["Turkey", "Turkiye"],
    "turkey": ["Turkey", "Turkiye"],
    "ir iran": ["Iran"],
    "iran": ["Iran"],
    "cote divoire": ["Ivory Coast"],
    "ivory coast": ["Ivory Coast"],
    "dr congo": ["DR Congo", "Congo DR"],
    "congo dr": ["DR Congo", "Congo DR"],
    "cape verde": ["Cape Verde", "Cape Verde Islands"],
    "cabo verde": ["Cape Verde", "Cape Verde Islands"],
    "curacao": ["Curacao"],          # accents already stripped by normalize
    "china pr": ["China PR", "China"],
    "china": ["China PR", "China"],
    "bosnia and herzegovina": ["Bosnia and Herzegovina", "Bosnia-Herzegovina"],
    "bosnia herzegovina": ["Bosnia and Herzegovina", "Bosnia-Herzegovina"],
    "bosnia": ["Bosnia and Herzegovina"],
    "uae": ["United Arab Emirates"],
    "united arab emirates": ["United Arab Emirates"],
    "north macedonia": ["North Macedonia", "Macedonia"],
    "republic of ireland": ["Republic of Ireland", "Ireland"],
    "ireland": ["Republic of Ireland", "Ireland"],
    "saudi arabia": ["Saudi Arabia"],
    "new zealand": ["New Zealand"],
    "trinidad and tobago": ["Trinidad and Tobago"],
}


def normalize(name: str) -> str:
    """Lowercase, strip accents/punctuation/club-suffixes, apply token aliases."""
    if not name:
        return ""
    s = name.lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.replace("&", " and ").replace("-", " ").replace(".", " ").replace("'", "")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = [_TOKEN_ALIASES.get(t, t) for t in s.split()]
    s = " ".join(toks)
    stripped = _SUFFIX.sub(" ", s)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped or s


# --------------------------------------------------------------------------- #
#  Matcher with per-league candidate cache + alias persistence                #
# --------------------------------------------------------------------------- #

class TeamMatcher:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DB_PATH
        self._cands: dict[int, list[str]] = {}          # league -> model names
        self._norm_map: dict[int, dict[str, str]] = {}  # league -> norm -> name

    # ---- infrastructure ---------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute(
            """CREATE TABLE IF NOT EXISTS team_aliases (
                 src_name   TEXT NOT NULL,
                 league_id  INTEGER NOT NULL,
                 model_name TEXT NOT NULL,
                 method     TEXT,
                 PRIMARY KEY (src_name, league_id)
               )"""
        )
        return con

    def clear_cache(self) -> None:
        """Forget cached team lists (call after new results are inserted)."""
        self._cands.clear()
        self._norm_map.clear()

    def candidates(self, league_id: int) -> list[str]:
        if league_id not in self._cands:
            con = self._conn()
            try:
                rows = con.execute(
                    "SELECT DISTINCT home_team FROM fixtures WHERE league_id=? "
                    "UNION SELECT DISTINCT away_team FROM fixtures WHERE league_id=?",
                    (league_id, league_id),
                ).fetchall()
            finally:
                con.close()
            names = sorted({r[0] for r in rows if r[0]})
            self._cands[league_id] = names
            nm: dict[str, str] = {}
            for n in names:
                nm.setdefault(normalize(n), n)   # first wins on rare collisions
            self._norm_map[league_id] = nm
        return self._cands[league_id]

    def _load_alias(self, src: str, league_id: int) -> str | None:
        con = self._conn()
        try:
            row = con.execute(
                "SELECT model_name FROM team_aliases WHERE src_name=? AND league_id=?",
                (src, league_id),
            ).fetchone()
        finally:
            con.close()
        return row[0] if row else None

    def _save_alias(self, src: str, league_id: int, model: str, method: str) -> None:
        try:
            con = self._conn()
            con.execute(
                "INSERT OR REPLACE INTO team_aliases (src_name, league_id, model_name, method) "
                "VALUES (?,?,?,?)",
                (src, league_id, model, method),
            )
            con.commit()
            con.close()
        except Exception:
            pass  # alias cache is an optimisation, never a hard failure

    # ---- the resolution chain ----------------------------------------------
    def resolve(self, src_name: str, league_id: int) -> tuple[str | None, str]:
        """Map a feed name to a model name. Returns (model_name|None, method)."""
        if not src_name:
            return None, "miss"
        cands = self.candidates(league_id)
        cset = set(cands)

        # 1. exact
        if src_name in cset:
            return src_name, "exact"

        # 2. learned alias
        hit = self._load_alias(src_name, league_id)
        if hit and hit in cset:
            return hit, "alias"

        norm = normalize(src_name)
        norm = _WHOLE_ALIASES.get(norm, norm)
        nmap = self._norm_map[league_id]

        # 3. equal after normalisation
        if norm in nmap:
            model = nmap[norm]
            self._save_alias(src_name, league_id, model, "norm")
            return model, "norm"

        # 4. international seeds
        for cand in _INTL_SEEDS.get(norm, []):
            if cand in cset:
                self._save_alias(src_name, league_id, cand, "seed")
                return cand, "seed"
            cn = normalize(cand)
            if cn in nmap:
                model = nmap[cn]
                self._save_alias(src_name, league_id, model, "seed")
                return model, "seed"

        # 5. token-subset — FORWARD only (feed name inside candidate name),
        #    unique winner, anchored by a real token. Reverse subset is banned:
        #    it let "Inter Milan" wrongly claim "AC Milan" (milan ⊆ inter milan).
        stoks = set(norm.split())
        if stoks and any(len(t) >= 4 for t in stoks):
            winners = sorted(
                {name for cn, name in nmap.items() if stoks <= set(cn.split())}
            )
            if len(winners) == 1:
                self._save_alias(src_name, league_id, winners[0], "subset")
                return winners[0], "subset"

        # 6. fuzzy, high floor, unique best
        best, second, best_name = 0.0, 0.0, None
        for cn, name in nmap.items():
            r = SequenceMatcher(None, norm, cn).ratio()
            if r > best:
                second, best, best_name = best, r, name
            elif r > second:
                second = r
        if best >= 0.86 and best_name and (best - second) >= 0.02:
            self._save_alias(src_name, league_id, best_name, "fuzzy")
            return best_name, "fuzzy"

        return None, "miss"


# Module-level singleton for everyday use ------------------------------------
_matcher: TeamMatcher | None = None


def get_matcher() -> TeamMatcher:
    global _matcher
    if _matcher is None:
        _matcher = TeamMatcher()
    return _matcher


def resolve(src_name: str, league_id: int) -> tuple[str | None, str]:
    return get_matcher().resolve(src_name, league_id)


def clear_cache() -> None:
    if _matcher is not None:
        _matcher.clear_cache()
