"""
xg_fetcher.py — Expected-goals (xG) adapter and the xG → model integration.

WHY xG MATTERS
--------------
Goals are a noisy signal. A side that creates 2.1 xG but scores once was
unlucky, not bad — and over the next few games it will tend to score more.
xG measures the *quality of chances created and conceded*, so it stabilises
faster than raw goals and is a better forward predictor of scoring. Feeding
xG into the Dixon-Coles core therefore sharpens every goals market when the
recent scoreline flatters or flatters-to-deceive a team.

HOW IT PLUGS IN  (this is the important part)
---------------------------------------------
The Dixon-Coles model already produces expected goals (lambda_home,
lambda_away) for a fixture from team ratings fitted on actual results. This
adapter produces a *second* estimate of those same lambdas from xG profiles
(a classic attack/defence ratio model, but on xG instead of goals). We then
blend the two and push the blended pair straight back through the model:

    lam_dc_h, lam_dc_a   = model.expected_goals(home, away)        # goals-fit
    lam_xg_h, lam_xg_a   = xg_lambdas(home, away, source)          # xG-fit
    lam_h = w*lam_dc_h + (1-w)*lam_xg_h
    lam_a = w*lam_dc_a + (1-w)*lam_xg_a
    markets = model.predict_markets_from_lambdas(home, away, lam_h, lam_a)

Because rho (low-score dependence) still comes from the fitted model, the
blended output stays coherent with the rest of the system — 1X2, O/U and BTTS
all derive from one score matrix, as before.

HONEST CAVEATS  (read before relying on this in production)
-----------------------------------------------------------
* There is no official, free xG API. Practical sources:
    - API-Football: some fixtures expose xG in the statistics endpoint.
    - football-data.co.uk: free CSVs with xG for several leagues.
    - Understat / FBref (StatsBomb): rich xG, but scraping is fragile and may
      breach ToS; a paid Opta/StatsBomb feed is the clean route.
  This module does NOT scrape anything. You wire a source (see XGSource); the
  adapter only consumes it. That keeps the legal/operational choice yours.
* xG model quality varies by provider. Garbage in, garbage out.
* Sample size: xG is steadier than goals but still wants ~6-10 recent matches
  per team. Profiles below `min_matches` are regressed toward the league mean.
* Prefer ROLLING xG (last N matches) over season-to-date so the blend reacts
  to current form — but whatever your source returns is what gets used.

Dependencies: standard library + numpy (only for the model hand-off).
No network calls live here.

Author: Rollover Betting AI · Tier-3 enrichment
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Protocol

# Imported lazily inside functions where possible, but the type is handy here.
from services.ml_service.dixon_coles import DixonColesModel, MarketProbabilities


# --------------------------------------------------------------------------- #
#  Data model                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class TeamXGProfile:
    """
    A team's recent xG production and concession.

    Rates are PER MATCH. If your source gives season totals, divide by matches
    before constructing this (or use `from_totals`).
    """
    team: str
    matches: int
    xg_for_per_game: float       # attacking output (xG created)
    xga_per_game: float          # defensive leakage (xG conceded)

    @classmethod
    def from_totals(
        cls, team: str, matches: int, xg_for_total: float, xga_total: float
    ) -> "TeamXGProfile":
        m = max(matches, 1)
        return cls(
            team=team,
            matches=matches,
            xg_for_per_game=xg_for_total / m,
            xga_per_game=xga_total / m,
        )


@dataclass
class XGBaseline:
    """League-wide average goals per team per match (the model's anchor)."""
    avg_home_goals: float = 1.50
    avg_away_goals: float = 1.15

    @property
    def avg_goals(self) -> float:
        return 0.5 * (self.avg_home_goals + self.avg_away_goals)


# --------------------------------------------------------------------------- #
#  Source interface — you implement / populate one of these                   #
# --------------------------------------------------------------------------- #
class XGSource(Protocol):
    """Minimal contract the blend needs from whatever feeds you wire."""

    def get_profile(self, team: str) -> TeamXGProfile | None: ...
    def baseline(self) -> XGBaseline: ...


class InMemoryXGSource:
    """
    Dead-simple source backed by a dict. Populate it from your scraper / CSV /
    API pull, then hand it to the blend functions. Good for tests and for a
    once-a-day refresh held in process memory.
    """

    def __init__(
        self,
        profiles: dict[str, TeamXGProfile] | None = None,
        baseline: XGBaseline | None = None,
    ) -> None:
        self._profiles = profiles or {}
        self._baseline = baseline or XGBaseline()

    def set(self, profile: TeamXGProfile) -> None:
        self._profiles[profile.team] = profile

    def get_profile(self, team: str) -> TeamXGProfile | None:
        return self._profiles.get(team)

    def baseline(self) -> XGBaseline:
        return self._baseline


class SQLiteXGSource:
    """
    Source backed by a `team_xg` table. Your data job (scraper / API pull)
    writes rows here; the blend reads them. Schema is created on first use.

        team_xg(team TEXT PRIMARY KEY,
                matches INTEGER,
                xg_for_total REAL,
                xga_total REAL,
                updated_at TEXT)

    Stored as TOTALS so re-aggregation is trivial; per-game rates are computed
    on read.
    """

    def __init__(
        self,
        db_path: str = "database/football.db",
        baseline: XGBaseline | None = None,
    ) -> None:
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._baseline = baseline or XGBaseline()
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS team_xg (
                    team          TEXT PRIMARY KEY,
                    matches       INTEGER NOT NULL,
                    xg_for_total  REAL    NOT NULL,
                    xga_total     REAL    NOT NULL,
                    updated_at    TEXT
                )
                """
            )

    def upsert_profile(
        self,
        team: str,
        matches: int,
        xg_for_total: float,
        xga_total: float,
        updated_at: str | None = None,
    ) -> None:
        """Called by your data job. Upserts one team's rolling xG totals."""
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO team_xg (team, matches, xg_for_total, xga_total, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(team) DO UPDATE SET
                    matches=excluded.matches,
                    xg_for_total=excluded.xg_for_total,
                    xga_total=excluded.xga_total,
                    updated_at=excluded.updated_at
                """,
                (team, matches, xg_for_total, xga_total, updated_at),
            )

    def get_profile(self, team: str) -> TeamXGProfile | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM team_xg WHERE team = ?", (team,)
            ).fetchone()
        if row is None:
            return None
        return TeamXGProfile.from_totals(
            team=row["team"],
            matches=int(row["matches"]),
            xg_for_total=float(row["xg_for_total"]),
            xga_total=float(row["xga_total"]),
        )

    def baseline(self) -> XGBaseline:
        return self._baseline


# --------------------------------------------------------------------------- #
#  xG ratio model -> expected goals                                           #
# --------------------------------------------------------------------------- #
def _regress(rate: float, anchor: float, matches: int, min_matches: int) -> float:
    """
    Shrink a per-game xG rate toward the league anchor when the sample is thin.

    weight = matches / (matches + min_matches): a team with exactly min_matches
    games sits halfway between its own rate and the league average; more games
    -> trust the team's own number more.
    """
    if matches <= 0:
        return anchor
    w = matches / (matches + float(min_matches))
    return w * rate + (1.0 - w) * anchor


def xg_lambdas(
    home_team: str,
    away_team: str,
    source: XGSource,
    min_matches: int = 6,
) -> tuple[float, float] | None:
    """
    Expected goals for the fixture, derived purely from xG profiles.

    Classic attack x defence x league-average ratio model:

        home_attack  = home.xg_for  / league_avg
        away_defence = away.xga     / league_avg
        lam_h        = avg_home_goals * home_attack * away_defence

    Returns None if either team has no profile (caller then falls back to the
    pure Dixon-Coles lambdas).
    """
    hp = source.get_profile(home_team)
    ap = source.get_profile(away_team)
    if hp is None or ap is None:
        return None

    base = source.baseline()
    anchor = base.avg_goals

    h_for = _regress(hp.xg_for_per_game, anchor, hp.matches, min_matches)
    h_against = _regress(hp.xga_per_game, anchor, hp.matches, min_matches)
    a_for = _regress(ap.xg_for_per_game, anchor, ap.matches, min_matches)
    a_against = _regress(ap.xga_per_game, anchor, ap.matches, min_matches)

    home_attack = h_for / anchor
    home_defence = h_against / anchor
    away_attack = a_for / anchor
    away_defence = a_against / anchor

    lam_h = base.avg_home_goals * home_attack * away_defence
    lam_a = base.avg_away_goals * away_attack * home_defence

    # Guard against absurd values from a bad feed.
    lam_h = float(min(max(lam_h, 0.05), 6.0))
    lam_a = float(min(max(lam_a, 0.05), 6.0))
    return lam_h, lam_a


def blend_lambdas(
    dc: tuple[float, float],
    xg: tuple[float, float],
    weight_dc: float = 0.6,
) -> tuple[float, float]:
    """
    Convex blend of the goals-fit and xG-fit lambdas.

    weight_dc in [0, 1]. Default 0.6 leans on Dixon-Coles because the fitted
    model already accounts for strength of schedule and time-decay, which the
    simple xG ratio does not — xG is used to *nudge*, not overrule. Push it
    toward 0.5 if you trust your xG feed and want it to bite harder.
    """
    w = float(min(max(weight_dc, 0.0), 1.0))
    lam_h = w * dc[0] + (1.0 - w) * xg[0]
    lam_a = w * dc[1] + (1.0 - w) * xg[1]
    return lam_h, lam_a


# --------------------------------------------------------------------------- #
#  Headline integration: xG-adjusted markets                                  #
# --------------------------------------------------------------------------- #
def xg_adjusted_markets(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
    source: XGSource,
    weight_dc: float = 0.6,
    min_matches: int = 6,
    top_n_scores: int = 5,
) -> tuple[MarketProbabilities, dict]:
    """
    Produce goals markets with xG folded into the expected goals, plus a
    diagnostic dict that lets the 'why this tip' view show what xG changed.

    Falls back gracefully to pure Dixon-Coles markets when xG is unavailable
    for either team; the diagnostic records that it did so.
    """
    lam_dc = model.expected_goals(home_team, away_team)
    lam_xg = xg_lambdas(home_team, away_team, source, min_matches=min_matches)

    if lam_xg is None:
        markets = model.predict_markets(home_team, away_team, top_n_scores=top_n_scores)
        diagnostic = {
            "xg_applied": False,
            "reason": "no xG profile for one or both teams",
            "lambda_dc": {"home": round(lam_dc[0], 3), "away": round(lam_dc[1], 3)},
            "lambda_used": {"home": round(lam_dc[0], 3), "away": round(lam_dc[1], 3)},
        }
        return markets, diagnostic

    lam_blend = blend_lambdas(lam_dc, lam_xg, weight_dc=weight_dc)
    markets = model.predict_markets_from_lambdas(
        home_team, away_team, lam_blend[0], lam_blend[1], top_n_scores=top_n_scores
    )

    diagnostic = {
        "xg_applied": True,
        "weight_dc": round(float(weight_dc), 2),
        "lambda_dc": {"home": round(lam_dc[0], 3), "away": round(lam_dc[1], 3)},
        "lambda_xg": {"home": round(lam_xg[0], 3), "away": round(lam_xg[1], 3)},
        "lambda_used": {"home": round(lam_blend[0], 3), "away": round(lam_blend[1], 3)},
        "total_goals_shift": round(
            (lam_blend[0] + lam_blend[1]) - (lam_dc[0] + lam_dc[1]), 3
        ),
    }
    return markets, diagnostic


__all__ = [
    "TeamXGProfile",
    "XGBaseline",
    "XGSource",
    "InMemoryXGSource",
    "SQLiteXGSource",
    "xg_lambdas",
    "blend_lambdas",
    "xg_adjusted_markets",
]
