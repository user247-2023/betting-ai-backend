"""
referee_fetcher.py — Referee profile adapter and a Poisson cards model.

WHY A SEPARATE MODEL
--------------------
The Dixon-Coles core models GOALS. Cards are a different process driven mostly
by the referee plus match intensity, so they get their own small Poisson model
here. The big, exploitable signal in the cards market is the referee: strict
referees average well over 4 yellows a game, lenient ones under 3, and that gap
swamps most team effects.

    expected = expected_cards(referee_profile, home_profile, away_profile)
    markets  = cards_markets(expected)   # Over/Under 3.5 / 4.5 / 5.5 total cards

These plug into the same de-vig / value / staking machinery as goals markets:
de-vig the book's Over/Under cards prices, compare to `markets`, bet the edge.

DATA SOURCE  (you wire it)
--------------------------
No clean free referee API. Sources: football-data.co.uk CSVs (cards columns),
some stats sites, or manual aggregation per competition. `SQLiteRefereeSource`
gives you a table your data job writes to; `InMemoryRefereeSource` is for tests
and small hand-maintained sets.

HONEST CAVEATS
--------------
* Cards also depend on stakes (derbies, relegation six-pointers) and team
  discipline, which this baseline only partly captures via team card rates.
  Treat referee average as the anchor and team rates as a modest tilt.
* A yellow-card Poisson is an approximation: cards aren't perfectly Poisson
  (correlated bookings, second yellows). Good enough for O/U totals; be more
  cautious on exact-count or team-cards markets.
* Referee samples are small (a ref works maybe 20-40 games a season). Regress
  thin samples toward the league average — `expected_cards` does this.

Dependencies: standard library + scipy (Poisson pmf for the markets).

Author: Rollover Betting AI · Tier-3 enrichment
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.stats import poisson


# --------------------------------------------------------------------------- #
#  Data model                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class RefereeProfile:
    """A referee's average yellow cards shown per match."""
    name: str
    matches: int
    yellows_per_game: float
    reds_per_game: float = 0.0


@dataclass
class TeamCardProfile:
    """A team's cards-per-game tendency (cards its players receive)."""
    team: str
    matches: int
    cards_per_game: float    # yellows received per match (reds optional, folded in)


@dataclass
class CardsBaseline:
    """League-average total yellow cards per match (both teams combined)."""
    avg_total_yellows: float = 3.8


# --------------------------------------------------------------------------- #
#  Source interface                                                           #
# --------------------------------------------------------------------------- #
class RefereeSource(Protocol):
    def get_referee(self, name: str) -> RefereeProfile | None: ...
    def get_team_cards(self, team: str) -> TeamCardProfile | None: ...
    def baseline(self) -> CardsBaseline: ...


class InMemoryRefereeSource:
    def __init__(
        self,
        referees: dict[str, RefereeProfile] | None = None,
        teams: dict[str, TeamCardProfile] | None = None,
        baseline: CardsBaseline | None = None,
    ) -> None:
        self._refs = referees or {}
        self._teams = teams or {}
        self._baseline = baseline or CardsBaseline()

    def set_referee(self, ref: RefereeProfile) -> None:
        self._refs[ref.name] = ref

    def set_team(self, team: TeamCardProfile) -> None:
        self._teams[team.team] = team

    def get_referee(self, name: str) -> RefereeProfile | None:
        return self._refs.get(name)

    def get_team_cards(self, team: str) -> TeamCardProfile | None:
        return self._teams.get(team)

    def baseline(self) -> CardsBaseline:
        return self._baseline


class SQLiteRefereeSource:
    """
    Backed by two tables your data job populates:

        referee_cards(name TEXT PK, matches INT, yellows_per_game REAL,
                      reds_per_game REAL)
        team_cards(team TEXT PK, matches INT, cards_per_game REAL)
    """

    def __init__(
        self,
        db_path: str = "database/football.db",
        baseline: CardsBaseline | None = None,
    ) -> None:
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._baseline = baseline or CardsBaseline()
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS referee_cards (
                    name             TEXT PRIMARY KEY,
                    matches          INTEGER NOT NULL,
                    yellows_per_game REAL    NOT NULL,
                    reds_per_game    REAL    DEFAULT 0
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS team_cards (
                    team           TEXT PRIMARY KEY,
                    matches        INTEGER NOT NULL,
                    cards_per_game REAL    NOT NULL
                )
                """
            )

    def upsert_referee(
        self, name: str, matches: int, yellows_per_game: float, reds_per_game: float = 0.0
    ) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO referee_cards (name, matches, yellows_per_game, reds_per_game)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    matches=excluded.matches,
                    yellows_per_game=excluded.yellows_per_game,
                    reds_per_game=excluded.reds_per_game
                """,
                (name, matches, yellows_per_game, reds_per_game),
            )

    def upsert_team(self, team: str, matches: int, cards_per_game: float) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO team_cards (team, matches, cards_per_game)
                VALUES (?, ?, ?)
                ON CONFLICT(team) DO UPDATE SET
                    matches=excluded.matches,
                    cards_per_game=excluded.cards_per_game
                """,
                (team, matches, cards_per_game),
            )

    def get_referee(self, name: str) -> RefereeProfile | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM referee_cards WHERE name = ?", (name,)).fetchone()
        if row is None:
            return None
        return RefereeProfile(
            name=row["name"],
            matches=int(row["matches"]),
            yellows_per_game=float(row["yellows_per_game"]),
            reds_per_game=float(row["reds_per_game"] or 0.0),
        )

    def get_team_cards(self, team: str) -> TeamCardProfile | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM team_cards WHERE team = ?", (team,)).fetchone()
        if row is None:
            return None
        return TeamCardProfile(
            team=row["team"], matches=int(row["matches"]), cards_per_game=float(row["cards_per_game"])
        )

    def baseline(self) -> CardsBaseline:
        return self._baseline


# --------------------------------------------------------------------------- #
#  Cards model                                                                #
# --------------------------------------------------------------------------- #
def _regress(rate: float, anchor: float, matches: int, min_matches: int) -> float:
    if matches <= 0:
        return anchor
    w = matches / (matches + float(min_matches))
    return w * rate + (1.0 - w) * anchor


def expected_cards(
    referee: RefereeProfile | None,
    home: TeamCardProfile | None,
    away: TeamCardProfile | None,
    baseline: CardsBaseline,
    ref_min_matches: int = 10,
    team_min_matches: int = 8,
    ref_weight: float = 0.65,
) -> float:
    """
    Expected TOTAL yellow cards in the match (both teams).

    Anchor on the referee's average (regressed for small samples), then tilt by
    how card-prone the two teams are relative to the league. The referee gets
    the larger say (`ref_weight`) because it is the dominant, more stable driver.

        ref_mean   = regressed referee yellows/game
        team_total = (home cards/game + away cards/game), regressed
        expected   = ref_weight*ref_mean + (1-ref_weight)*team_total
    """
    league_total = baseline.avg_total_yellows
    league_team = league_total / 2.0  # per-team league average

    if referee is not None:
        ref_mean = _regress(referee.yellows_per_game, league_total, referee.matches, ref_min_matches)
    else:
        ref_mean = league_total

    h = _regress(home.cards_per_game, league_team, home.matches, team_min_matches) if home else league_team
    a = _regress(away.cards_per_game, league_team, away.matches, team_min_matches) if away else league_team
    team_total = h + a

    w = float(min(max(ref_weight, 0.0), 1.0))
    expected = w * ref_mean + (1.0 - w) * team_total
    return float(min(max(expected, 0.5), 12.0))


def cards_markets(expected_total: float, lines: tuple[float, ...] = (3.5, 4.5, 5.5)) -> dict:
    """
    Over/Under probabilities for total-cards lines from a Poisson(expected).

    Returns {"expected_total": x, "lines": {"3.5": {"over":..,"under":..}, ...}}.
    """
    out: dict = {"expected_total": round(expected_total, 3), "lines": {}}
    max_cards = 20
    g = np.arange(max_cards + 1)
    pmf = poisson.pmf(g, expected_total)
    for line in lines:
        floor = int(np.ceil(line))  # e.g. 3.5 -> over means >= 4
        over = float(pmf[g >= floor].sum())
        out["lines"][str(line)] = {"over": round(over, 4), "under": round(1.0 - over, 4)}
    return out


def cards_market_for_fixture(
    home_team: str,
    away_team: str,
    referee_name: str | None,
    source: RefereeSource,
    lines: tuple[float, ...] = (3.5, 4.5, 5.5),
    ref_weight: float = 0.65,
) -> tuple[dict, dict]:
    """
    Convenience: pull profiles from the source and return (markets, diagnostic).
    """
    ref = source.get_referee(referee_name) if referee_name else None
    home = source.get_team_cards(home_team)
    away = source.get_team_cards(away_team)
    base = source.baseline()

    expected = expected_cards(ref, home, away, base, ref_weight=ref_weight)
    markets = cards_markets(expected, lines=lines)

    diagnostic = {
        "referee": referee_name,
        "referee_known": ref is not None,
        "referee_yellows_per_game": round(ref.yellows_per_game, 2) if ref else None,
        "home_cards_per_game": round(home.cards_per_game, 2) if home else None,
        "away_cards_per_game": round(away.cards_per_game, 2) if away else None,
        "expected_total": markets["expected_total"],
    }
    return markets, diagnostic


__all__ = [
    "RefereeProfile",
    "TeamCardProfile",
    "CardsBaseline",
    "RefereeSource",
    "InMemoryRefereeSource",
    "SQLiteRefereeSource",
    "expected_cards",
    "cards_markets",
    "cards_market_for_fixture",
]
