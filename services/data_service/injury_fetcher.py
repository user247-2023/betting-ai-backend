"""
injury_fetcher.py — Injury / availability adapter and its model hand-off.

WHAT IT DOES
------------
Missing players move lines. A side without its first-choice striker scores
less; without its first-choice centre-backs it concedes more. This adapter
turns an availability report into a *multiplicative adjustment* on the
Dixon-Coles expected goals, so the rest of the pipeline (markets, value,
staking) automatically reflects the team news.

    lam_h, lam_a = model.expected_goals(home, away)
    lam_h, lam_a = apply_injury_adjustment(lam_h, lam_a, home_report, away_report)
    markets = model.predict_markets_from_lambdas(home, away, lam_h, lam_a)

DATA SOURCE  (you wire it)
--------------------------
API-Football exposes injuries at GET /injuries (by fixture, or by team+season).
`APIFootballInjurySource` calls it if the `requests` package is installed and
API_FOOTBALL_KEY is set; otherwise use `InMemoryInjurySource` and populate it
from whatever feed you already pull fixtures from (you are presumably already
hitting API-Football for fixtures, so reuse that client/key).

HONEST CAVEATS
--------------
* The free injuries endpoint gives you a LIST of unavailable players — not how
  important each one is. A true adjustment needs per-player importance
  (minutes share, goal involvement, or a rating). Without that, every absence
  looks equal, which is wrong (your 4th-choice full-back ≠ your top scorer).
  So this module accepts an optional `importance` map (player -> 0..1 weight);
  supply it from your own data and the adjustment gets real. With no weights it
  falls back to a blunt per-absence penalty, which is a rough heuristic only.
* Lineups firm up ~1 hour before kickoff. Pull as late as you safely can.
* Suspensions are availability too — fold them into the same report.

Dependencies: standard library. `requests` only if you use the HTTP source.

Author: Rollover Betting AI · Tier-3 enrichment
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol


# --------------------------------------------------------------------------- #
#  Data model                                                                 #
# --------------------------------------------------------------------------- #
# Coarse position buckets — what matters for goals is attack vs defence.
ATTACK = "attack"      # forwards, attacking mids, key creators
DEFENCE = "defence"    # defenders, defensive mids, goalkeeper
NEUTRAL = "neutral"    # squad rotation / unclear


@dataclass
class PlayerInjury:
    player: str
    position_group: str = NEUTRAL   # ATTACK | DEFENCE | NEUTRAL
    status: str = "out"             # out | doubtful | suspended
    detail: str = ""                # e.g. "Hamstring", from the feed


@dataclass
class TeamInjuryReport:
    team: str
    injuries: list[PlayerInjury] = field(default_factory=list)

    def out_players(self) -> list[PlayerInjury]:
        """Players effectively unavailable (out or suspended; doubtful counts half)."""
        return [i for i in self.injuries if i.status in ("out", "suspended")]


# --------------------------------------------------------------------------- #
#  Source interface                                                           #
# --------------------------------------------------------------------------- #
class InjurySource(Protocol):
    def get_report(self, team: str) -> TeamInjuryReport | None: ...


class InMemoryInjurySource:
    """Populate from your own feed; hand to the adjustment functions."""

    def __init__(self, reports: dict[str, TeamInjuryReport] | None = None) -> None:
        self._reports = reports or {}

    def set(self, report: TeamInjuryReport) -> None:
        self._reports[report.team] = report

    def get_report(self, team: str) -> TeamInjuryReport | None:
        return self._reports.get(team)


class APIFootballInjurySource:
    """
    Live source backed by API-Football's /injuries endpoint.

    Requires `requests` and the API_FOOTBALL_KEY env var (same key you use for
    fixtures). Maps the feed's position field to ATTACK/DEFENCE/NEUTRAL with a
    simple rule you can refine. Network/parse failures degrade to None so the
    pipeline can fall back to "no adjustment" rather than crash.

    Note: the endpoint does not tell you player importance — see module docstring.
    """

    BASE = "https://v3.football.api-sports.io"

    # Map common API-Football position strings to our buckets.
    _POS = {
        "Attacker": ATTACK,
        "Midfielder": NEUTRAL,   # mids split both ways; treat as neutral unless you refine
        "Defender": DEFENCE,
        "Goalkeeper": DEFENCE,
    }

    def __init__(self, api_key: str | None = None, season: int | None = None) -> None:
        self.api_key = api_key or os.environ.get("API_FOOTBALL_KEY")
        self.season = season

    def get_report(self, team_id_or_name: str, team_id: int | None = None) -> TeamInjuryReport | None:
        if not self.api_key:
            return None
        try:
            import requests  # local import: only needed for the live source
        except ImportError:
            return None

        params: dict = {}
        if team_id is not None:
            params["team"] = team_id
        if self.season is not None:
            params["season"] = self.season
        if not params:
            # Without a team id or fixture id the endpoint can't be queried usefully.
            return None

        try:
            r = requests.get(
                f"{self.BASE}/injuries",
                headers={"x-apisports-key": self.api_key},
                params=params,
                timeout=10,
            )
            r.raise_for_status()
            payload = r.json()
        except Exception:
            return None

        injuries: list[PlayerInjury] = []
        for item in payload.get("response", []):
            player = item.get("player", {}) or {}
            name = player.get("name") or "Unknown"
            pos = player.get("position") or ""
            reason = player.get("reason") or ""
            group = self._POS.get(pos, NEUTRAL)
            status = "out"
            if "suspend" in reason.lower():
                status = "suspended"
            elif "doubt" in reason.lower():
                status = "doubtful"
            injuries.append(
                PlayerInjury(player=name, position_group=group, status=status, detail=reason)
            )

        return TeamInjuryReport(team=str(team_id_or_name), injuries=injuries)


# --------------------------------------------------------------------------- #
#  Availability -> expected-goals adjustment                                  #
# --------------------------------------------------------------------------- #
# Blunt fallback penalties used ONLY when no importance weights are supplied.
# Each *attacking* absence trims the team's own attack; each *defensive*
# absence lifts the opponent's attack. Capped so a long injury list can't
# nuke a lambda to zero.
_FALLBACK_ATTACK_HIT = 0.06      # per key attacker missing
_FALLBACK_DEFENCE_HIT = 0.05     # per key defender missing
_DOUBTFUL_FACTOR = 0.5           # doubtful players count half
_MAX_TOTAL_HIT = 0.30            # never remove more than 30% either way


def _absence_units(report: TeamInjuryReport, group: str, importance: dict[str, float] | None) -> float:
    """
    Sum of 'importance units' missing for a position group.

    With an importance map: sum the weights of absent players in that group
    (doubtful at half weight). Without one: count bodies (doubtful at half).
    """
    units = 0.0
    for inj in report.injuries:
        if inj.position_group != group:
            continue
        if inj.status not in ("out", "suspended", "doubtful"):
            continue
        weight = 1.0
        if importance is not None:
            weight = float(importance.get(inj.player, 0.0))
        if inj.status == "doubtful":
            weight *= _DOUBTFUL_FACTOR
        units += weight
    return units


def availability_factors(
    report: TeamInjuryReport | None,
    importance: dict[str, float] | None = None,
) -> tuple[float, float]:
    """
    Return (own_attack_factor, opponent_attack_factor) for one team's report.

    * own_attack_factor (<=1) multiplies THIS team's expected goals down for
      missing attackers.
    * opponent_attack_factor (>=1) multiplies the OPPONENT's expected goals up
      for this team's missing defenders.

    With importance weights the magnitude scales with how good the missing
    players are; without them it falls back to a flat per-absence heuristic.
    """
    if report is None:
        return 1.0, 1.0

    if importance is not None:
        # Importance units already encode 'how much attack/defence we lost'.
        att_units = _absence_units(report, ATTACK, importance)
        def_units = _absence_units(report, DEFENCE, importance)
        # A full unit (e.g. losing your single most important attacker, weight 1.0)
        # trims ~20% of attack; symmetric on defence -> opponent attack.
        own_attack = max(1.0 - 0.20 * att_units, 1.0 - _MAX_TOTAL_HIT)
        opp_attack = min(1.0 + 0.18 * def_units, 1.0 + _MAX_TOTAL_HIT)
        return own_attack, opp_attack

    # Blunt fallback: count bodies.
    att_units = _absence_units(report, ATTACK, None)
    def_units = _absence_units(report, DEFENCE, None)
    own_attack = max(1.0 - _FALLBACK_ATTACK_HIT * att_units, 1.0 - _MAX_TOTAL_HIT)
    opp_attack = min(1.0 + _FALLBACK_DEFENCE_HIT * def_units, 1.0 + _MAX_TOTAL_HIT)
    return own_attack, opp_attack


def apply_injury_adjustment(
    lam_h: float,
    lam_a: float,
    home_report: TeamInjuryReport | None,
    away_report: TeamInjuryReport | None,
    home_importance: dict[str, float] | None = None,
    away_importance: dict[str, float] | None = None,
) -> tuple[float, float, dict]:
    """
    Adjust (lam_h, lam_a) for both teams' availability. Returns the new lambdas
    plus a diagnostic dict for the 'why this tip' view.

    Logic:
      home attackers out  -> home lambda down
      home defenders out  -> away lambda up
      away attackers out  -> away lambda down
      away defenders out  -> home lambda up
    """
    h_own_att, h_opp_att = availability_factors(home_report, home_importance)
    a_own_att, a_opp_att = availability_factors(away_report, away_importance)

    new_h = lam_h * h_own_att * a_opp_att
    new_a = lam_a * a_own_att * h_opp_att

    diagnostic = {
        "injury_applied": bool(
            (home_report and home_report.injuries) or (away_report and away_report.injuries)
        ),
        "lambda_before": {"home": round(lam_h, 3), "away": round(lam_a, 3)},
        "lambda_after": {"home": round(new_h, 3), "away": round(new_a, 3)},
        "home_factors": {"own_attack": round(h_own_att, 3), "lifts_away": round(h_opp_att, 3)},
        "away_factors": {"own_attack": round(a_own_att, 3), "lifts_home": round(a_opp_att, 3)},
        "used_importance_weights": bool(home_importance or away_importance),
    }
    return float(new_h), float(new_a), diagnostic


__all__ = [
    "ATTACK",
    "DEFENCE",
    "NEUTRAL",
    "PlayerInjury",
    "TeamInjuryReport",
    "InjurySource",
    "InMemoryInjurySource",
    "APIFootballInjurySource",
    "availability_factors",
    "apply_injury_adjustment",
]
