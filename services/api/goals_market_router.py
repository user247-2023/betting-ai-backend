"""
goals_market_router.py — FastAPI router exposing the Dixon-Coles goals model.

Mount in main.py:
    from services.api.goals_market_router import router as goals_router
    app.include_router(goals_router)

MULTI-COMPETITION
-----------------
Ratings are league-relative (a +0.5 attack in the Premier League is not the
same as +0.5 in Serie B, and national teams live in their own world), so we
fit ONE model per competition and keep them side by side in a cache keyed by
league_id. Each competition is fitted automatically the first time it's asked
for, then reused until the next restart — no manual step.

Endpoints
---------
  GET  /api/goals/leagues                competitions available (id, name, matches)
  GET  /api/goals/status                 overview, or one league with ?league=
  POST /api/goals/refresh                force a refit of one league
  GET  /api/goals/ratings?league=39      attack/defence ratings, strongest first
  GET  /api/goals/predict?league=39      full goals-market view for one fixture
  POST /api/goals/value                  value bets for a fixture given live odds
  GET  /api/goals/clv/summary            Closing Line Value KPIs

`league` defaults to 39 (Premier League) so older callers keep working.

Author: Rollover Betting AI · API layer
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.ml_service.dixon_coles import DixonColesModel, load_matches_from_sqlite
from services.decision_engine.goals_value import find_goals_value
from services.decision_engine.clv_tracker import CLVTracker

router = APIRouter(prefix="/api/goals", tags=["goals-model"])

# --------------------------------------------------------------------------- #
#  Module-level model cache — one fitted model per competition                #
# --------------------------------------------------------------------------- #
_MODELS: dict[int, DixonColesModel] = {}
_DB_PATH = os.getenv("FOOTBALL_DB", "database/football.db")
_CLV = CLVTracker(db_path=os.getenv("CLV_DB", "database/clv.db"))
_DEFAULT_LEAGUE = int(os.getenv("GOALS_DEFAULT_LEAGUE", "39"))  # 39 = Premier League
_INTERNATIONAL_LEAGUE = 1  # national teams / World Cup

_COLS = {
    "home_team": "home_team",
    "away_team": "away_team",
    "home_goals": "home_goals",
    "away_goals": "away_goals",
    "match_date": "date",
    "league_id": "league_id",
}


def _half_life_for(league_id: int) -> float:
    """National teams play seldom and spread over years — keep more history."""
    return 540.0 if league_id == _INTERNATIONAL_LEAGUE else 180.0


def refresh_model(
    league_id: int = _DEFAULT_LEAGUE,
    half_life_days: float | None = None,
    ridge: float = 0.02,
) -> dict:
    """Fit (or refit) one competition from football.db and cache it."""
    hl = half_life_days if half_life_days is not None else _half_life_for(league_id)
    matches = load_matches_from_sqlite(
        _DB_PATH, table="fixtures", league_id=league_id, columns=_COLS
    )
    if len(matches) < 5:
        raise HTTPException(
            status_code=422,
            detail=f"League {league_id}: only {len(matches)} settled matches "
                   "available; need at least 5.",
        )
    model = DixonColesModel(half_life_days=hl, ridge=ridge)
    model.fit(matches)
    _MODELS[league_id] = model
    return {"league": league_id, **model.summary()}


def get_model(league_id: int = _DEFAULT_LEAGUE) -> DixonColesModel:
    """
    Return the fitted model for a competition, fitting it on first use.

    Models live in memory, so a restart (deploy, or the host sleeping/waking)
    clears them. Instead of a manual refresh, each competition fits itself from
    football.db the first time it's needed, then stays cached. No console step.
    """
    m = _MODELS.get(league_id)
    if m is not None and m.fitted:
        return m
    try:
        refresh_model(league_id=league_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not auto-fit league {league_id} from football.db: {e}",
        )
    return _MODELS[league_id]


def _available_leagues() -> list[dict]:
    """Competitions that actually have settled matches in football.db."""
    if not os.path.exists(_DB_PATH):
        return []
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT league_id,
                   MAX(league_name) AS league_name,
                   COUNT(*)         AS matches
            FROM fixtures
            WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
            GROUP BY league_id
            ORDER BY CASE league_id
                       WHEN 39 THEN 1 WHEN 140 THEN 2 WHEN 135 THEN 3
                       WHEN 78 THEN 4 WHEN 61 THEN 5 WHEN 1 THEN 99
                       ELSE 50 END,
                     league_name
            """
        ).fetchall()
    finally:
        con.close()
    return [
        {"league_id": r["league_id"], "name": r["league_name"], "matches": r["matches"]}
        for r in rows
    ]


# --------------------------------------------------------------------------- #
#  Request bodies                                                             #
# --------------------------------------------------------------------------- #
class RefreshRequest(BaseModel):
    league_id: Optional[int] = Field(None, description="Competition to refit (defaults to Premier League)")
    half_life_days: Optional[float] = Field(None, gt=0)
    ridge: float = Field(0.02, ge=0)


class ValueRequest(BaseModel):
    home_team: str
    away_team: str
    league: int = Field(_DEFAULT_LEAGUE, description="Competition the teams play in")
    # market_key -> {book: [odds...]}  OR  market_key -> [odds...]
    # keys: ou_1_5, ou_2_5, ou_3_5, btts, 1x2 (lists are [over,under] / [yes,no] / [home,draw,away])
    odds_by_market: dict
    devig_method: str = Field("power", pattern="^(multiplicative|additive|power|shin)$")
    consensus_aggregate: str = Field("median", pattern="^(median|mean|weighted_mean)$")
    min_edge: float = Field(0.05, ge=0)
    min_prob_gap: float = Field(0.02, ge=0)
    bankroll: Optional[float] = Field(None, gt=0)


# --------------------------------------------------------------------------- #
#  Endpoints                                                                  #
# --------------------------------------------------------------------------- #
@router.get("/leagues")
def leagues() -> dict:
    """List the competitions available to pick from (drives the app dropdown)."""
    return {"leagues": _available_leagues()}


@router.get("/status")
def status(league: Optional[int] = Query(None)) -> dict:
    if league is not None:
        m = _MODELS.get(league)
        if m is not None and m.fitted:
            return {"league": league, "fitted": True, **m.summary()}
        return {"league": league, "fitted": False,
                "note": "Fits automatically on the first prediction."}
    return {
        "db_path": _DB_PATH,
        "loaded_leagues": sorted(lid for lid, m in _MODELS.items() if m.fitted),
        "available": _available_leagues(),
    }


@router.post("/refresh")
def refresh(body: RefreshRequest) -> dict:
    league_id = body.league_id if body.league_id is not None else _DEFAULT_LEAGUE
    summary = refresh_model(league_id, body.half_life_days, body.ridge)
    return {"refreshed": True, "summary": summary}


@router.get("/ratings")
def ratings(league: int = Query(_DEFAULT_LEAGUE)) -> dict:
    model = get_model(league)
    return {"league": league, "n_teams": len(model.teams), "ratings": model.team_ratings()}


@router.get("/predict")
def predict(
    home_team: str = Query(..., description="Exact team name as stored during fit"),
    away_team: str = Query(...),
    league: int = Query(_DEFAULT_LEAGUE, description="Competition the teams play in"),
    top_n_scores: int = Query(5, ge=1, le=20),
) -> dict:
    model = get_model(league)
    try:
        out = model.predict_markets(home_team, away_team, top_n_scores).as_dict()
        out["league"] = league
        return out
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/value")
def value(body: ValueRequest) -> dict:
    model = get_model(body.league)
    try:
        return find_goals_value(
            model,
            body.home_team,
            body.away_team,
            body.odds_by_market,
            devig_method=body.devig_method,
            consensus_aggregate=body.consensus_aggregate,
            min_edge=body.min_edge,
            min_prob_gap=body.min_prob_gap,
            bankroll=body.bankroll,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/clv/summary")
def clv_summary(market: Optional[str] = Query(None)) -> dict:
    return _CLV.summary(market=market)
