"""
goals_market_router.py — FastAPI router exposing the Dixon-Coles goals model.

Mount in main.py:
    from services.api.goals_market_router import router as goals_router
    app.include_router(goals_router)

The fitted model is cached at module level (fitting scans the whole match
history, so we do it once and reuse). Refit on a schedule (the project already
runs APScheduler — call refresh_model() Sunday alongside ML training) or via
POST /api/goals/refresh.

Endpoints
---------
  GET  /api/goals/status                 model fit status + params + ratings count
  POST /api/goals/refresh                refit from football.db (optionally a league)
  GET  /api/goals/ratings                attack/defence ratings, strongest first
  GET  /api/goals/predict                full goals-market view for one fixture
  POST /api/goals/value                  value bets for a fixture given live odds
  GET  /api/goals/clv/summary            Closing Line Value KPIs

Author: Rollover Betting AI · API layer
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.ml_service.dixon_coles import DixonColesModel, load_matches_from_sqlite
from services.decision_engine.goals_value import find_goals_value
from services.decision_engine.clv_tracker import CLVTracker

router = APIRouter(prefix="/api/goals", tags=["goals-model"])

# --------------------------------------------------------------------------- #
#  Module-level model cache                                                   #
# --------------------------------------------------------------------------- #
_MODEL: Optional[DixonColesModel] = None
_DB_PATH = os.getenv("FOOTBALL_DB", "database/football.db")
_CLV = CLVTracker(db_path=os.getenv("CLV_DB", "database/clv.db"))
_DEFAULT_LEAGUE = int(os.getenv("GOALS_DEFAULT_LEAGUE", "39"))  # 39 = Premier League


def get_model() -> DixonColesModel:
    """
    Return the fitted model, fitting it automatically on first use.

    The model lives in memory, so any restart (deploy, or the host sleeping and
    waking) leaves it empty. Rather than require a manual refresh each time, we
    fit it on demand from football.db for the default league the first time it's
    needed, then keep it cached until the next restart. No console step needed.
    """
    global _MODEL
    if _MODEL is not None and _MODEL.fitted:
        return _MODEL
    try:
        refresh_model(league_id=_DEFAULT_LEAGUE)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Goals model could not auto-fit from football.db: {e}",
        )
    return _MODEL


def refresh_model(
    league_id: int | None = None,
    half_life_days: float = 180.0,
    ridge: float = 0.02,
) -> dict:
    """Refit the Dixon-Coles model from football.db. Safe to call from a scheduler."""
    global _MODEL
    matches = load_matches_from_sqlite(
        _DB_PATH,
        table="fixtures",
        league_id=league_id,
        columns={
            "home_team": "home_team",
            "away_team": "away_team",
            "home_goals": "home_goals",
            "away_goals": "away_goals",
            "match_date": "date",
            "league_id": "league_id",
        },
    )
    if len(matches) < 5:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(matches)} settled matches available; need >=5. "
                   "Let the dataset builder accumulate more results.",
        )
    model = DixonColesModel(half_life_days=half_life_days, ridge=ridge)
    model.fit(matches)
    _MODEL = model
    return model.summary()


# --------------------------------------------------------------------------- #
#  Request bodies                                                             #
# --------------------------------------------------------------------------- #
class RefreshRequest(BaseModel):
    league_id: Optional[int] = Field(None, description="Restrict fit to one league (recommended — ratings are league-relative)")
    half_life_days: float = Field(180.0, gt=0)
    ridge: float = Field(0.02, ge=0)


class ValueRequest(BaseModel):
    home_team: str
    away_team: str
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
@router.get("/status")
def status() -> dict:
    if _MODEL is not None and _MODEL.fitted:
        return {"fitted": True, **_MODEL.summary(), "db_path": _DB_PATH}
    return {"fitted": False, "db_path": _DB_PATH,
            "note": "Model fits automatically on the first prediction; no action needed."}


@router.post("/refresh")
def refresh(body: RefreshRequest) -> dict:
    summary = refresh_model(body.league_id, body.half_life_days, body.ridge)
    return {"refreshed": True, "summary": summary}


@router.get("/ratings")
def ratings() -> dict:
    model = get_model()
    return {"n_teams": len(model.teams), "ratings": model.team_ratings()}


@router.get("/predict")
def predict(
    home_team: str = Query(..., description="Exact team name as stored during fit"),
    away_team: str = Query(...),
    top_n_scores: int = Query(5, ge=1, le=20),
) -> dict:
    model = get_model()
    try:
        return model.predict_markets(home_team, away_team, top_n_scores).as_dict()
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/value")
def value(body: ValueRequest) -> dict:
    model = get_model()
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
