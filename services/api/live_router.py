"""
services/api/live_router.py

The live layer's front door. Self-maintaining: the first request after a
quiet spell quietly tops up results (free) and odds snapshots (budgeted),
then serves from the local store. Nothing here can overspend — every paid
call goes through live_data's budget meter.

Endpoints:
  GET  /api/live/matches            upcoming fixtures + model's instant read
  GET  /api/live/value/{event_id}   full value verdict vs stored multi-book odds
  GET  /api/live/budget             the fuel gauge
  GET  /api/live/health             which keys are set (booleans only, never values)
  GET  /api/live/unmatched          feed names the translator couldn't place
  GET  /api/live/apifootball-test   honest probe of the API-Football free key
  POST /api/live/snapshot           force a snapshot now (still budget-capped)
  POST /api/live/results/refresh    force the free results refresh now
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from services.api.goals_market_router import get_model, _MODELS
from services.data_service import live_data, team_matcher
from services.decision_engine.goals_value import find_goals_value
from services.decision_engine import market_blend

router = APIRouter(prefix="/api/live", tags=["live"])

ACTIVE_SPORTS = [
    s.strip()
    for s in os.getenv("ACTIVE_SPORTS", "soccer_fifa_world_cup").split(",")
    if s.strip()
]
SNAPSHOT_TTL_MIN = int(os.getenv("ODDS_SNAPSHOT_TTL_MIN", "360"))


def _minutes_since(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return (
            datetime.now(timezone.utc) - datetime.fromisoformat(iso)
        ).total_seconds() / 60
    except Exception:
        return None


def _after_new_results() -> None:
    """New finished matches arrived -> let the intl model re-learn."""
    _MODELS.pop(1, None)
    team_matcher.clear_cache()


def _ensure_fresh(force: bool = False) -> list[str]:
    """Lazy upkeep. Failures never break the request — we serve stale + a note."""
    notes: list[str] = []

    # 1. Free results river
    try:
        r = live_data.refresh_results(force=force)
        if r.get("added"):
            _after_new_results()
            notes.append(f"results: +{r['added']} new finished matches")
    except Exception as e:
        notes.append(f"results refresh failed (serving existing data): {e}")

    # 2. Budgeted odds snapshots, one per active competition, only when stale
    for sport in ACTIVE_SPORTS:
        age = _minutes_since(live_data.meta_get(f"last_snapshot:{sport}"))
        if not force and age is not None and age < SNAPSHOT_TTL_MIN:
            continue
        try:
            s = live_data.snapshot(sport)
            if s.get("skipped"):
                notes.append(f"{sport}: snapshot skipped — {s.get('reason')}")
            else:
                notes.append(
                    f"{sport}: snapshot ok ({s.get('events_stored', 0)} events, "
                    f"{s.get('credits_remaining')} credits left)"
                )
        except RuntimeError as e:          # key not set
            notes.append(f"{sport}: {e}")
        except Exception as e:
            notes.append(f"{sport}: snapshot failed (serving stored odds): {e}")
    return notes


def _readout(model, home: str, away: str, books: dict, league_id: int) -> dict | None:
    """Model + market consensus + blended prediction + double chance + tips.

    World Cup matches (league 1) are treated as neutral-venue, so the model's
    home-advantage term is dropped — the single biggest accuracy fix for the
    tournament. The blended prediction (model + de-vigged market) is the most
    accurate single forecast; the pure model and the model-vs-market edge ride
    alongside it so value-finding stays intact.
    """
    try:
        return market_blend.full_readout(
            model, home, away,
            books=books,
            neutral=(league_id == 1),
            league_id=league_id,
        )
    except KeyError:
        return None


# --------------------------------------------------------------------------- #
#  Endpoints                                                                  #
# --------------------------------------------------------------------------- #

@router.get("/matches")
def live_matches(
    hours: int = Query(96, ge=1, le=336, description="Look-ahead window"),
    include_started: bool = Query(False),
) -> dict:
    notes = _ensure_fresh()
    events = live_data.latest_events(hours_ahead=hours, include_started=include_started)

    matches = []
    model_cache: dict[int, object] = {}
    for ev in events:
        pred = None
        if ev["home_model"] and ev["away_model"]:
            lid = ev["league_id"]
            try:
                if lid not in model_cache:
                    model_cache[lid] = get_model(lid)
                pred = _readout(
                    model_cache[lid], ev["home_model"], ev["away_model"],
                    ev["books"], lid,
                )
            except Exception as e:
                notes.append(f"model league {lid}: {e}")
        matches.append(
            {
                "event_id": ev["event_id"],
                "league_id": ev["league_id"],
                "kickoff": ev["commence"],
                "started": ev["started"],
                "home": ev["home_model"] or ev["home_src"],
                "away": ev["away_model"] or ev["away_src"],
                "home_feed_name": ev["home_src"],
                "away_feed_name": ev["away_src"],
                "matched": bool(ev["home_model"] and ev["away_model"]),
                "books_count": ev["books_count"],
                "snapshot_age_min": ev["snapshot_age_min"],
                "prediction": pred,
            }
        )

    # Track record: freeze pre-kickoff predictions + grade any that finished.
    # Both are cheap and local (no API); failures never break the listing.
    try:
        live_data.log_predictions(matches)
        graded = live_data.settle_predictions()
        if graded:
            notes.append(f"scorecard: graded {graded} finished prediction(s)")
    except Exception as e:
        notes.append(f"scorecard logging skipped: {e}")

    return {
        "count": len(matches),
        "matches": matches,
        "notes": notes,
        "budget": live_data.budget_status(),
    }


@router.get("/value/{event_id}")
def live_value(
    event_id: str,
    bankroll: float | None = Query(None, gt=0),
    min_edge: float = Query(0.05, ge=0.0, le=0.5),
) -> dict:
    ev = live_data.get_event(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="event not found in store")
    if not (ev["home_model"] and ev["away_model"]):
        raise HTTPException(
            status_code=422,
            detail=(
                "Team names could not be matched to the model: "
                f"home='{ev['home_src']}' -> {ev['home_model']}, "
                f"away='{ev['away_src']}' -> {ev['away_model']}"
            ),
        )

    odds_by_market: dict[str, dict[str, list[float]]] = {}
    for book, markets in (ev["books"] or {}).items():
        for mkey, prices in markets.items():
            odds_by_market.setdefault(mkey, {})[book] = prices
    if not odds_by_market:
        raise HTTPException(
            status_code=422, detail="no usable odds stored for this event yet"
        )

    model = get_model(ev["league_id"])
    try:
        result = find_goals_value(
            model,
            ev["home_model"],
            ev["away_model"],
            odds_by_market,
            min_edge=min_edge,
            bankroll=bankroll,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result["event"] = {
        "event_id": ev["event_id"],
        "league_id": ev["league_id"],
        "kickoff": ev["commence"],
        "books_count": ev["books_count"],
        "books_used": sorted((ev["books"] or {}).keys()),
        "snapshot_age_min": ev["snapshot_age_min"],
        "feed_names": {"home": ev["home_src"], "away": ev["away_src"]},
    }

    # Blended prediction + double chance + plain-language tips alongside value
    readout = _readout(model, ev["home_model"], ev["away_model"], ev["books"], ev["league_id"])
    if readout:
        result["prediction_1x2"] = readout["prediction_1x2"]
        result["model_1x2"] = readout["model_1x2"]
        result["market_1x2"] = readout.get("market_1x2")
        result["double_chance"] = readout["double_chance"]
        result["tips"] = readout["tips"]
        result["neutral_venue"] = readout["neutral_venue"]
    return result


@router.post("/snapshot")
def force_snapshot(sport: str = Query("soccer_fifa_world_cup")) -> dict:
    if sport not in live_data.SPORT_TO_LEAGUE:
        raise HTTPException(
            status_code=422,
            detail=f"unknown sport; known: {sorted(live_data.SPORT_TO_LEAGUE)}",
        )
    try:
        return live_data.snapshot(sport)   # budget still enforced — by design
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"snapshot failed: {e}")


@router.post("/results/refresh")
def force_results_refresh() -> dict:
    try:
        r = live_data.refresh_results(force=True)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"results refresh failed: {e}")
    if r.get("added"):
        _after_new_results()
    return r


@router.get("/budget")
def budget() -> dict:
    b = live_data.budget_status()
    b["last_snapshots"] = {
        s: live_data.meta_get(f"last_snapshot:{s}") for s in ACTIVE_SPORTS
    }
    b["active_sports"] = ACTIVE_SPORTS
    b["snapshot_ttl_min"] = SNAPSHOT_TTL_MIN
    return b


@router.get("/health")
def live_health() -> dict:
    return {
        "odds_api_key_set": bool(os.getenv("ODDS_API_KEY")),
        "api_football_key_set": bool(os.getenv("API_FOOTBALL_KEY")),
        "db_path": live_data.DB_PATH,
        "db_exists": os.path.exists(live_data.DB_PATH),
        "active_sports": ACTIVE_SPORTS,
        "snapshot_ttl_min": SNAPSHOT_TTL_MIN,
        "daily_snapshot_cap": live_data.DAILY_SNAPSHOT_CAP,
    }


@router.get("/unmatched")
def unmatched() -> dict:
    try:
        items = json.loads(live_data.meta_get("unmatched", "[]"))
    except Exception:
        items = []
    return {"count": len(items), "unmatched": items}


@router.get("/scorecard")
def scorecard() -> dict:
    """The track record — accuracy, sharpness, and whether the blend beats the market."""
    try:
        live_data.settle_predictions()
    except Exception:
        pass
    return live_data.scorecard()


@router.post("/scorecard/settle")
def scorecard_settle() -> dict:
    graded = live_data.settle_predictions()
    return {"graded": graded, **live_data.scorecard()}


@router.get("/apifootball-test")
def apifootball_test() -> dict:
    return live_data.apifootball_test()
