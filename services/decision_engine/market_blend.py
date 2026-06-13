"""
services/decision_engine/market_blend.py

Two accuracy upgrades and two new tip markets, in one place.

1. MARKET BLEND  (accuracy)
   The de-vigged bookmaker consensus is the single strongest public predictor
   of a result — it silently aggregates injuries, team news, lineups and sharp
   money the historical goals model never sees. Blending the model with that
   consensus (a convex combination) is the most reliable way to cut prediction
   error, and it matters most for internationals, where the model is weakest.
   The PURE model probabilities are kept alongside: the gap between pure-model
   and market IS the value signal, so accuracy and value-finding stay separable.

2. NEUTRAL-VENUE EXPECTED GOALS  (accuracy, World Cup)
   Dixon-Coles credits the nominal "home" team with a home-advantage term. At a
   World Cup almost every match is at a neutral venue, so handing "Qatar" home
   advantage against Switzerland is simply wrong. neutral_lambdas() rebuilds the
   expected goals with the home-advantage term removed, then feeds them through
   the model's own score-matrix machinery — no refit, no DB change required.

3. DOUBLE CHANCE + 1X2 TIPS  (new markets)
   Double chance is an exact function of the 1X2 distribution
   (1X = home+draw, 12 = home+away, X2 = draw+away). It is exposed with fair
   odds, plus a plain-language tip: the pick and a confidence band.
"""

from __future__ import annotations

import math
import os

from services.decision_engine.devig import market_consensus


# --------------------------------------------------------------------------- #
#  Blend weight (env-tunable per competition family)                          #
# --------------------------------------------------------------------------- #

def model_weight(league_id: int | None) -> float:
    """How much to trust the model vs the market consensus (0..1, model share)."""
    if league_id == 1:        # internationals / World Cup — model is weakest here
        return _clamp(float(os.getenv("BLEND_W_MODEL_INTL", "0.40")))
    return _clamp(float(os.getenv("BLEND_W_MODEL_CLUB", "0.55")))


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


# --------------------------------------------------------------------------- #
#  Neutral-venue expected goals                                               #
# --------------------------------------------------------------------------- #

def neutral_lambdas(model, home: str, away: str) -> tuple[float, float]:
    """
    Expected goals with the home-advantage term removed (neutral venue).
    Uses the fitted model's parameters directly; raises KeyError if a team
    isn't in the model (caller maps that to a 404 / 'unmatched').
    """
    h = model._idx[home]
    a = model._idx[away]
    lam_h = math.exp(model.intercept + float(model.attack[h]) - float(model.defence[a]))
    lam_a = math.exp(model.intercept + float(model.attack[a]) - float(model.defence[h]))
    return lam_h, lam_a


def model_markets(model, home: str, away: str, neutral: bool = False):
    """Return (probs_1x2, full_as_dict). Neutral drops home advantage."""
    if neutral:
        lh, la = neutral_lambdas(model, home, away)
        mk = model.predict_markets_from_lambdas(home, away, lh, la)
    else:
        mk = model.predict_markets(home, away)
    d = mk.as_dict()
    p = [d["1x2"]["home"], d["1x2"]["draw"], d["1x2"]["away"]]
    return p, d


# --------------------------------------------------------------------------- #
#  Market consensus + blend                                                   #
# --------------------------------------------------------------------------- #

def consensus_1x2(books: dict | None):
    """
    books: {book: {"1x2":[home,draw,away], ...}}
    -> (consensus_probs[home,draw,away], consensus_info) or None if no 1x2 odds.
    """
    odds = {
        b: m["1x2"]
        for b, m in (books or {}).items()
        if isinstance(m, dict) and len(m.get("1x2", [])) == 3
    }
    if not odds:
        return None
    info = market_consensus(odds, method="power", aggregate="median")
    return info["consensus_probabilities"], info


def blend_1x2(model_probs, market_probs, w_model: float):
    blended = [w_model * m + (1.0 - w_model) * k for m, k in zip(model_probs, market_probs)]
    s = sum(blended) or 1.0
    return [x / s for x in blended]


# --------------------------------------------------------------------------- #
#  Double chance + tips                                                       #
# --------------------------------------------------------------------------- #

def double_chance(p1x2):
    h, d, a = p1x2
    pairs = {"1X": h + d, "12": h + a, "X2": d + a}
    return {
        k: {"prob": round(v, 4), "fair_odds": round(1.0 / v, 3) if v > 0 else None}
        for k, v in pairs.items()
    }


def _confidence(p: float) -> str:
    if p >= 0.60:
        return "High"
    if p >= 0.45:
        return "Medium"
    return "Lean"


def make_tips(p1x2, home: str, away: str) -> dict:
    results = [("1", f"{home} win", p1x2[0]), ("X", "Draw", p1x2[1]), ("2", f"{away} win", p1x2[2])]
    best = max(results, key=lambda x: x[2])
    dc = double_chance(p1x2)
    dc_names = {"1X": f"{home} or Draw", "12": f"{home} or {away}", "X2": f"Draw or {away}"}
    dc_best = max(dc, key=lambda k: dc[k]["prob"])
    return {
        "result_pick": {
            "selection": best[0],
            "label": best[1],
            "prob": round(best[2], 4),
            "confidence": _confidence(best[2]),
        },
        "double_chance_pick": {
            "selection": dc_best,
            "label": dc_names[dc_best],
            "prob": dc[dc_best]["prob"],
            "fair_odds": dc[dc_best]["fair_odds"],
        },
    }


# --------------------------------------------------------------------------- #
#  One readout to feed the UI                                                 #
# --------------------------------------------------------------------------- #

def full_readout(model, home: str, away: str, books: dict | None = None,
                 neutral: bool = False, league_id: int | None = None) -> dict:
    """
    Everything the live cards need:
      model_1x2     — pure model
      market_1x2    — de-vigged bookmaker consensus (if odds present)
      prediction_1x2— blended = the most accurate single forecast
      double_chance — derived from the blended (or pure) distribution
      tips          — plain-language result pick + safest double-chance pick
      model_vs_market_edge — where the model disagrees (the value signal)
    """
    m_probs, full = model_markets(model, home, away, neutral=neutral)
    out = {
        "neutral_venue": neutral,
        "model_1x2": _xyz(m_probs),
        "expected_goals": full["expected_goals"],
        "over_under_2_5": full["over_under"]["2.5"],
        "btts": full["btts"],
        "top_score": (full.get("correct_score") or [None])[0],
    }

    market = consensus_1x2(books)
    if market:
        k_probs, cinfo = market
        w = model_weight(league_id)
        blended = blend_1x2(m_probs, k_probs, w)
        out["market_1x2"] = _xyz(k_probs)
        out["prediction_1x2"] = _xyz(blended)
        out["blend_weight_model"] = w
        out["n_books"] = cinfo["n_books"]
        out["model_vs_market_edge"] = {
            key: round(m - k, 4)
            for key, m, k in zip(("home", "draw", "away"), m_probs, k_probs)
        }
        basis = blended
    else:
        out["prediction_1x2"] = out["model_1x2"]
        basis = m_probs

    out["double_chance"] = double_chance(basis)
    out["tips"] = make_tips(basis, home, away)
    return out


def _xyz(p):
    return {"home": round(p[0], 4), "draw": round(p[1], 4), "away": round(p[2], 4)}
