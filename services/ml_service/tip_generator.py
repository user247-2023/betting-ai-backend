"""
tip_generator.py — Model-driven multi-market tip generation.

Volume problem: once tips are grounded in real data, the AI only emits a few
picks per slate, so the feed looks thin. Solution: for every fixture we DO have
real data on, price EVERY supported market with the Dixon-Coles model and surface
the ones that clear a probability bar. This adds many tips while keeping accuracy,
because each added tip is a real model probability — never an invented stat.

These model tips are merged with the AI's tips (AI keeps its richer reasoning;
the model fills the gaps and adds breadth). Everything downstream (odds, value,
filters) treats them identically.
"""
import os
import re
from typing import Dict, List

from services.data_service.form_provider import fixture_key
from services.ml_service.model_probs import (
    expected_goals, score_matrix,
    p_over, p_btts, p_home, p_away, p_draw,
    p_home_clean, p_away_clean, p_odd, p_home_atleast, p_away_atleast,
)

# Quality gate: only surface markets the model rates at/above this probability.
# Env-tunable: TIP_THRESHOLD (default 0.55 — was 0.62; the calibration tracker
# grades every band, so mid-probability tips are measured, not guessed).
THRESHOLD = float(os.getenv("TIP_THRESHOLD", "0.55"))
# Don't flood one match — keep its strongest N markets.
MAX_PER_FIXTURE = int(os.getenv("TIP_MAX_PER_FIXTURE", "6"))
# Hard ceiling on model tips added per slate.
MAX_TOTAL = int(os.getenv("TIP_MAX_TOTAL", "60"))
# Data-sufficiency guard: never tip a fixture we can't ground (junk protection).
MIN_FORM_MATCHES = 3
# Near-certainties (Over 0.5 at ~97%) carry odds near 1.02 — no betting value and
# they crowd out real picks. Anything above this ceiling is dropped.
MAX_PROB = float(os.getenv("TIP_MAX_PROB", "0.92"))


# One tip per market FAMILY per fixture. Without this the sweep emits several
# lines off the same market (Over 0.5 AND Under 4.5 AND Over 1.5...), which look
# contradictory to a user and can never sensibly share a slip.
def _family(settle: str, pick: str, home: str, away: str) -> str:
    if settle in ("team_goals", "clean_sheet", "win_to_nil"):
        who = "home" if home.lower() in (pick or "").lower() else "away"
        return f"{settle}:{who}"
    if settle in ("result", "double_chance", "dnb"):
        return "result"            # 1X2 / DC / DNB all price the same question
    return settle or "other"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _dedupe_key(match: str, settle: str, side, line) -> str:
    return f"{_norm(match)}|{settle}|{side}|{line}"


def _candidates(m, home: str, away: str) -> List[tuple]:
    """(prob, market, pick, settle_type, line, side) for every supported market."""
    h, d, a = p_home(m), p_draw(m), p_away(m)
    o15, o25, o35 = p_over(m, 1.5), p_over(m, 2.5), p_over(m, 3.5)
    btts = p_btts(m)
    odd = p_odd(m)
    dnb_den = h + a
    return [
        (o15, "Over/Under 1.5 Goals", "Over 1.5 Goals", "goals_ou", 1.5, "over"),
        (1 - o35, "Over/Under 3.5 Goals", "Under 3.5 Goals", "goals_ou", 3.5, "under"),
        (o25, "Over/Under 2.5 Goals", "Over 2.5 Goals", "goals_ou", 2.5, "over"),
        (1 - o25, "Over/Under 2.5 Goals", "Under 2.5 Goals", "goals_ou", 2.5, "under"),
        (btts, "Both Teams To Score", "BTTS Yes", "btts", None, "yes"),
        (1 - btts, "Both Teams To Score", "BTTS No", "btts", None, "no"),
        (h + d, "Double Chance", f"{home} or Draw", "double_chance", None, "1x"),
        (d + a, "Double Chance", f"Draw or {away}", "double_chance", None, "x2"),
        (h + a, "Double Chance", f"{home} or {away}", "double_chance", None, "12"),
        (h / dnb_den if dnb_den > 0 else 0, "Draw No Bet", f"{home} (DNB)", "dnb", None, "home"),
        (a / dnb_den if dnb_den > 0 else 0, "Draw No Bet", f"{away} (DNB)", "dnb", None, "away"),
        (h, "Match Result", f"{home} Win", "result", None, "home"),
        (a, "Match Result", f"{away} Win", "result", None, "away"),
        (p_home_clean(m), "Clean Sheet", f"{home} Clean Sheet", "clean_sheet", None, "home"),
        (p_away_clean(m), "Clean Sheet", f"{away} Clean Sheet", "clean_sheet", None, "away"),
        (p_home_atleast(m, 2), "Team Goals", f"{home} Over 1.5 Goals", "team_goals", 1.5, "over"),
        (p_away_atleast(m, 2), "Team Goals", f"{away} Over 1.5 Goals", "team_goals", 1.5, "over"),
        (p_home_atleast(m, 1), "Team To Score", f"{home} To Score", "team_goals", 0.5, "over"),
        (p_away_atleast(m, 1), "Team To Score", f"{away} To Score", "team_goals", 0.5, "over"),
        (o35, "Over/Under 3.5 Goals", "Over 3.5 Goals", "goals_ou", 3.5, "over"),
        (1 - odd, "Odd/Even Goals", "Even Total Goals", "odd_even", None, "even"),
    ]


def _risk(prob: float) -> str:
    if prob >= 0.72:
        return "LOW"
    if prob >= 0.62:
        return "MEDIUM"
    return "HIGH"


def _reason(home, away, lam_h, lam_a, rates, pick, prob) -> tuple:
    tot = lam_h + lam_a
    reasoning = (
        f"Model projects {home} {lam_h:.1f} - {lam_a:.1f} {away} "
        f"({tot:.1f} expected goals). {home} averaged {rates['home_gf']}/game scored, "
        f"{rates['home_ga']}/game conceded over their last {rates['home_n']}; "
        f"{away} {rates['away_gf']}/game scored, {rates['away_ga']}/game conceded "
        f"over their last {rates['away_n']}. '{pick}' lands in {prob*100:.0f}% of model outcomes."
    )
    key_stats = [
        f"Model expected goals: {tot:.1f}",
        f"{home}: {rates['home_gf']} scored / {rates['home_ga']} conceded per game (last {rates['home_n']})",
        f"{away}: {rates['away_gf']} scored / {rates['away_ga']} conceded per game (last {rates['away_n']})",
        f"Model probability: {prob*100:.0f}%",
    ]
    return reasoning, key_stats


def generate_model_tips(existing_tips: List[Dict], rates_map: Dict[str, Dict],
                        fixtures: List[Dict], date: str,
                        threshold: float = THRESHOLD,
                        max_per_fixture: int = MAX_PER_FIXTURE) -> List[Dict]:
    """Return existing tips + model-generated tips for covered fixtures."""
    if not rates_map:
        return existing_tips

    # fixture metadata (league/time/display names) keyed like rates_map
    meta = {
        fixture_key(fx.get("home", ""), fx.get("away", "")): fx
        for fx in fixtures
    }

    # don't duplicate what the AI already tipped
    seen = set()
    for t in existing_tips:
        seen.add(_dedupe_key(t.get("match", ""), t.get("settle_type", ""),
                             t.get("side"), t.get("line")))

    added: List[Dict] = []
    for key, rates in rates_map.items():
        if len(added) >= MAX_TOTAL:
            break
        fx = meta.get(key)
        if not fx:
            continue
        home, away = fx.get("home", ""), fx.get("away", "")
        match = f"{home} vs {away}"

        # Junk protection: tip only if the fitted model knows BOTH teams, or we
        # have at least MIN_FORM_MATCHES of real recent form for both. Fixtures
        # with no grounding produce prior-only probabilities — skip them.
        try:
            from services.ml_service import dixon_coles_fit as _dc
            fitted_ok = _dc.lambdas(home, away, bool(rates.get("neutral"))) is not None
        except Exception:
            fitted_ok = False
        form_ok = (rates.get("home_n", 0) >= MIN_FORM_MATCHES
                   and rates.get("away_n", 0) >= MIN_FORM_MATCHES)
        if not (fitted_ok or form_ok):
            continue

        try:
            lam_h, lam_a = expected_goals(rates)
            m = score_matrix(lam_h, lam_a)
            cands = _candidates(m, home, away)
        except Exception:
            continue

        cands.sort(key=lambda c: c[0], reverse=True)
        picked = 0
        fam_used = set()
        for prob, market, pick, settle, line, side in cands:
            if picked >= max_per_fixture or len(added) >= MAX_TOTAL:
                break
            if prob < threshold or prob > MAX_PROB:
                continue
            fam = _family(settle, pick, home, away)
            if fam in fam_used:
                continue          # never two picks off the same market family
            dk = _dedupe_key(match, settle, side, line)
            if dk in seen:
                continue
            seen.add(dk)
            fam_used.add(fam)
            prob = min(prob, 0.95)
            reasoning, key_stats = _reason(home, away, lam_h, lam_a, rates, pick, prob)
            added.append({
                "id": os.urandom(4).hex(),
                "match": match, "home_team": home, "away_team": away,
                "league": f"{fx.get('league','')} ({fx.get('country','')})".strip(),
                "time": fx.get("time", "TBD"),
                "market": market, "pick": pick,
                "settle_type": settle, "line": line, "side": side,
                "odds_range": "",
                "confidence": int(round(prob * 100)),
                "risk": _risk(prob),
                "reasoning": reasoning, "key_stats": key_stats,
                "ai_source": "Model", "ais": ["Model"], "votes": 1,
                "aiCount": 1, "multiAI": False, "confirmed": False,
                "predicted_probability": round(prob, 4),
                "model_probability": round(prob, 4),
                "prob_source": "model",
                "lambda_home": round(lam_h, 2), "lambda_away": round(lam_a, 2),
                "expected_goals": round(lam_h + lam_a, 2),
            })
            picked += 1

    return existing_tips + added
