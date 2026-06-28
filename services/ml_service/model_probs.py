"""
model_probs.py — Dixon-Coles market probabilities from REAL scoring rates.

This replaces the AI's self-reported confidence as the source of each tip's
probability. For every goals/result market the AI can produce, it computes a
proper probability from a bivariate-Poisson (Dixon-Coles) model built on the
two teams' actual recent scoring/conceding rates (supplied by form_provider).

Pure Python, no external dependencies — safe to import anywhere.

Pipeline effect:
    predicted_probability  <- model probability (for supported markets)
    edge = predicted_probability * odds - 1   (now a real estimate, not AI vibes)
"""
import math
from typing import Dict, List, Optional, Tuple

from services.data_service.form_provider import fixture_key
from services.ml_service import dixon_coles_fit as _dc

# ── Model parameters ─────────────────────────────────────────────────────────
PRIOR = 1.35        # league-average goals per team (shrinkage target)
SHRINK_K = 5.0      # shrinkage strength: weight = n / (n + K)
HOME_ADV = 1.10     # home attack multiplier
AWAY_PEN = 0.92     # away attack multiplier
RHO = -0.10         # Dixon-Coles low-score dependence
MAX_GOALS = 10
CLIP = (0.20, 4.50)
PROB_CLAMP = (0.02, 0.98)


def _shrink(avg: float, n: int) -> float:
    w = n / (n + SHRINK_K)
    return w * avg + (1.0 - w) * PRIOR


def expected_goals(r: Dict) -> Tuple[float, float]:
    """Expected goals for the fixture.
    Prefers the FITTED Dixon-Coles strength model (real attack/defense ratings);
    falls back to the simple last-6 shrinkage estimate if the model isn't fitted
    yet or a team is unknown."""
    fitted = None
    try:
        fitted = _dc.lambdas(r.get("home_team", ""), r.get("away_team", ""),
                             bool(r.get("neutral")))
    except Exception:
        fitted = None
    if fitted:
        return fitted

    h_gf = _shrink(r["home_gf"], r["home_n"])
    h_ga = _shrink(r["home_ga"], r["home_n"])
    a_gf = _shrink(r["away_gf"], r["away_n"])
    a_ga = _shrink(r["away_ga"], r["away_n"])
    lam_h = (h_gf + a_ga) / 2.0
    lam_a = (a_gf + h_ga) / 2.0
    if not r.get("neutral"):
        lam_h *= HOME_ADV
        lam_a *= AWAY_PEN
    lam_h = min(max(lam_h, CLIP[0]), CLIP[1])
    lam_a = min(max(lam_a, CLIP[0]), CLIP[1])
    return lam_h, lam_a


def _pois(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def score_matrix(lam_h: float, lam_a: float, rho: float = None, maxg: int = MAX_GOALS):
    if rho is None:
        try:
            rho = _dc.rho()
        except Exception:
            rho = RHO
    """Joint P(home=i, away=j) with Dixon-Coles low-score correction, renormalised."""
    ph = [_pois(i, lam_h) for i in range(maxg + 1)]
    pa = [_pois(j, lam_a) for j in range(maxg + 1)]
    m = [[ph[i] * pa[j] for j in range(maxg + 1)] for i in range(maxg + 1)]
    m[0][0] *= 1.0 - lam_h * lam_a * rho
    m[0][1] *= 1.0 + lam_h * rho
    m[1][0] *= 1.0 + lam_a * rho
    m[1][1] *= 1.0 - rho
    s = sum(sum(row) for row in m)
    if s > 0:
        m = [[v / s for v in row] for row in m]
    return m


# ── Market readouts from the score matrix ────────────────────────────────────
def _R(m):
    return range(len(m))


def p_over(m, line: float) -> float:
    need = math.floor(line) + 1  # total strictly greater than line
    return sum(m[i][j] for i in _R(m) for j in _R(m) if i + j >= need)


def p_home(m) -> float:
    return sum(m[i][j] for i in _R(m) for j in _R(m) if i > j)


def p_away(m) -> float:
    return sum(m[i][j] for i in _R(m) for j in _R(m) if i < j)


def p_draw(m) -> float:
    return sum(m[i][i] for i in _R(m))


def p_btts(m) -> float:
    return sum(m[i][j] for i in range(1, len(m)) for j in range(1, len(m)))


def p_home_atleast(m, k: int) -> float:
    return sum(m[i][j] for i in _R(m) for j in _R(m) if i >= k)


def p_away_atleast(m, k: int) -> float:
    return sum(m[i][j] for i in _R(m) for j in _R(m) if j >= k)


def p_home_clean(m) -> float:  # away scores 0
    return sum(m[i][0] for i in _R(m))


def p_away_clean(m) -> float:  # home scores 0
    return sum(m[0][j] for j in _R(m))


def p_odd(m) -> float:
    return sum(m[i][j] for i in _R(m) for j in _R(m) if (i + j) % 2 == 1)


def _clamp(p: Optional[float]) -> Optional[float]:
    if p is None:
        return None
    return min(max(p, PROB_CLAMP[0]), PROB_CLAMP[1])


def _which_team(text: str, home: str, away: str) -> Optional[str]:
    """Decide if a team-specific pick refers to home or away."""
    t = (text or "").lower()
    if home and home.lower() in t:
        return "home"
    if away and away.lower() in t:
        return "away"
    return None


def market_prob(m, tip: Dict) -> Optional[float]:
    """Map a tip's settle_type/side/line to a probability. None if unsupported."""
    st = (tip.get("settle_type") or "").lower()
    side = (tip.get("side") or "").lower().strip()
    line = tip.get("line")
    home = tip.get("home_team") or ""
    away = tip.get("away_team") or ""
    pick = tip.get("pick") or ""

    try:
        if st == "goals_ou" and line is not None:
            over = p_over(m, float(line))
            return _clamp(over if side != "under" else 1.0 - over)

        if st == "btts":
            yes = p_btts(m)
            return _clamp(yes if side != "no" else 1.0 - yes)

        if st == "result":
            if side == "home":
                return _clamp(p_home(m))
            if side == "away":
                return _clamp(p_away(m))
            if side == "draw":
                return _clamp(p_draw(m))
            return None

        if st == "double_chance":
            h, d, a = p_home(m), p_draw(m), p_away(m)
            if side == "1x":
                return _clamp(h + d)
            if side == "12":
                return _clamp(h + a)
            if side == "x2":
                return _clamp(d + a)
            return None

        if st == "dnb":  # draw refunded -> renormalise over decisive outcomes
            h, a = p_home(m), p_away(m)
            denom = h + a
            if denom <= 0:
                return None
            if side == "home":
                return _clamp(h / denom)
            if side == "away":
                return _clamp(a / denom)
            return None

        if st == "win_to_nil":
            who = side if side in ("home", "away") else _which_team(pick, home, away)
            if who == "home":
                return _clamp(sum(m[i][0] for i in range(1, len(m))))  # home>0, away=0
            if who == "away":
                return _clamp(sum(m[0][j] for j in range(1, len(m))))  # away>0, home=0
            return None

        if st == "clean_sheet":
            who = side if side in ("home", "away") else _which_team(pick, home, away)
            if who == "home":
                return _clamp(p_home_clean(m))
            if who == "away":
                return _clamp(p_away_clean(m))
            return None

        if st == "odd_even":
            odd = p_odd(m)
            return _clamp(odd if side == "odd" else 1.0 - odd)

        if st == "team_goals" and line is not None:
            who = _which_team(pick, home, away)
            need = math.floor(float(line)) + 1
            if who == "home":
                base = p_home_atleast(m, need)
            elif who == "away":
                base = p_away_atleast(m, need)
            else:
                return None
            return _clamp(base if side != "under" else 1.0 - base)

        if st == "correct_score":
            # parse "H-A" from pick text, e.g. "2-1"
            import re
            mm = re.search(r"(\d+)\s*[-:]\s*(\d+)", pick)
            if mm:
                i, j = int(mm.group(1)), int(mm.group(2))
                if i < len(m) and j < len(m):
                    return _clamp(m[i][j])
            return None
    except Exception:
        return None

    return None


def enrich(tips: List[Dict], rates_map: Dict[str, Dict]) -> List[Dict]:
    """
    Attach Dixon-Coles model probabilities to tips and make them authoritative
    for supported markets. Tips whose fixture/market isn't supported keep their
    existing (AI-derived) probability and are marked prob_source='ai'.
    """
    if not rates_map:
        for t in tips:
            t["prob_source"] = "ai"
        return tips

    cache: Dict[str, list] = {}

    for t in tips:
        key = fixture_key(t.get("home_team", ""), t.get("away_team", ""))
        rates = rates_map.get(key)
        if not rates:
            t["prob_source"] = "ai"
            continue

        if key not in cache:
            lam_h, lam_a = expected_goals(rates)
            cache[key] = [score_matrix(lam_h, lam_a), lam_h, lam_a]
        m, lam_h, lam_a = cache[key]

        t["lambda_home"] = round(lam_h, 2)
        t["lambda_away"] = round(lam_a, 2)
        t["expected_goals"] = round(lam_h + lam_a, 2)

        p = market_prob(m, t)
        if p is None:
            t["prob_source"] = "ai"
            continue

        t["ai_probability"] = t.get("predicted_probability")
        t["model_probability"] = round(p, 4)
        t["predicted_probability"] = round(p, 4)
        t["fair_odds"] = round(1.0 / p, 2) if p > 0.01 else 50.0
        t["prob_source"] = "model"

    return tips
