"""
dixon_coles_fit.py — Properly fitted Dixon-Coles strength model.

Upgrades the tip probabilities from a simple last-6 shrinkage estimate to a real
statistical model: each team gets an ATTACK and DEFENSE rating learned by
maximum-likelihood over the whole match database, with recent matches weighted
more heavily (time decay) and a global home-advantage term. A low-score
correlation term (rho) is fitted on top.

Fitting is done offline (admin endpoint / script) and the result is saved to a
small JSON file that is loaded at runtime — so requests never pay the fit cost.
If the JSON is missing or a team is unknown, callers fall back to the simple
shrinkage model, so nothing ever breaks.

expected goals:
    lam_home = exp(mu + home_adv*(not neutral) + att[home] + def[away])
    lam_away = exp(mu +                            att[away] + def[home])
"""
import json
import math
import os
import sqlite3
from datetime import date, datetime
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

_DIR = os.path.dirname(__file__)
_DB_PATH = os.path.join(_DIR, "..", "..", "database", "football.db")
_PARAMS_PATH = os.path.join(_DIR, "..", "..", "database", "dc_params.json")

# fitting knobs
WINDOW_DAYS = 730      # only learn from the last ~2 years
MIN_MATCHES = 4        # teams with fewer games are dropped (fall back to shrinkage)
HALF_LIFE_DAYS = 365   # a match 1 year old counts half as much
RIDGE = 0.05           # L2 on ratings (stability + soft identifiability)
RHO_GRID = np.linspace(-0.20, 0.02, 23)
LAM_CLIP = (0.18, 4.6)

_CACHE: Optional[Dict] = None  # loaded params


# ─────────────────────────── fitting ────────────────────────────────────────
def _load_matches(conn, as_of: str):
    cutoff = (datetime.strptime(as_of[:10], "%Y-%m-%d").toordinal() - WINDOW_DAYS)
    rows = conn.execute(
        "SELECT substr(date,1,10) d, home_team, away_team, home_goals, away_goals "
        "FROM fixtures WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL "
        "AND substr(date,1,10) < ? ORDER BY date", (as_of,)
    ).fetchall()
    out = []
    for d, h, a, hg, ag in rows:
        try:
            o = datetime.strptime(d, "%Y-%m-%d").toordinal()
        except Exception:
            continue
        if o < cutoff:
            continue
        out.append((o, h, a, int(hg), int(ag)))
    return out


def fit_and_save(as_of: Optional[str] = None, db_path: str = _DB_PATH,
                 params_path: str = _PARAMS_PATH) -> Dict:
    """Fit the model on matches before `as_of` (default today) and save JSON."""
    as_of = as_of or date.today().isoformat()
    conn = sqlite3.connect(db_path)
    try:
        matches = _load_matches(conn, as_of)
    finally:
        conn.close()
    if len(matches) < 200:
        raise ValueError(f"Not enough matches to fit ({len(matches)}).")

    # team counts -> keep teams with enough games
    counts: Dict[str, int] = {}
    for _o, h, a, _hg, _ag in matches:
        counts[h] = counts.get(h, 0) + 1
        counts[a] = counts.get(a, 0) + 1
    teams = sorted(t for t, c in counts.items() if c >= MIN_MATCHES)
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)

    # keep matches where both teams are kept
    m = [r for r in matches if r[1] in idx and r[2] in idx]
    latest = max(r[0] for r in m)
    decay = math.log(2) / HALF_LIFE_DAYS

    hi = np.array([idx[r[1]] for r in m])
    ai = np.array([idx[r[2]] for r in m])
    gh = np.array([r[3] for r in m], dtype=float)
    ga = np.array([r[4] for r in m], dtype=float)
    w  = np.exp(-decay * (latest - np.array([r[0] for r in m], dtype=float)))
    neutral = np.zeros(len(m))  # club fixtures assumed home/away; intl handled at predict

    # params: [att(T), def(T), mu, home]
    def unpack(p):
        return p[:T], p[T:2*T], p[2*T], p[2*T+1]

    def nll_grad(p):
        att, dfn, mu, home = unpack(p)
        llh = mu + home + att[hi] + dfn[ai]
        lla = mu + att[ai] + dfn[hi]
        lam_h = np.exp(np.clip(llh, -3, 3))
        lam_a = np.exp(np.clip(lla, -3, 3))
        rh = w * (lam_h - gh)          # dNLL/d log_lam_h
        ra = w * (lam_a - ga)
        nll = np.sum(w * (lam_h - gh*llh + lam_a - ga*lla))
        nll += RIDGE * (np.sum(att*att) + np.sum(dfn*dfn))
        g_att = np.zeros(T); g_def = np.zeros(T)
        np.add.at(g_att, hi, rh); np.add.at(g_att, ai, ra)
        np.add.at(g_def, ai, rh); np.add.at(g_def, hi, ra)
        g_att += 2*RIDGE*att; g_def += 2*RIDGE*dfn
        g_mu = np.sum(rh + ra)
        g_home = np.sum(rh)
        return nll, np.concatenate([g_att, g_def, [g_mu, g_home]])

    p0 = np.zeros(2*T + 2)
    p0[2*T] = math.log(max(gh.mean(), 0.5))      # mu init
    p0[2*T+1] = 0.2                               # home init
    res = minimize(nll_grad, p0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 300, "maxfun": 400})
    att, dfn, mu, home = unpack(res.x)
    # center for identifiability, fold into mu
    mu = mu + att.mean() + dfn.mean()
    att = att - att.mean(); dfn = dfn - dfn.mean()

    # fit rho on a grid using the fixed lambdas (cheap)
    llh = mu + home + att[hi] + dfn[ai]
    lla = mu + att[ai] + dfn[hi]
    lam_h = np.exp(np.clip(llh, -3, 3)); lam_a = np.exp(np.clip(lla, -3, 3))
    rho = _fit_rho(gh, ga, lam_h, lam_a, w)

    params = {
        "fitted_at": datetime.utcnow().isoformat() + "Z",
        "as_of": as_of, "n_matches": len(m), "n_teams": T,
        "mu": float(mu), "home_adv": float(home), "rho": float(rho),
        "window_days": WINDOW_DAYS, "half_life_days": HALF_LIFE_DAYS,
        "teams": {t: [float(att[idx[t]]), float(dfn[idx[t]])] for t in teams},
    }
    with open(params_path, "w") as f:
        json.dump(params, f)
    global _CACHE
    _CACHE = params
    return {"n_matches": len(m), "n_teams": T, "mu": round(float(mu), 3),
            "home_adv": round(float(home), 3), "rho": round(float(rho), 3),
            "fitted_at": params["fitted_at"]}


def _dc_tau(gh, ga, lam_h, lam_a, rho):
    tau = np.ones_like(lam_h)
    m00 = (gh == 0) & (ga == 0); tau[m00] = 1 - lam_h[m00]*lam_a[m00]*rho
    m01 = (gh == 0) & (ga == 1); tau[m01] = 1 + lam_h[m01]*rho
    m10 = (gh == 1) & (ga == 0); tau[m10] = 1 + lam_a[m10]*rho
    m11 = (gh == 1) & (ga == 1); tau[m11] = 1 - rho
    return np.clip(tau, 1e-6, None)


def _fit_rho(gh, ga, lam_h, lam_a, w):
    best_rho, best = -0.10, math.inf
    for rho in RHO_GRID:
        tau = _dc_tau(gh, ga, lam_h, lam_a, rho)
        nll = -np.sum(w * np.log(tau))
        if nll < best:
            best, best_rho = nll, rho
    return float(best_rho)


# ─────────────────────────── runtime use ────────────────────────────────────
def load(params_path: str = _PARAMS_PATH) -> Optional[Dict]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        with open(params_path) as f:
            _CACHE = json.load(f)
    except Exception:
        _CACHE = None
    return _CACHE


def is_ready() -> bool:
    return load() is not None


def lambdas(home: str, away: str, neutral: bool = False) -> Optional[Tuple[float, float]]:
    """Fitted expected goals for a fixture, or None if unavailable/unknown teams."""
    p = load()
    if not p:
        return None
    teams = p["teams"]
    if home not in teams or away not in teams:
        return None
    ah, dh = teams[home]
    aa, da = teams[away]
    mu, hadv = p["mu"], p["home_adv"]
    lam_h = math.exp(mu + (0.0 if neutral else hadv) + ah + da)
    lam_a = math.exp(mu + aa + dh)
    lo, hi = LAM_CLIP
    return min(max(lam_h, lo), hi), min(max(lam_a, lo), hi)


def rho() -> float:
    p = load()
    return float(p["rho"]) if p else -0.10
