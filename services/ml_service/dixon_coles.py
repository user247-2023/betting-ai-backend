"""
dixon_coles.py — Dixon-Coles bivariate Poisson goals model.

This is the mathematical core for all GOALS-based markets (1X2, Over/Under,
BTTS, Correct Score). It is the model professional football modellers use as
their baseline, because a single fitted model produces a full score matrix from
which *every* goals market derives consistently — no more contradictory
standalone predictions (e.g. an O/U 2.5 that disagrees with the BTTS price).

How it works
------------
Each team has an attack rating and a defence rating. For a fixture:

    log(lambda_home) = intercept + home_adv + attack[home] - defence[away]
    log(lambda_away) = intercept           + attack[away] - defence[home]

(`defence` is signed so that a *higher* value = a *better* defence, i.e. it
subtracts from the opponent's expected goals.)

Independent Poisson goals (home ~ Poisson(lambda_home), away ~ Poisson(lambda_away))
systematically *underestimate* draws (0-0, 1-1) and *overestimate* the 1-0/0-1
scorelines. Dixon & Coles (1997) correct this low-score dependence with a tau
adjustment governed by a single parameter rho (typically small and negative):

    tau(0,0) = 1 - lambda_h * lambda_a * rho
    tau(0,1) = 1 + lambda_h * rho
    tau(1,0) = 1 + lambda_a * rho
    tau(1,1) = 1 - rho
    tau(x,y) = 1   otherwise

    P(X=x, Y=y) = tau(x,y) * Poisson(x; lambda_h) * Poisson(y; lambda_a)

Recent form matters more than old form, so each match m carries an exponential
time weight phi_m = exp(-xi * age_in_days), controlled by a half-life. Parameters
are fit by weighted maximum likelihood with a small ridge penalty on the
attack/defence vectors (resolves the additive non-identifiability and shrinks
ratings for teams with few matches).

Dependencies: numpy, scipy only. No storage coupling — `fit` takes plain dicts.

Author: Rollover Betting AI · goals-model core
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


# --------------------------------------------------------------------------- #
#  Low-score dependence correction                                            #
# --------------------------------------------------------------------------- #
def _tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles tau adjustment for the (x, y) scoreline."""
    if x == 0 and y == 0:
        return 1.0 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1.0 + lam_h * rho
    if x == 1 and y == 0:
        return 1.0 + lam_a * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


@dataclass
class Match:
    """A single observed result fed to the model."""
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    match_date: date | None = None  # used for time-decay weighting


@dataclass
class MarketProbabilities:
    """All goals-market probabilities derived from one fitted score matrix."""
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float

    # 1X2
    home_win: float
    draw: float
    away_win: float

    # Over/Under (Yes = "over")
    over_1_5: float
    under_1_5: float
    over_2_5: float
    under_2_5: float
    over_3_5: float
    under_3_5: float

    # Both teams to score
    btts_yes: float
    btts_no: float

    # Most-likely scorelines: list of (home, away, prob) sorted desc
    top_scores: list[tuple[int, int, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "expected_goals": {
                "home": round(self.expected_home_goals, 3),
                "away": round(self.expected_away_goals, 3),
                "total": round(self.expected_home_goals + self.expected_away_goals, 3),
            },
            "1x2": {
                "home": round(self.home_win, 4),
                "draw": round(self.draw, 4),
                "away": round(self.away_win, 4),
            },
            "over_under": {
                "1.5": {"over": round(self.over_1_5, 4), "under": round(self.under_1_5, 4)},
                "2.5": {"over": round(self.over_2_5, 4), "under": round(self.under_2_5, 4)},
                "3.5": {"over": round(self.over_3_5, 4), "under": round(self.under_3_5, 4)},
            },
            "btts": {"yes": round(self.btts_yes, 4), "no": round(self.btts_no, 4)},
            "correct_score": [
                {"home": h, "away": a, "prob": round(p, 4)} for (h, a, p) in self.top_scores
            ],
        }
        return d


# --------------------------------------------------------------------------- #
#  Pure lambda -> markets helpers (no fitted model required)                  #
#                                                                             #
#  These are split out so any source of expected goals can drive the full     #
#  market set — not just team ratings looked up from a fitted model. The      #
#  Tier-3 xG adapter, for example, blends goals-based lambdas with xG-based   #
#  lambdas and feeds the result straight through build_score_matrix ->        #
#  markets_from_matrix to get a coherent MarketProbabilities object.          #
# --------------------------------------------------------------------------- #
def build_score_matrix(
    lam_h: float, lam_a: float, rho: float, max_goals: int = 10
) -> np.ndarray:
    """
    Normalised score matrix P[i, j] = P(home i, away j) for arbitrary lambdas.

    Applies the Dixon-Coles tau correction to the four low-score cells, clips
    any tiny negatives the correction can introduce, and renormalises so the
    matrix sums to 1 (tau + truncation perturb the total mass).
    """
    g = np.arange(max_goals + 1)
    ph = poisson.pmf(g, lam_h)
    pa = poisson.pmf(g, lam_a)
    m = np.outer(ph, pa)  # independent baseline

    m[0, 0] *= 1.0 - lam_h * lam_a * rho
    m[0, 1] *= 1.0 + lam_h * rho
    m[1, 0] *= 1.0 + lam_a * rho
    m[1, 1] *= 1.0 - rho

    m = np.clip(m, 0.0, None)
    total = m.sum()
    if total > 0:
        m /= total
    return m


def markets_from_matrix(
    m: np.ndarray,
    home_team: str,
    away_team: str,
    lam_h: float,
    lam_a: float,
    top_n_scores: int = 5,
) -> MarketProbabilities:
    """Derive every goals market from a (already normalised) score matrix."""
    n = m.shape[0]
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]

    home_win = float(m[i > j].sum())
    draw = float(np.trace(m))
    away_win = float(m[i < j].sum())

    total = i + j

    def over(line_floor: int) -> float:
        return float(m[total >= line_floor].sum())

    over_1_5 = over(2)
    over_2_5 = over(3)
    over_3_5 = over(4)

    btts_yes = float(m[1:, 1:].sum())

    flat = [(int(r), int(c), float(m[r, c])) for r in range(n) for c in range(n)]
    flat.sort(key=lambda t: t[2], reverse=True)
    top = flat[:top_n_scores]

    return MarketProbabilities(
        home_team=home_team,
        away_team=away_team,
        expected_home_goals=lam_h,
        expected_away_goals=lam_a,
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        over_1_5=over_1_5,
        under_1_5=1.0 - over_1_5,
        over_2_5=over_2_5,
        under_2_5=1.0 - over_2_5,
        over_3_5=over_3_5,
        under_3_5=1.0 - over_3_5,
        btts_yes=btts_yes,
        btts_no=1.0 - btts_yes,
        top_scores=top,
    )


class DixonColesModel:
    """
    Fit on historical results, then predict any goals market for any fixture.

    Parameters
    ----------
    max_goals:
        Truncation point for the score matrix (0..max_goals each side). 10 is
        ample — P(>10 goals for one team) is vanishingly small.
    half_life_days:
        Exponential time-decay half-life. A match this many days old counts
        half as much as a match today. ~180 days is a sensible football default;
        shorter (~90) reacts faster to form, longer (~365) is steadier.
    ridge:
        L2 penalty on attack/defence ratings. Stabilises the fit and shrinks
        small-sample teams toward average. 0.01-0.05 works well.
    rho_bounds:
        Hard bounds on the dependence parameter. Optimal rho is small (~ -0.1),
        so [-0.2, 0.2] never binds in practice but guards tau positivity.
    """

    def __init__(
        self,
        max_goals: int = 10,
        half_life_days: float = 180.0,
        ridge: float = 0.02,
        rho_bounds: tuple[float, float] = (-0.2, 0.2),
    ) -> None:
        self.max_goals = max_goals
        self.half_life_days = half_life_days
        self.ridge = ridge
        self.rho_bounds = rho_bounds

        self.teams: list[str] = []
        self._idx: dict[str, int] = {}
        self.intercept: float = 0.0
        self.home_adv: float = 0.0
        self.rho: float = 0.0
        self.attack: np.ndarray | None = None
        self.defence: np.ndarray | None = None
        self.n_matches: int = 0
        self.fitted: bool = False
        self.log_likelihood: float | None = None

    # ------------------------------------------------------------------ #
    #  Fitting                                                           #
    # ------------------------------------------------------------------ #
    def _time_weights(self, dates: Sequence[date | None], ref: date) -> np.ndarray:
        if self.half_life_days is None or self.half_life_days <= 0:
            return np.ones(len(dates))
        xi = math.log(2.0) / self.half_life_days
        w = np.ones(len(dates))
        for i, d in enumerate(dates):
            if d is None:
                w[i] = 1.0
            else:
                age = max((ref - d).days, 0)
                w[i] = math.exp(-xi * age)
        return w

    def fit(
        self,
        matches: Iterable[Match | dict],
        reference_date: date | None = None,
    ) -> "DixonColesModel":
        rows = [self._coerce(m) for m in matches]
        if len(rows) < 5:
            raise ValueError(
                f"Need at least 5 matches to fit; got {len(rows)}. "
                "Let the dataset builder accumulate more results."
            )

        teams = sorted({r.home_team for r in rows} | {r.away_team for r in rows})
        self.teams = teams
        self._idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        self.n_matches = len(rows)

        hi = np.array([self._idx[r.home_team] for r in rows])
        ai = np.array([self._idx[r.away_team] for r in rows])
        hg = np.array([r.home_goals for r in rows], dtype=float)
        ag = np.array([r.away_goals for r in rows], dtype=float)

        ref = reference_date or max(
            (r.match_date for r in rows if r.match_date is not None),
            default=date.today(),
        )
        weights = self._time_weights([r.match_date for r in rows], ref)

        # Param vector: [intercept, home_adv, rho, attack(n), defence(n)]
        def unpack(p):
            intercept = p[0]
            home_adv = p[1]
            rho = p[2]
            attack = p[3 : 3 + n]
            defence = p[3 + n : 3 + 2 * n]
            return intercept, home_adv, rho, attack, defence

        def neg_log_likelihood(p):
            intercept, home_adv, rho, attack, defence = unpack(p)

            log_lh = intercept + home_adv + attack[hi] - defence[ai]
            log_la = intercept + attack[ai] - defence[hi]
            lam_h = np.exp(log_lh)
            lam_a = np.exp(log_la)

            # Poisson log-pmf for each side (vectorised)
            ll_h = hg * log_lh - lam_h - _gammaln_int(hg)
            ll_a = ag * log_la - lam_a - _gammaln_int(ag)

            # tau correction only touches the four low scorelines
            tau = np.ones(len(rows))
            m00 = (hg == 0) & (ag == 0)
            m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0)
            m11 = (hg == 1) & (ag == 1)
            tau[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
            tau[m01] = 1.0 + lam_h[m01] * rho
            tau[m10] = 1.0 + lam_a[m10] * rho
            tau[m11] = 1.0 - rho
            tau = np.clip(tau, 1e-10, None)  # guard log of non-positive

            ll = weights * (np.log(tau) + ll_h + ll_a)
            penalty = self.ridge * (np.sum(attack**2) + np.sum(defence**2))
            return -np.sum(ll) + penalty

        x0 = np.concatenate(
            [
                [math.log(max(hg.mean() + ag.mean(), 0.2) / 2.0)],  # intercept
                [0.2],   # home advantage prior
                [-0.05], # rho prior (slight negative)
                np.zeros(n),  # attack
                np.zeros(n),  # defence
            ]
        )
        bounds = (
            [(None, None), (None, None), self.rho_bounds]
            + [(-3.0, 3.0)] * n
            + [(-3.0, 3.0)] * n
        )

        res = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        intercept, home_adv, rho, attack, defence = unpack(res.x)

        # Recentre attack/defence to mean 0 for interpretability, folding the
        # shift into the intercept so lambda predictions are unchanged.
        a_mean = attack.mean()
        d_mean = defence.mean()
        attack = attack - a_mean
        defence = defence - d_mean
        intercept = intercept + a_mean - d_mean

        self.intercept = float(intercept)
        self.home_adv = float(home_adv)
        self.rho = float(rho)
        self.attack = attack
        self.defence = defence
        self.log_likelihood = float(-res.fun)
        self.fitted = True
        return self

    # ------------------------------------------------------------------ #
    #  Prediction                                                        #
    # ------------------------------------------------------------------ #
    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        self._require_fit()
        self._require_team(home_team)
        self._require_team(away_team)
        h = self._idx[home_team]
        a = self._idx[away_team]
        lam_h = math.exp(self.intercept + self.home_adv + self.attack[h] - self.defence[a])
        lam_a = math.exp(self.intercept + self.attack[a] - self.defence[h])
        return lam_h, lam_a

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Normalised P[i, j] = P(home scores i, away scores j)."""
        lam_h, lam_a = self.expected_goals(home_team, away_team)
        return build_score_matrix(lam_h, lam_a, self.rho, self.max_goals)

    def predict_markets(
        self, home_team: str, away_team: str, top_n_scores: int = 5
    ) -> MarketProbabilities:
        lam_h, lam_a = self.expected_goals(home_team, away_team)
        m = build_score_matrix(lam_h, lam_a, self.rho, self.max_goals)
        return markets_from_matrix(
            m, home_team, away_team, lam_h, lam_a, top_n_scores=top_n_scores
        )

    def predict_markets_from_lambdas(
        self,
        home_team: str,
        away_team: str,
        lam_h: float,
        lam_a: float,
        top_n_scores: int = 5,
    ) -> MarketProbabilities:
        """
        Build markets from externally supplied expected goals.

        Used by the xG adapter (and any other lambda source): the caller blends
        the model's own lambdas with an xG-based estimate, then passes the
        blended pair here. rho from the fitted model still governs low-score
        dependence, so the output stays coherent with the rest of the system.
        """
        self._require_fit()
        m = build_score_matrix(lam_h, lam_a, self.rho, self.max_goals)
        return markets_from_matrix(
            m, home_team, away_team, lam_h, lam_a, top_n_scores=top_n_scores
        )

    # ------------------------------------------------------------------ #
    #  Inspection ("why this tip")                                       #
    # ------------------------------------------------------------------ #
    def team_ratings(self) -> list[dict]:
        """Per-team attack/defence ratings, sorted by overall strength."""
        self._require_fit()
        out = []
        for t in self.teams:
            i = self._idx[t]
            out.append(
                {
                    "team": t,
                    "attack": round(float(self.attack[i]), 4),
                    "defence": round(float(self.defence[i]), 4),
                    "overall": round(float(self.attack[i] + self.defence[i]), 4),
                }
            )
        out.sort(key=lambda r: r["overall"], reverse=True)
        return out

    def summary(self) -> dict:
        self._require_fit()
        return {
            "fitted": True,
            "n_teams": len(self.teams),
            "n_matches": self.n_matches,
            "intercept": round(self.intercept, 4),
            "home_advantage": round(self.home_adv, 4),
            "rho": round(self.rho, 4),
            "half_life_days": self.half_life_days,
            "log_likelihood": round(self.log_likelihood, 2) if self.log_likelihood else None,
            "implied_avg_home_goals": round(math.exp(self.intercept + self.home_adv), 3),
            "implied_avg_away_goals": round(math.exp(self.intercept), 3),
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce(m: Match | dict) -> Match:
        if isinstance(m, Match):
            return m
        d = m.get("match_date") or m.get("date")
        if isinstance(d, str):
            d = _parse_date(d)
        elif isinstance(d, datetime):
            d = d.date()
        return Match(
            home_team=str(m["home_team"]).strip(),
            away_team=str(m["away_team"]).strip(),
            home_goals=int(m["home_goals"]),
            away_goals=int(m["away_goals"]),
            match_date=d,
        )

    def _require_fit(self) -> None:
        if not self.fitted:
            raise RuntimeError("Model is not fitted. Call .fit(matches) first.")

    def _require_team(self, team: str) -> None:
        if team not in self._idx:
            raise KeyError(
                f"Unknown team '{team}'. Model knows {len(self.teams)} teams; "
                "it can only predict for teams seen during fit."
            )


def _gammaln_int(k: np.ndarray) -> np.ndarray:
    """log(k!) for non-negative integer arrays, via gammaln(k+1)."""
    from scipy.special import gammaln
    return gammaln(k + 1.0)


def _parse_date(s: str) -> date | None:
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%d/%m/%Y"):
        try:
            return datetime.strptime(s[: len(fmt) + 6], fmt).date()
        except (ValueError, TypeError):
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
#  Optional SQLite loader (matches the project's football.db convention)      #
# --------------------------------------------------------------------------- #
def load_matches_from_sqlite(
    db_path: str,
    table: str = "matches",
    league_id: int | None = None,
    columns: dict | None = None,
) -> list[Match]:
    """
    Convenience loader for `database/football.db`. Adjust `columns` to match
    your actual schema if it differs from the defaults below.

    Default expected columns:
        home_team, away_team, home_goals, away_goals, match_date  (+ optional league_id)
    """
    import sqlite3

    cols = columns or {
        "home_team": "home_team",
        "away_team": "away_team",
        "home_goals": "home_goals",
        "away_goals": "away_goals",
        "match_date": "match_date",
        "league_id": "league_id",
    }
    select = ", ".join(
        [cols["home_team"], cols["away_team"], cols["home_goals"],
         cols["away_goals"], cols["match_date"]]
    )
    sql = f"SELECT {select} FROM {table}"
    params: tuple = ()
    if league_id is not None and "league_id" in cols:
        sql += f" WHERE {cols['league_id']} = ?"
        params = (league_id,)

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    out: list[Match] = []
    for r in rows:
        if r[2] is None or r[3] is None:
            continue  # skip unsettled fixtures
        out.append(
            DixonColesModel._coerce(
                {
                    "home_team": r[0],
                    "away_team": r[1],
                    "home_goals": r[2],
                    "away_goals": r[3],
                    "match_date": r[4],
                }
            )
        )
    return out
