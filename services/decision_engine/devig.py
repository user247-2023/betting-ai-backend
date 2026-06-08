"""
devig.py — remove the bookmaker margin ("vig"/"overround") from odds to recover
fair, comparable probabilities, and build a multi-bookmaker consensus.

Why this matters
----------------
Raw implied probabilities (1 / decimal_odds) for a market sum to MORE than 1 —
the excess is the bookmaker's built-in margin. If you compare the Dixon-Coles
probability against the *raw* implied probability you systematically understate
your edge (the book's margin masquerades as "the market"). De-vigging removes
that margin so the comparison is apples-to-apples.

The de-vigged price across several sharp books is the single best public
estimate of the true probability — the "market consensus". That consensus is
the honest yardstick a value bet must beat.

Methods (favourite-longshot bias handled increasingly well, top to bottom)
--------------------------------------------------------------------------
  multiplicative : fair_i = imp_i / sum(imp)               (simplest; proportional)
  additive       : fair_i = imp_i - (sum(imp)-1)/n         (equal absolute margin)
  power          : fair_i = imp_i**k,  k solved so sum=1   (good, robust)
  shin           : models a share z of insider money       (best for FL bias)

Dependencies: numpy, scipy only.

Author: Rollover Betting AI · market-intelligence layer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import brentq


@dataclass
class DevigResult:
    method: str
    fair_probabilities: list[float]   # sums to 1.0
    fair_odds: list[float]            # 1 / fair_probabilities
    overround: float                  # sum(1/odds); >1, the booksum
    margin_pct: float                 # (overround - 1) * 100
    insider_share: float | None = None  # z, Shin only

    def as_dict(self) -> dict:
        d = {
            "method": self.method,
            "fair_probabilities": [round(p, 5) for p in self.fair_probabilities],
            "fair_odds": [round(o, 4) for o in self.fair_odds],
            "overround": round(self.overround, 5),
            "margin_pct": round(self.margin_pct, 3),
        }
        if self.insider_share is not None:
            d["insider_share"] = round(self.insider_share, 5)
        return d


# --------------------------------------------------------------------------- #
#  Core de-vig methods                                                        #
# --------------------------------------------------------------------------- #
def _implied(odds: Sequence[float]) -> np.ndarray:
    o = np.asarray(odds, dtype=float)
    if np.any(o <= 1.0):
        raise ValueError("Decimal odds must all be > 1.0")
    return 1.0 / o


def devig_multiplicative(odds: Sequence[float]) -> DevigResult:
    imp = _implied(odds)
    booksum = float(imp.sum())
    fair = imp / booksum
    return _result("multiplicative", fair, booksum)


def devig_additive(odds: Sequence[float]) -> DevigResult:
    imp = _implied(odds)
    booksum = float(imp.sum())
    n = len(imp)
    fair = imp - (booksum - 1.0) / n
    fair = np.clip(fair, 1e-9, None)
    fair = fair / fair.sum()  # renormalise after clipping
    return _result("additive", fair, booksum)


def devig_power(odds: Sequence[float]) -> DevigResult:
    """Find k such that sum(imp_i ** k) == 1.  k > 1 shrinks the booksum to 1."""
    imp = _implied(odds)
    booksum = float(imp.sum())

    def f(k: float) -> float:
        return float(np.sum(imp ** k)) - 1.0

    # booksum > 1  => need k > 1; bracket generously
    k = brentq(f, 0.5, 10.0, maxiter=200)
    fair = imp ** k
    fair = fair / fair.sum()  # numerical tidy-up
    return _result("power", fair, booksum)


def devig_shin(odds: Sequence[float]) -> DevigResult:
    """
    Shin (1992) model: prices reflect a proportion z of insider traders, which
    produces the empirically-observed favourite-longshot bias. Solve for z so
    the recovered probabilities sum to 1.

        p_i = ( sqrt(z^2 + 4(1-z) * imp_i^2 / booksum) - z ) / ( 2(1-z) )
    """
    imp = _implied(odds)
    booksum = float(imp.sum())

    def p_of_z(z: float) -> np.ndarray:
        inside = z * z + 4.0 * (1.0 - z) * (imp ** 2) / booksum
        return (np.sqrt(inside) - z) / (2.0 * (1.0 - z))

    def g(z: float) -> float:
        return float(p_of_z(z).sum()) - 1.0

    # g(0) = booksum - 1 > 0 ; increasing z lowers the sum. Find the root.
    lo, hi = 0.0, 0.999
    if g(hi) > 0:  # extreme book; fall back to multiplicative
        return devig_multiplicative(odds)
    z = brentq(g, lo, hi, maxiter=200)
    fair = p_of_z(z)
    fair = np.clip(fair, 1e-9, None)
    fair = fair / fair.sum()
    r = _result("shin", fair, booksum)
    r.insider_share = float(z)
    return r


_METHODS = {
    "multiplicative": devig_multiplicative,
    "additive": devig_additive,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(odds: Sequence[float], method: str = "power") -> DevigResult:
    """Devig a single book's odds for one market. Default 'power' is robust."""
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(_METHODS)}.")
    return _METHODS[method](odds)


# --------------------------------------------------------------------------- #
#  Multi-bookmaker consensus                                                  #
# --------------------------------------------------------------------------- #
def market_consensus(
    books_odds: dict[str, Sequence[float]],
    method: str = "power",
    aggregate: str = "median",
    weights: dict[str, float] | None = None,
) -> dict:
    """
    De-vig each bookmaker's odds for the SAME market, then aggregate the fair
    probabilities into a single consensus. Sharp books (e.g. Pinnacle) can be
    up-weighted via `weights`.

    Parameters
    ----------
    books_odds : {bookmaker_name: [odds_outcome_1, odds_outcome_2, ...]}
        Every list must describe the same outcomes in the same order.
    aggregate  : "median" (robust, default), "mean", or "weighted_mean".
    weights    : {bookmaker_name: weight} — only used for "weighted_mean".

    Returns a dict with the consensus probabilities, consensus fair odds, the
    per-book devigged probabilities, and the average market margin.
    """
    if not books_odds:
        raise ValueError("books_odds is empty")

    lengths = {len(v) for v in books_odds.values()}
    if len(lengths) != 1:
        raise ValueError("All bookmakers must quote the same number of outcomes")
    n_outcomes = lengths.pop()

    per_book: dict[str, DevigResult] = {}
    margins: list[float] = []
    matrix = []  # rows = books, cols = outcomes
    names = []
    for book, odds in books_odds.items():
        res = devig(odds, method=method)
        per_book[book] = res
        margins.append(res.margin_pct)
        matrix.append(res.fair_probabilities)
        names.append(book)
    arr = np.asarray(matrix)  # (n_books, n_outcomes)

    if aggregate == "median":
        consensus = np.median(arr, axis=0)
    elif aggregate == "mean":
        consensus = arr.mean(axis=0)
    elif aggregate == "weighted_mean":
        w = np.array([(weights or {}).get(b, 1.0) for b in names], dtype=float)
        w = w / w.sum()
        consensus = (arr * w[:, None]).sum(axis=0)
    else:
        raise ValueError("aggregate must be 'median', 'mean', or 'weighted_mean'")

    consensus = consensus / consensus.sum()  # renormalise post-aggregation
    fair_odds = [float(1.0 / p) if p > 0 else float("inf") for p in consensus]

    return {
        "method": method,
        "aggregate": aggregate,
        "n_books": len(books_odds),
        "n_outcomes": n_outcomes,
        "consensus_probabilities": [round(float(p), 5) for p in consensus],
        "consensus_fair_odds": [round(o, 4) for o in fair_odds],
        "avg_market_margin_pct": round(float(np.mean(margins)), 3),
        "per_book": {b: per_book[b].as_dict() for b in names},
    }


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _result(method: str, fair: np.ndarray, booksum: float) -> DevigResult:
    fair = np.asarray(fair, dtype=float)
    return DevigResult(
        method=method,
        fair_probabilities=[float(p) for p in fair],
        fair_odds=[float(1.0 / p) if p > 0 else float("inf") for p in fair],
        overround=float(booksum),
        margin_pct=float((booksum - 1.0) * 100.0),
    )
