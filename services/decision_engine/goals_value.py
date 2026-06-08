"""
goals_value.py — turn a fitted Dixon-Coles model + live bookmaker odds into
ranked, risk-graded value bets for goals markets.

Pipeline for one fixture:
    1. Dixon-Coles      -> model probability for each goals-market selection
    2. devig (consensus)-> fair market probability (the honest yardstick)
    3. value            -> edge = model_prob * best_odds - 1
    4. Kelly            -> quarter-Kelly stake, capped (matches project policy)
    5. grade            -> risk tier from confidence + edge + market agreement

A selection is flagged a VALUE BET only when the model's probability clears the
de-vigged market probability by a margin (default 2%) AND the raw value edge is
positive (default >= 5%). This double guard avoids "phantom edges" that are
really just the bookmaker's margin.

Consumes: services.ml_service.dixon_coles, services.decision_engine.devig
Author: Rollover Betting AI · decision engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from services.ml_service.dixon_coles import DixonColesModel, MarketProbabilities
from services.decision_engine.devig import devig, market_consensus


# Two-outcome markets we read straight off the Dixon-Coles matrix.
# Each entry: selection_key -> (label, attribute_on_MarketProbabilities)
_GOALS_SELECTIONS = {
    "over_1_5":  ("Over 1.5",  "over_1_5"),
    "under_1_5": ("Under 1.5", "under_1_5"),
    "over_2_5":  ("Over 2.5",  "over_2_5"),
    "under_2_5": ("Under 2.5", "under_2_5"),
    "over_3_5":  ("Over 3.5",  "over_3_5"),
    "under_3_5": ("Under 3.5", "under_3_5"),
    "btts_yes":  ("BTTS Yes",  "btts_yes"),
    "btts_no":   ("BTTS No",   "btts_no"),
    "home":      ("Home Win",  "home_win"),
    "draw":      ("Draw",      "draw"),
    "away":      ("Away Win",  "away_win"),
}

# Which selections pair into the same market (for de-vig, you need both sides).
_MARKET_PAIRS = {
    "ou_1_5": ("over_1_5", "under_1_5"),
    "ou_2_5": ("over_2_5", "under_2_5"),
    "ou_3_5": ("over_3_5", "under_3_5"),
    "btts":   ("btts_yes", "btts_no"),
    "1x2":    ("home", "draw", "away"),
}


@dataclass
class ValueBet:
    selection: str
    label: str
    market: str
    model_prob: float
    market_fair_prob: float       # de-vigged consensus
    best_odds: float              # best available price (line-shopped)
    best_book: str | None
    implied_prob: float           # 1 / best_odds (raw)
    edge: float                   # model_prob * best_odds - 1
    prob_gap: float               # model_prob - market_fair_prob
    kelly_fraction: float         # raw full-Kelly
    stake_fraction: float         # capped quarter-Kelly (of bankroll)
    is_value: bool
    risk_tier: str                # "low" | "medium" | "high" | "skip"

    def as_dict(self) -> dict:
        return {
            "selection": self.selection,
            "label": self.label,
            "market": self.market,
            "model_prob": round(self.model_prob, 4),
            "market_fair_prob": round(self.market_fair_prob, 4),
            "best_odds": round(self.best_odds, 3),
            "best_book": self.best_book,
            "implied_prob": round(self.implied_prob, 4),
            "edge": round(self.edge, 4),
            "edge_pct": round(self.edge * 100, 2),
            "prob_gap": round(self.prob_gap, 4),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "stake_fraction": round(self.stake_fraction, 4),
            "is_value": self.is_value,
            "risk_tier": self.risk_tier,
        }


def kelly_fraction(prob: float, odds: float) -> float:
    """Full-Kelly fraction. b = odds-1, p = prob, q = 1-p. Negative => no bet."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    p, q = prob, 1.0 - prob
    return (b * p - q) / b


def find_goals_value(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
    odds_by_market: dict[str, dict[str, list[float]] | list[float]],
    *,
    devig_method: str = "power",
    consensus_aggregate: str = "median",
    min_edge: float = 0.05,
    min_prob_gap: float = 0.02,
    kelly_multiplier: float = 0.25,
    max_stake: float = 0.02,
    bankroll: float | None = None,
) -> dict:
    """
    Evaluate every goals market for a fixture and return ranked value bets.

    odds_by_market structure (per market key from _MARKET_PAIRS):
        Single book:   {"ou_2_5": [over_odds, under_odds], ...}
        Multiple books:{"ou_2_5": {"Pinnacle": [o,u], "Bet365": [o,u]}, ...}
        For 1x2 the lists are [home, draw, away].

    Returns a dict with the model's full market view, per-selection value
    analysis, and the ranked list of qualifying value bets.
    """
    markets: MarketProbabilities = model.predict_markets(home_team, away_team)

    all_bets: list[ValueBet] = []
    for market_key, selection_keys in _MARKET_PAIRS.items():
        if market_key not in odds_by_market:
            continue
        raw = odds_by_market[market_key]

        # Normalise to multi-book form, then build consensus + best price.
        if isinstance(raw, dict):
            books = {b: list(o) for b, o in raw.items()}
        else:
            books = {"book": list(raw)}

        # Fair market probability per selection (de-vigged consensus).
        cons = market_consensus(books, method=devig_method, aggregate=consensus_aggregate)
        fair_probs = cons["consensus_probabilities"]

        # Best available odds per selection (line shopping = max across books).
        n = len(selection_keys)
        best_odds = [0.0] * n
        best_book: list[str | None] = [None] * n
        for b, o in books.items():
            for i in range(n):
                if o[i] > best_odds[i]:
                    best_odds[i] = o[i]
                    best_book[i] = b if b != "book" else None

        for i, sel in enumerate(selection_keys):
            label, attr = _GOALS_SELECTIONS[sel]
            model_p = float(getattr(markets, attr))
            fair_p = float(fair_probs[i])
            odds = float(best_odds[i])
            edge = model_p * odds - 1.0
            gap = model_p - fair_p
            kf = kelly_fraction(model_p, odds)
            stake = max(min(kf * kelly_multiplier, max_stake), 0.0)
            is_value = (edge >= min_edge) and (gap >= min_prob_gap) and (kf > 0)
            tier = _grade(model_p, edge, gap, is_value)

            all_bets.append(
                ValueBet(
                    selection=sel,
                    label=label,
                    market=market_key,
                    model_prob=model_p,
                    market_fair_prob=fair_p,
                    best_odds=odds,
                    best_book=best_book[i],
                    implied_prob=1.0 / odds if odds > 0 else 0.0,
                    edge=edge,
                    prob_gap=gap,
                    kelly_fraction=kf,
                    stake_fraction=stake,
                    is_value=is_value,
                    risk_tier=tier,
                )
            )

    value_bets = sorted(
        [b for b in all_bets if b.is_value], key=lambda b: b.edge, reverse=True
    )

    out = {
        "fixture": f"{home_team} vs {away_team}",
        "model": markets.as_dict(),
        "params": {
            "devig_method": devig_method,
            "consensus_aggregate": consensus_aggregate,
            "min_edge": min_edge,
            "min_prob_gap": min_prob_gap,
            "kelly_multiplier": kelly_multiplier,
            "max_stake": max_stake,
        },
        "value_bets": [b.as_dict() for b in value_bets],
        "all_selections": [b.as_dict() for b in all_bets],
        "n_value_bets": len(value_bets),
    }
    if bankroll is not None:
        for b in out["value_bets"]:
            b["stake_amount"] = round(bankroll * b["stake_fraction"], 2)
    return out


def _grade(model_prob: float, edge: float, gap: float, is_value: bool) -> str:
    if not is_value:
        return "skip"
    # Higher model confidence + bigger edge = lower risk.
    if model_prob >= 0.70 and edge >= 0.08:
        return "low"
    if model_prob >= 0.55 and edge >= 0.06:
        return "medium"
    return "high"
