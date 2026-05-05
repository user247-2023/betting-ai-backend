"""
value_engine.py - Calculates expected value vs bookmaker odds.
value = (probability * odds) - 1
Only flags bets with positive edge.
"""
from typing import Dict, List


class ValueEngine:
    """Identifies value bets where model probability exceeds bookmaker implied probability."""

    def __init__(self, min_threshold: float = 0.05, min_prob: float = 0.55,
                 min_odds: float = 1.20, max_odds: float = 3.50,
                 kelly_fraction: float = 0.25, max_stake_pct: float = 2.0):
        self.min_threshold = min_threshold
        self.min_prob = min_prob
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct = max_stake_pct

    def calculate(self, probability: float, odds: float) -> Dict:
        """
        Core value calculation.
        Returns: {"value": 0.08, "is_value_bet": True, "edge_pct": "+8.0%", ...}
        """
        if probability <= 0 or odds <= 1:
            return self._empty_result()

        value = (probability * odds) - 1.0
        implied = 1.0 / odds
        edge = probability - implied

        # Kelly Criterion: f* = (bp - q) / b
        b = odds - 1.0
        kelly = (b * probability - (1 - probability)) / b if b > 0 else 0
        quarter_kelly = max(0, kelly * self.kelly_fraction)
        stake = min(quarter_kelly * 100, self.max_stake_pct)

        is_value = (
            value >= self.min_threshold
            and probability >= self.min_prob
            and self.min_odds <= odds <= self.max_odds
        )

        return {
            "value": round(value, 4),
            "is_value_bet": is_value,
            "edge_pct": f"+{value*100:.1f}%" if value > 0 else f"{value*100:.1f}%",
            "implied_probability": round(implied, 4),
            "probability_edge": round(edge, 4),
            "kelly_fraction": round(kelly, 4),
            "recommended_stake_pct": round(stake, 2),
        }

    def evaluate_tip(self, tip: Dict) -> Dict:
        """Add value fields to a tip that already has predicted_probability and bookmaker_odds."""
        prob = tip.get("predicted_probability", 0.5)
        odds = tip.get("bookmaker_odds", 1.90)
        val = self.calculate(prob, odds)
        tip.update(val)
        return tip

    def evaluate_batch(self, tips: List[Dict]) -> List[Dict]:
        evaluated = [self.evaluate_tip(t) for t in tips]
        return sorted(evaluated, key=lambda x: x.get("value", -1), reverse=True)

    def filter_value_only(self, tips: List[Dict]) -> List[Dict]:
        return [t for t in tips if t.get("is_value_bet", False)]

    def _empty_result(self) -> Dict:
        return {
            "value": -1.0, "is_value_bet": False, "edge_pct": "N/A",
            "implied_probability": 0, "probability_edge": 0,
            "kelly_fraction": 0, "recommended_stake_pct": 0,
        }
