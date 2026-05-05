"""
rollover_filter.py - Strict safety checks before approving a bet for rollover.
Must pass ALL conditions to be approved.
"""
from typing import Dict


class RolloverFilter:
    """Applies strict filters before approving a bet for rollover chains."""

    def __init__(self, min_prob: float = 0.70, min_odds: float = 1.25,
                 max_odds: float = 1.50):
        self.min_prob = min_prob
        self.min_odds = min_odds
        self.max_odds = max_odds

    def check(self, tip: Dict) -> Dict:
        """
        Check if a tip passes all rollover safety filters.

        Returns: {"approved": True/False, "reason": "...", "flags": [...]}
        """
        prob = tip.get("predicted_probability", 0)
        odds = tip.get("bookmaker_odds", 0)
        value = tip.get("value", -1)
        risk = tip.get("risk", "HIGH").upper()
        reasoning = tip.get("reasoning", "").lower()

        flags = []

        # Check 1: Probability must be >= threshold
        if prob < self.min_prob:
            flags.append(f"Probability {prob:.2f} below minimum {self.min_prob}")

        # Check 2: Odds must be in safe range
        if odds < self.min_odds:
            flags.append(f"Odds {odds} below minimum {self.min_odds}")
        if odds > self.max_odds:
            flags.append(f"Odds {odds} above maximum {self.max_odds}")

        # Check 3: Value must be positive
        if value < 0:
            flags.append(f"Negative value: {value:.3f}")

        # Check 4: Risk must be LOW
        if risk != "LOW":
            flags.append(f"Risk level is {risk}, must be LOW")

        # Check 5: No dangerous keywords in reasoning
        danger_words = ["suspended", "injured", "missing", "rotation",
                        "rested", "benched", "unstable form", "poor form",
                        "relegation battle", "nothing to play"]
        for word in danger_words:
            if word in reasoning:
                flags.append(f"Risk factor detected: '{word}'")

        approved = len(flags) == 0

        return {
            "approved": approved,
            "reason": "All rollover safety checks passed" if approved else "; ".join(flags),
            "flags": flags,
            "filter_results": {
                "probability_ok": prob >= self.min_prob,
                "odds_ok": self.min_odds <= odds <= self.max_odds,
                "value_ok": value >= 0,
                "risk_ok": risk == "LOW",
                "no_danger_flags": len([f for f in flags if "Risk factor" in f]) == 0,
            },
        }
