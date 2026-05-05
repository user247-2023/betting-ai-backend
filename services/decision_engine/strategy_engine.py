"""
strategy_engine.py - Decides between rollover mode and value betting mode.
Rollover only when conditions are exceptionally safe.
"""
from typing import Dict


class StrategyEngine:
    """
    Hybrid strategy selector.
    Chooses between rollover (high prob, low risk) and value betting (positive EV).
    """

    def __init__(self, rollover_min_prob: float = 0.75,
                 rollover_min_value: float = 0.03,
                 rollover_max_risk: str = "LOW"):
        self.rollover_min_prob = rollover_min_prob
        self.rollover_min_value = rollover_min_value
        self.rollover_max_risk = rollover_max_risk.upper()

    def decide(self, tip: Dict) -> Dict:
        """
        Decide which strategy to use for a given tip.

        Returns: {"strategy": "rollover" or "value_bet", "reason": "..."}
        """
        prob = tip.get("predicted_probability", 0)
        value = tip.get("value", -1)
        risk = tip.get("risk", "HIGH").upper()
        odds = tip.get("bookmaker_odds", 0)
        is_value = tip.get("is_value_bet", False)
        approved = tip.get("approved", False)  # from rollover filter

        # Rollover conditions: ALL must be true
        rollover_ok = (
            prob >= self.rollover_min_prob
            and value >= self.rollover_min_value
            and risk == self.rollover_max_risk
            and 1.20 <= odds <= 1.55
            and approved
        )

        if rollover_ok:
            return {
                "strategy": "rollover",
                "reason": (
                    f"Probability {prob:.0%} >= {self.rollover_min_prob:.0%}, "
                    f"value +{value*100:.1f}%, risk {risk}, "
                    f"odds {odds} in safe range. Ideal for rollover chain."
                ),
                "confidence": "HIGH",
                "details": {
                    "probability_check": True,
                    "value_check": True,
                    "risk_check": True,
                    "odds_check": True,
                    "filter_check": True,
                },
            }

        if is_value:
            return {
                "strategy": "value_bet",
                "reason": (
                    f"Value +{value*100:.1f}% detected but not safe enough for rollover "
                    f"(prob={prob:.0%}, risk={risk}, odds={odds}). Use standard value stake."
                ),
                "confidence": "MEDIUM" if value > 0.10 else "LOW",
                "details": {
                    "probability_check": prob >= self.rollover_min_prob,
                    "value_check": value >= self.rollover_min_value,
                    "risk_check": risk == self.rollover_max_risk,
                    "odds_check": 1.20 <= odds <= 1.55,
                    "filter_check": approved,
                },
            }

        return {
            "strategy": "skip",
            "reason": f"No value detected (value={value:.3f}). Skip this bet.",
            "confidence": "NONE",
            "details": {
                "probability_check": prob >= self.rollover_min_prob,
                "value_check": value >= self.rollover_min_value,
                "risk_check": risk == self.rollover_max_risk,
                "odds_check": 1.20 <= odds <= 1.55,
                "filter_check": approved,
            },
        }

    def batch_decide(self, tips: list) -> list:
        """Apply strategy decision to all tips."""
        for tip in tips:
            decision = self.decide(tip)
            tip["strategy"] = decision["strategy"]
            tip["strategy_reason"] = decision["reason"]
            tip["strategy_confidence"] = decision["confidence"]
        return tips
