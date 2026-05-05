"""
streak_engine.py - Calculates probability of completing rollover chains.
streak_probability = probability ^ steps
"""
from typing import Dict


class StreakEngine:
    """Calculates the probability of completing N consecutive winning bets."""

    def __init__(self, min_threshold: float = 0.05):
        self.min_threshold = min_threshold  # Stop rollover if below this

    def calculate(self, probability: float, steps: int) -> Dict:
        """
        Calculate probability of hitting N wins in a row.

        Args:
            probability: single-bet win probability (0-1)
            steps: number of consecutive wins needed

        Returns:
            {"streak_probability": 0.168, "should_continue": True, ...}
        """
        if probability <= 0 or steps <= 0:
            return {
                "streak_probability": 0.0,
                "should_continue": False,
                "reason": "Invalid input",
                "expected_attempts": 0,
            }

        streak_prob = probability ** steps
        should_continue = streak_prob >= self.min_threshold

        # Expected attempts to complete the streak: 1 / streak_prob
        expected_attempts = round(1.0 / streak_prob, 1) if streak_prob > 0 else float("inf")

        return {
            "streak_probability": round(streak_prob, 6),
            "streak_probability_pct": f"{streak_prob * 100:.2f}%",
            "should_continue": should_continue,
            "reason": (
                f"Streak probability {streak_prob*100:.1f}% is "
                + ("above" if should_continue else "BELOW")
                + f" threshold {self.min_threshold*100:.0f}%"
            ),
            "expected_attempts": expected_attempts,
            "steps": steps,
            "single_bet_probability": round(probability, 4),
        }

    def optimal_chain_length(self, probability: float) -> Dict:
        """Find the maximum chain length where streak_prob >= threshold."""
        best = 1
        for n in range(1, 21):
            if probability ** n >= self.min_threshold:
                best = n
            else:
                break
        return {
            "optimal_steps": best,
            "streak_probability": round(probability ** best, 4),
            "probability": round(probability, 4),
        }
