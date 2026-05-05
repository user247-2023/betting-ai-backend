"""
SERVICE 2: ML PREDICTION SERVICE
probability_engine.py - Calibrated probability prediction
Uses Platt scaling + multi-AI consensus boosting.
"""
import math
from typing import Dict, List


class ProbabilityEngine:
    """Converts raw AI confidence into calibrated betting probabilities."""

    # Platt scaling: maps raw confidence -> true probability
    PLATT_A = -2.2
    PLATT_B = 1.1

    # Market calibration offsets (tuned via backtesting)
    CALIBRATION = {
        "over 1.5 goals":  0.03,
        "over 2.5 goals":  0.00,
        "over 3.5 goals": -0.03,
        "over 4.5 goals": -0.06,
        "under 1.5 goals":-0.03,
        "under 2.5 goals": 0.02,
        "btts yes":        0.01,
        "btts no":        -0.01,
        "over 8.5 corners": 0.00,
        "over 9.5 corners":-0.02,
        "over 10.5 corners":-0.05,
        "over 3.5 cards":  0.02,
        "over 4.5 cards": -0.01,
        "first half over 0.5": 0.04,
        "first half over 1.5":-0.02,
        "clean sheet":    -0.02,
    }

    def _sigmoid(self, raw: float) -> float:
        x = raw / 100.0 if raw > 1 else raw
        exp = self.PLATT_A * x + self.PLATT_B
        try:
            return 1.0 / (1.0 + math.exp(exp))
        except OverflowError:
            return 0.0 if exp > 0 else 1.0

    def _market_offset(self, pick: str) -> float:
        pick_lower = pick.lower()
        for key, offset in self.CALIBRATION.items():
            if key in pick_lower:
                return offset
        return 0.0

    def _ai_boost(self, prob: float, ai_count: int) -> float:
        if ai_count >= 3:
            return min(0.95, prob * 1.06)
        elif ai_count >= 2:
            return min(0.93, prob * 1.03)
        return prob

    def calibrate(self, tip: Dict) -> Dict:
        """
        Calibrate a single tip's probability.

        Input:  {"confidence": 84, "pick": "Over 2.5 Goals", "aiCount": 2, ...}
        Output: adds predicted_probability, fair_odds, bookmaker_odds
        """
        raw = tip.get("confidence", 72)
        ai_count = tip.get("aiCount", tip.get("ai_count", 1))
        pick = tip.get("pick", "")

        # Platt scaling -> market calibration -> AI boost
        base = self._sigmoid(raw)
        offset = self._market_offset(pick)
        calibrated = max(0.01, min(0.99, base + offset))
        final = self._ai_boost(calibrated, ai_count)

        # Fair odds = 1 / probability
        fair_odds = round(1.0 / final, 2) if final > 0.01 else 50.0

        # Parse bookmaker odds
        bookie = self._parse_odds(tip.get("odds_range", "1.85-1.95"))

        tip["predicted_probability"] = round(final, 4)
        tip["fair_odds"] = fair_odds
        tip["bookmaker_odds"] = bookie
        return tip

    def calibrate_batch(self, tips: List[Dict]) -> List[Dict]:
        return [self.calibrate(t) for t in tips]

    def _parse_odds(self, odds_range) -> float:
        try:
            s = str(odds_range).replace("~", "").strip()
            if "-" in s:
                parts = s.split("-")
                return round((float(parts[0]) + float(parts[1])) / 2, 2)
            return float(s)
        except (ValueError, IndexError):
            return 1.90
