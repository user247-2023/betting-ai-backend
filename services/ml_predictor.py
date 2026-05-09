"""
services/ml_predictor.py
Loads trained ML models and produces calibrated probability predictions.
Combines ML probability with multi-AI consensus for final probability.
"""
import os
import pickle
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


MODELS_DIR = os.getenv("MODELS_DIR", "models")

MARKET_MAP = {
    "Over 1.5 Goals":  "target_over15",
    "Over 2.5 Goals":  "target_over25",
    "Over 3.5 Goals":  "target_over35",
    "Under 2.5 Goals": "target_under25",
    "BTTS Yes":        "target_btts",
    "BTTS No":         "target_btts",
    "Over 9.5 Corners":"target_over95c",
    "Over 3.5 Cards":  "target_over35k",
    "Over 4.5 Cards":  "target_over35k",
}

# Weights for combining ML + AI probabilities
ML_WEIGHT = 0.55   # ML model is more data-driven
AI_WEIGHT  = 0.45  # AI adds contextual reasoning


class MLPredictor:
    """Loads trained models and predicts market probabilities."""

    def __init__(self):
        self._models: Dict[str, Dict] = {}
        self._load_models()

    def _load_models(self):
        """Load all trained models from disk."""
        if not os.path.exists(MODELS_DIR):
            print(f"[MLPredictor] No models directory found at {MODELS_DIR}")
            return

        for path in Path(MODELS_DIR).glob("*.pkl"):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                market = data.get("market", "")
                self._models[market] = data
                print(f"[MLPredictor] Loaded: {path.name}")
            except Exception as e:
                print(f"[MLPredictor] Failed to load {path}: {e}")

        print(f"[MLPredictor] Loaded {len(self._models)} models")

    def _extract_features(self, match_context: Dict,
                           feature_cols: List[str]) -> Optional[np.ndarray]:
        """Extract features for a single match from context dict."""
        try:
            vec = [match_context.get(col, 0) or 0 for col in feature_cols]
            return np.array([vec])
        except Exception as e:
            print(f"[MLPredictor] Feature extraction error: {e}")
            return None

    def predict_market(self, market_pick: str,
                       match_context: Dict,
                       ai_probability: float) -> Dict:
        """
        Predict probability for a specific market.

        Args:
            market_pick: e.g. "Over 2.5 Goals"
            match_context: dict of engineered features for the match
            ai_probability: calibrated probability from AI analysis (0-1)

        Returns:
            {
                "market": "Over 2.5 Goals",
                "ml_probability": 0.74,
                "ai_probability": 0.71,
                "final_probability": 0.73,
                "ml_available": True,
                "confidence_source": "ML+AI"
            }
        """
        target = MARKET_MAP.get(market_pick)

        # Invert BTTS No
        invert = "No" in market_pick and "BTTS" in market_pick

        ml_prob = None
        if target and target in self._models:
            model_data = self._models[target]
            model      = model_data["model"]
            feat_cols  = model_data["feature_cols"]
            X          = self._extract_features(match_context, feat_cols)

            if X is not None:
                try:
                    prob = model.predict_proba(X)[0][1]
                    ml_prob = 1 - prob if invert else prob
                except Exception as e:
                    print(f"[MLPredictor] Prediction error for {market_pick}: {e}")

        # Combine ML + AI
        if ml_prob is not None:
            final_prob = ML_WEIGHT * ml_prob + AI_WEIGHT * ai_probability
            source = "ML+AI"
        else:
            final_prob = ai_probability
            source = "AI only"

        final_prob = round(max(0.01, min(0.99, final_prob)), 4)

        return {
            "market":            market_pick,
            "ml_probability":    round(ml_prob, 4) if ml_prob is not None else None,
            "ai_probability":    round(ai_probability, 4),
            "final_probability": final_prob,
            "ml_available":      ml_prob is not None,
            "confidence_source": source,
        }

    def enrich_tips(self, tips: List[Dict],
                    match_context: Dict = None) -> List[Dict]:
        """
        Enrich a list of AI tips with ML probabilities.
        match_context: pre-computed feature dict for the match.
        """
        ctx = match_context or {}

        for tip in tips:
            pick = tip.get("pick", "")
            ai_prob = tip.get("predicted_probability",
                              tip.get("confidence", 72) / 100)

            ml_result = self.predict_market(pick, ctx, ai_prob)

            tip["ml_probability"]    = ml_result["ml_probability"]
            tip["ai_probability"]    = ml_result["ai_probability"]
            tip["final_probability"] = ml_result["final_probability"]
            tip["confidence_source"] = ml_result["confidence_source"]
            tip["predicted_probability"] = ml_result["final_probability"]

        return tips

    def is_ready(self) -> bool:
        """Return True if at least one model is loaded."""
        return len(self._models) > 0

    def loaded_markets(self) -> List[str]:
        """Return list of markets with trained models."""
        return list(self._models.keys())

    def model_metrics(self) -> Dict:
        """Return metrics for all loaded models."""
        return {
            mkt: data.get("metrics", {})
            for mkt, data in self._models.items()
        }


# Singleton instance
_predictor: Optional[MLPredictor] = None

def get_predictor() -> MLPredictor:
    """Get or create the singleton predictor."""
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor


if __name__ == "__main__":
    predictor = get_predictor()
    print("Models ready:", predictor.is_ready())
    print("Loaded markets:", predictor.loaded_markets())

    if predictor.is_ready():
        result = predictor.predict_market(
            "Over 2.5 Goals",
            match_context={"home_xg_avg5": 0.7, "away_xg_avg5": 0.6,
                          "home_form5": 0.65, "away_form5": 0.45},
            ai_probability=0.71,
        )
        print(result)
