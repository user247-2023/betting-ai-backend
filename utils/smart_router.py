"""
utils/smart_router.py
═══════════════════════════════════════════════════════════════
Smart AI Routing Engine
═══════════════════════════════════════════════════════════════

Philosophy:
  1. ML models run first (free, fast, no API cost)
  2. Only call LLMs when ML confidence is uncertain (0.40-0.65)
  3. Cache AI responses for 30 minutes (same fixtures = same analysis)
  4. Downgrade gracefully if one AI fails
  5. Skip expensive AI calls for low-value markets

This dramatically reduces API costs while maintaining tip quality.

Cost impact:
  - Before smart routing: 3 AI calls per GET TIPS (Claude + Gemini + Groq)
  - After smart routing:  0-2 AI calls depending on ML confidence
  - Estimated cost reduction: 40-70% on AI API spend
"""
import os
import time
import hashlib
from typing import Dict, List, Optional, Tuple
from utils.cache import get_cache, TTL_AI_TIPS
from utils.logger import get_logger

logger = get_logger("smart_router")


# ── Confidence thresholds ──────────────────────────────────────
# When ML probability is in the UNCERTAIN zone, we call LLMs
# Outside this zone, ML alone is sufficient
ML_CERTAIN_HIGH = 0.72   # Above this: ML says YES confidently → skip expensive AIs
ML_CERTAIN_LOW  = 0.35   # Below this: ML says NO confidently  → skip expensive AIs
ML_UNCERTAIN_MIN = 0.35  # Between these two: uncertain → call LLMs for validation
ML_UNCERTAIN_MAX = 0.72

# Minimum fixtures needed before we trust ML without AI validation
MIN_ML_TRAINING_SAMPLES = 200

# Markets where AI adds significant value beyond ML
# (tactical/contextual information that statistics miss)
AI_HIGH_VALUE_MARKETS = {
    "BTTS Yes",
    "BTTS No",
    "Both Halves Over 0.5",
    "1st Half Over 1.5",
    "Clean Sheet Yes",
}

# Markets where ML alone is usually sufficient
AI_LOW_VALUE_MARKETS = {
    "Over 1.5 Goals",    # Very high base rate, statistics reliable
    "Over 9.5 Corners",  # Corners well-predicted by team styles
    "Over 10.5 Corners",
}


class SmartRouter:
    """
    Intelligently routes fixture analysis between ML models and LLMs
    based on confidence levels, caching, and market type.
    """

    def __init__(self):
        self.cache = get_cache()

    # ── Cache key generation ───────────────────────────────────

    def _fixtures_hash(self, fixture_text: str) -> str:
        """Generate a stable hash for a set of fixtures."""
        return hashlib.md5(fixture_text.encode()).hexdigest()[:16]

    def get_cached_ai_response(self, fixture_text: str,
                               ai_name: str) -> Optional[str]:
        """Check if we already have an AI response for these fixtures."""
        key = f"ai:{ai_name}:{self._fixtures_hash(fixture_text)}"
        cached = self.cache.get(key)
        if cached:
            logger.info(f"[SmartRouter] Cache HIT for {ai_name} — skipping API call")
        return cached

    def cache_ai_response(self, fixture_text: str, ai_name: str, response: str):
        """Store AI response in cache for reuse."""
        key = f"ai:{ai_name}:{self._fixtures_hash(fixture_text)}"
        self.cache.set(key, response, ttl=TTL_AI_TIPS)

    # ── ML confidence assessment ───────────────────────────────

    def assess_ml_confidence(self, ml_probability: Optional[float],
                              market: str) -> Tuple[str, bool]:
        """
        Assess whether ML alone is sufficient or LLMs are needed.

        Returns:
            (confidence_zone, needs_ai)
            confidence_zone: "certain_yes" | "uncertain" | "certain_no"
            needs_ai: True if LLM validation recommended
        """
        if ml_probability is None:
            # No ML prediction → always need AI
            return ("no_ml", True)

        # Market-based override
        if market in AI_HIGH_VALUE_MARKETS:
            return ("uncertain", True)  # Always use AI for these markets

        if ml_probability >= ML_CERTAIN_HIGH:
            zone    = "certain_yes"
            need_ai = market not in AI_LOW_VALUE_MARKETS  # Only skip for simple markets
        elif ml_probability <= ML_CERTAIN_LOW:
            zone    = "certain_no"
            need_ai = False  # ML is very confident it won't happen
        else:
            zone    = "uncertain"
            need_ai = True   # ML is uncertain → use AI

        logger.info(
            f"[SmartRouter] ML prob={ml_probability:.2f}, market='{market}', "
            f"zone={zone}, needs_ai={need_ai}"
        )
        return (zone, need_ai)

    # ── AI selection strategy ──────────────────────────────────

    def select_ais_to_call(self,
                            ml_ready: bool,
                            ml_confidence_zones: List[str],
                            available_ais: List[str]) -> List[str]:
        """
        Decide which AIs to call based on ML state and confidence.

        Args:
            ml_ready: whether ML models are trained and ready
            ml_confidence_zones: zones from assess_ml_confidence per fixture
            available_ais: list of working AIs ["claude", "gemini", "groq"]

        Returns:
            List of AI names to call (may be empty if ML alone sufficient)
        """
        if not ml_ready:
            # No ML → use all available AIs
            logger.info("[SmartRouter] ML not ready → using all available AIs")
            return available_ais

        # Count how many fixtures are in uncertain zone
        uncertain_count = sum(1 for z in ml_confidence_zones if z == "uncertain")
        certain_count   = len(ml_confidence_zones) - uncertain_count
        uncertain_ratio = uncertain_count / len(ml_confidence_zones) if ml_confidence_zones else 1.0

        if uncertain_ratio < 0.20:
            # Less than 20% uncertain → ML handles it all, no AI needed
            logger.info(f"[SmartRouter] Only {uncertain_ratio:.0%} uncertain → skipping all AIs")
            return []

        elif uncertain_ratio < 0.50:
            # 20-50% uncertain → use one AI for spot-checking
            primary = available_ais[0] if available_ais else []
            logger.info(f"[SmartRouter] {uncertain_ratio:.0%} uncertain → using 1 AI ({primary})")
            return [primary] if primary else []

        else:
            # >50% uncertain → use all available AIs for full consensus
            logger.info(f"[SmartRouter] {uncertain_ratio:.0%} uncertain → using all AIs")
            return available_ais

    # ── Cost tracker ───────────────────────────────────────────

    def estimate_cost_saved(self, ais_skipped: List[str]) -> Dict:
        """
        Estimate API cost saved by skipping certain AI calls.
        Approximate costs per 1000 token request.
        """
        cost_per_call = {
            "claude":  0.0025,   # Claude Haiku: ~$0.0025 per full tips call
            "gemini":  0.0001,   # Gemini Flash: nearly free
            "groq":    0.0001,   # Groq: nearly free
        }
        saved = sum(cost_per_call.get(ai, 0) for ai in ais_skipped)
        return {
            "ais_skipped": ais_skipped,
            "estimated_saved_usd": round(saved, 5),
            "estimated_monthly_saving_usd": round(saved * 30 * 3, 3),  # 3 calls/day
        }

    # ── Main routing decision ──────────────────────────────────

    def route(self,
              fixture_text: str,
              ml_ready: bool,
              ml_predictions: Dict[str, float],  # {fixture_key: ml_prob}
              available_ais: List[str],
              markets_requested: List[str]) -> Dict:
        """
        Master routing function. Returns complete routing decision.

        Args:
            fixture_text:     formatted fixture string for AI prompt
            ml_ready:         whether ML models are loaded
            ml_predictions:   ML probabilities per fixture
            available_ais:    working AI list
            markets_requested: markets the user wants tips for

        Returns:
            {
                "ais_to_call": [...],
                "use_cache": bool,
                "cached_responses": {ai: response},
                "confidence_zones": {...},
                "cost_estimate": {...},
                "reasoning": str,
            }
        """
        # Step 1: Check cache for all AIs
        cached_responses = {}
        ais_needing_call = []

        for ai in available_ais:
            cached = self.get_cached_ai_response(fixture_text, ai)
            if cached:
                cached_responses[ai] = cached
            else:
                ais_needing_call.append(ai)

        if len(cached_responses) == len(available_ais):
            return {
                "ais_to_call":       [],
                "use_cache":         True,
                "cached_responses":  cached_responses,
                "confidence_zones":  {},
                "cost_estimate":     self.estimate_cost_saved(available_ais),
                "reasoning":         "All AIs served from cache — zero API calls needed",
            }

        # Step 2: Assess ML confidence per market
        zones = {}
        for market in markets_requested:
            # Use average ML probability across fixtures for this market
            avg_prob = None
            if ml_predictions:
                avg_prob = sum(ml_predictions.values()) / len(ml_predictions)
            zone, needs_ai = self.assess_ml_confidence(avg_prob, market)
            zones[market] = {"zone": zone, "needs_ai": needs_ai, "ml_prob": avg_prob}

        zone_list = [z["zone"] for z in zones.values()]

        # Step 3: Select which AIs to actually call
        ais_to_call = self.select_ais_to_call(ml_ready, zone_list, ais_needing_call)
        ais_skipped = [ai for ai in ais_needing_call if ai not in ais_to_call]

        reasoning_parts = []
        if not ml_ready:
            reasoning_parts.append("ML not trained → full AI analysis required")
        elif ais_skipped:
            reasoning_parts.append(
                f"ML confident on {sum(1 for z in zone_list if z != 'uncertain')}"
                f"/{len(zone_list)} fixtures → skipping {', '.join(ais_skipped)}"
            )
        else:
            reasoning_parts.append("High uncertainty → full multi-AI consensus")

        if cached_responses:
            reasoning_parts.append(f"{len(cached_responses)} AIs from cache")

        return {
            "ais_to_call":      ais_to_call,
            "use_cache":        bool(cached_responses),
            "cached_responses": cached_responses,
            "confidence_zones": zones,
            "cost_estimate":    self.estimate_cost_saved(ais_skipped),
            "reasoning":        " | ".join(reasoning_parts),
        }


# Singleton
_router: Optional[SmartRouter] = None

def get_router() -> SmartRouter:
    global _router
    if _router is None:
        _router = SmartRouter()
    return _router
