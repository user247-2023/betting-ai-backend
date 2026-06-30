"""
ROLLOVER BETTING AI - MAIN API v4.1.1
Upgraded with:
  - Rate limiting (slowapi) — abuse protection
  - Structured JSON logging — full request/error tracking
  - Smart AI routing — ML-first, LLMs only when uncertain
  - Retry logic — graceful AI fallback
  - Request validation — Pydantic input sanitization
  - Backtesting engine — strategy validation
"""
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime

# Rate limiting — fault-tolerant import
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING = True
except ImportError:
    print("[Main] WARNING: slowapi not installed — rate limiting disabled")
    RATE_LIMITING = False
    # Stub classes so app still starts
    class Limiter:
        def __init__(self, **kwargs): pass
        def limit(self, *args, **kwargs):
            def decorator(f): return f
            return decorator
    def get_remote_address(request): return "0.0.0.0"
    class RateLimitExceeded(Exception): pass
    def _rate_limit_exceeded_handler(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "rate limited"}, status_code=429)

# Core services
from services.data_service.fetcher import DataService
from services.ml_service.probability_engine import ProbabilityEngine
from services.ml_service.streak_engine import StreakEngine
from services.ml_service.ai_analyzer import AIAnalyzer
from services.decision_engine.value_engine import ValueEngine
from services.decision_engine.rollover_filter import RolloverFilter
from services.decision_engine.rollover_manager import RolloverManager
from services.decision_engine.risk_manager import RiskManager
from services.decision_engine.strategy_engine import StrategyEngine

# Utilities — fault-tolerant so a missing file can never crash startup
try:
    from utils.cache import get_cache, TTL_FIXTURES, TTL_AI_TIPS
except Exception as _e:
    print(f"[Main] WARNING: utils.cache unavailable ({_e}) — using in-memory stub")
    TTL_FIXTURES, TTL_AI_TIPS = 1800, 1800
    class _StubCache:
        _d = {}
        def get(self, k, *a, **kw): return self._d.get(k)
        def set(self, k, v, *a, **kw): self._d[k] = v
        def delete(self, k, *a, **kw): self._d.pop(k, None)
        def clear(self, *a, **kw): self._d.clear()
        def stats(self): return {"backend": "stub", "keys": len(self._d), "total": len(self._d)}
    _stub_cache = _StubCache()
    def get_cache(): return _stub_cache

try:
    from utils.logger import get_logger, get_request_logger
    logger         = get_logger("main")
    request_logger = get_request_logger()
except Exception as _e:
    print(f"[Main] WARNING: utils.logger unavailable ({_e}) — using print fallback")
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    logger = _logging.getLogger("main")
    class _StubReqLogger:
        def log_request(self, *a, **kw): pass
        def log_ai_call(self, *a, **kw): pass
        def log_tips_generated(self, *a, **kw): pass
        def log_error(self, *a, **kw): pass
    request_logger = _StubReqLogger()

# ML pipeline
try:
    from services.ml_predictor import get_predictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("[Main] ML predictor not available")

# Smart AI router
try:
    from utils.smart_router import get_router
    SMART_ROUTING = True
except ImportError:
    SMART_ROUTING = False
    logger.warning("[Main] Smart router not available")

# Optional odds integration
try:
    from services.data_service.odds_fetcher import OddsFetcher
    odds_fetcher = OddsFetcher()
    ODDS_ENABLED = True
    logger.info("[Main] OddsFetcher loaded successfully")
except ImportError:
    odds_fetcher = None
    ODDS_ENABLED = False

# Result settlement
try:
    from services.result_settlement import ResultSettlement
    settler = ResultSettlement()
    SETTLEMENT_ENABLED = True
except ImportError:
    settler = None
    SETTLEMENT_ENABLED = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("[Main] Starting up...")

    # Start background scheduler
    try:
        from utils.scheduler import get_scheduler
        scheduler = get_scheduler()
        scheduler.start()
        print("[Main] Scheduler started")
    except Exception as e:
        print(f"[Main] Scheduler failed to start: {e}")

    yield

    # Shutdown
    try:
        scheduler.stop()
    except Exception:
        pass
    print("[Main] Shutdown complete")


# ── Rate limiter ─────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Rollover Betting AI",
    description="ML-powered quantitative football prediction system v4.1.2",
    version="4.1.2",
    lifespan=lifespan,
    docs_url=None if os.getenv("ENVIRONMENT") == "production" else "/docs",
    redoc_url=None if os.getenv("ENVIRONMENT") == "production" else "/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# Goals-market router (Dixon-Coles model, value detection, CLV) — Tier 1-3
from services.api.goals_market_router import router as goals_router
app.include_router(goals_router)

# Live layer — upcoming fixtures + multi-book odds (The Odds API, budgeted)
from services.api.live_router import router as live_router
app.include_router(live_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with timing."""
    start  = time.time()
    try:
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        request_logger.log_request(
            method      = request.method,
            path        = request.url.path,
            client_ip   = get_remote_address(request),
            status      = response.status_code,
            duration_ms = duration,
        )
        return response
    except Exception as e:
        duration = (time.time() - start) * 1000
        request_logger.log_error(e, context=request.url.path)
        raise

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")

async def verify_api_key(request: Request):
    if request.url.path in ("/api/health", "/api/debug/ai", "/api/debug/odds"):
        return True
    if not INTERNAL_API_KEY:
        return True
    key = request.headers.get("X-API-Key", "")
    if not key:
        key = request.query_params.get("key", "")
    if key != INTERNAL_API_KEY:
        print(f"Auth failed. Received key: '{key[:10]}...' Expected: '{INTERNAL_API_KEY[:10]}...'")
        raise HTTPException(status_code=401, detail="Unauthorized - invalid API key")
    return True

# ── Initialise Services ─────────────────────────────────────────
data_svc = DataService()
prob_engine = ProbabilityEngine()
streak_engine = StreakEngine(min_threshold=0.05)
ai_analyzer = AIAnalyzer()
value_engine = ValueEngine()
rollover_filter = RolloverFilter()
rollover_mgr = RolloverManager(max_steps=5, starting_bank=50000)
risk_mgr = RiskManager(bankroll=100000)
strategy_engine = StrategyEngine()


# ── Request Models ───────────────────────────────────────────────
class TipsRequest(BaseModel):
    date: Optional[str] = None  # YYYY-MM-DD, defaults to today

class BetResult(BaseModel):
    result: str  # "WIN" or "LOSS"
    odds: float = 1.0
    stake: float = 0

class RolloverStart(BaseModel):
    bank: float = 50000

class BankrollUpdate(BaseModel):
    bankroll: float


# ══════════════════════════════════════════════════════════════════
#  ENDPOINT 1: /api/tips - THE MAIN ENDPOINT (called by Vercel app)
#  Full pipeline: Fixtures -> AI Analysis -> Probability -> Value ->
#                 Rollover Filter -> Strategy Decision
# ══════════════════════════════════════════════════════════════════
@app.post("/api/tips")
@limiter.limit("10/minute;50/hour")
async def get_tips(request: Request, req: TipsRequest, auth: bool = Depends(verify_api_key)):
    """
    Main tips endpoint — full ML + AI pipeline.
    Rate limited: 10 requests/minute, 50/hour per IP.
    Smart routing: uses ML first, only calls LLMs when uncertain.
    """
    t_start = time.time()
    today   = req.date or datetime.utcnow().strftime("%Y-%m-%d")

    # ── STEP 1: Data Service - get real fixtures ─────────────────
    fixtures = await data_svc.get_fixtures(today)
    if not fixtures:
        return {
            "tips": [], "count": 0, "date": today,
            "fixturesFound": 0,
            "message": f"No matches in your leagues for {today}.",
        }

    # Build REAL pre-match data (recent form, H2H, goals) from the local DB.
    # This grounds the AI in actual results so it cannot fabricate statistics.
    from services.data_service.form_provider import build_prompt_data, fixture_rates
    fixture_text = build_prompt_data(fixtures, today)
    rates_map    = fixture_rates(fixtures, today)   # real scoring rates for the model

    # ── STEP 2: Smart AI Routing decision ───────────────────────
    available_ais = ["claude", "gemini", "groq"]
    routing_info  = {}
    if SMART_ROUTING and ML_AVAILABLE:
        try:
            router    = get_router()
            predictor = get_predictor()
            routing   = router.route(
                fixture_text    = fixture_text,
                ml_ready        = predictor.is_ready(),
                ml_predictions  = {},
                available_ais   = available_ais,
                markets_requested = [
                    "Over 2.5 Goals", "BTTS Yes", "Over 1.5 Goals",
                    "Over 9.5 Corners", "BTTS No"
                ],
            )
            routing_info = routing
            logger.info(f"[SmartRouter] Decision: {routing['reasoning']}")
            # Use cached responses if available
            if routing.get("cached_responses"):
                logger.info("[SmartRouter] Serving from cache — no AI calls made")
        except Exception as e:
            logger.warning(f"[SmartRouter] Failed, falling back to full AI: {e}")

    # ── STEP 3: AI Analysis (3 AIs in parallel) ─────────────────
    ai_result  = await ai_analyzer.analyse(fixture_text, today)
    tips       = ai_result["tips"]
    active_ais = ai_result["activeAIs"]

    # ── STEP 3b: Model-driven multi-market tips (more volume, model-priced) ──
    # Adds strong markets the AI didn't cover on fixtures we have real data for,
    # so the feed is fuller without ever inventing a statistic.
    try:
        from services.ml_service.tip_generator import generate_model_tips
        before = len(tips)
        tips = generate_model_tips(tips, rates_map, fixtures, today)
        logger.info(f"[TipGen] AI tips {before} -> {len(tips)} after model market sweep")
    except Exception as e:
        logger.warning(f"[TipGen] model tip generation failed: {e}")

    if not tips:
        return {
            "tips": [], "count": 0, "date": today,
            "fixturesFound": len(fixtures),
            "activeAIs": active_ais,
            "message": f"AI debug: {ai_result['debug']}",
        }

    # ── STEP 4: Probability calibration ─────────────────────────
    tips = prob_engine.calibrate_batch(tips)

    # ── STEP 5: ML enrichment (blend ML 55% + AI 45%) ───────────
    ml_enriched = False
    if ML_AVAILABLE:
        try:
            predictor = get_predictor()
            if predictor.is_ready():
                tips        = predictor.enrich_tips(tips)
                ml_enriched = True
                logger.info(f"[Main] ML enriched {len(tips)} tips")
        except Exception as e:
            logger.warning(f"[Main] ML enrichment failed: {e}")

    # ── STEP 5b: Model probabilities (Dixon-Coles on real scoring rates) ──
    # Replace AI-confidence-derived probabilities with a transparent statistical
    # model for every supported goals/result market, so the downstream value/edge
    # (edge = probability * odds - 1) reflects real data, not the AI's optimism.
    try:
        from services.ml_service.model_probs import enrich as model_enrich
        tips = model_enrich(tips, rates_map)
        n_model = sum(1 for t in tips if t.get("prob_source") == "model")
        logger.info(f"[ModelProbs] Dixon-Coles probabilities on {n_model}/{len(tips)} tips")
    except Exception as e:
        logger.warning(f"[ModelProbs] Failed: {e}")

    # ── STEP 6: Real odds enrichment ────────────────────────────
    if ODDS_ENABLED and odds_fetcher and os.getenv("ODDS_API_KEY"):
        try:
            tips = odds_fetcher.enrich_tips_with_odds(tips)
            real_odds_count = sum(1 for t in tips if t.get("real_odds_available"))
            logger.info(f"[OddsFetcher] Enriched {real_odds_count}/{len(tips)} tips")
        except Exception as e:
            logger.warning(f"[OddsFetcher] Failed: {e}")

    # ── STEP 7: Value calculation ────────────────────────────────
    tips = value_engine.evaluate_batch(tips)

    # ── STEP 8: Rollover filter ──────────────────────────────────
    for tip in tips:
        filter_result          = rollover_filter.check(tip)
        tip["rollover_approved"] = filter_result["approved"]
        tip["rollover_reason"]   = filter_result["reason"]
        tip["approved"]          = filter_result["approved"]

    # ── STEP 9: Strategy decision ────────────────────────────────
    tips = strategy_engine.batch_decide(tips)

    # ── STEP 10: Streak probability ──────────────────────────────
    for tip in tips:
        if tip.get("strategy") == "rollover":
            sp = streak_engine.calculate(tip["predicted_probability"], 5)
            tip["streak_probability_5"]     = sp["streak_probability"]
            tip["streak_probability_5_pct"] = sp["streak_probability_pct"]
            tip["optimal_chain"]            = streak_engine.optimal_chain_length(
                tip["predicted_probability"]
            )["optimal_steps"]

    # ── STEP 11: Risk check ──────────────────────────────────────
    risk_status = risk_mgr.status()
    for tip in tips:
        if risk_status["risk_level"] in ("STOPPED", "SHUTDOWN"):
            tip["strategy"]        = "skip"
            tip["strategy_reason"] = f"Risk manager: {risk_status['risk_level']}"

    duration_ms = (time.time() - t_start) * 1000
    request_logger.log_tips_generated(
        n_fixtures  = len(fixtures),
        n_tips      = len(tips),
        active_ais  = active_ais,
        ml_used     = ml_enriched,
        duration_ms = duration_ms,
    )

    # Log predictions for calibration tracking (best-effort, never blocks the response)
    try:
        from services.ml_service.calibration import log_predictions
        log_predictions(tips, today)
    except Exception as e:
        logger.warning(f"[Calibration] log failed: {e}")

    return {
        "tips":            tips,
        "count":           len(tips),
        "date":            today,
        "fixturesFound":   len(fixtures),
        "activeAIs":       active_ais,
        "realOddsEnabled": bool(os.getenv("ODDS_API_KEY", "")),
        "mlEnabled":       ml_enriched,
        "risk_status":     risk_status,
        "routing":         routing_info.get("reasoning", "standard"),
        "costSaved":       routing_info.get("cost_estimate", {}),
        "durationMs":      round(duration_ms, 0),
        "generatedAt":     int(datetime.utcnow().timestamp() * 1000),
    }


# ══════════════════════════════════════════════════════════════════
#  UTILITY ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/api/fixtures/{date}")
async def get_fixtures_only(date: str):
    """Get raw fixtures for a date."""
    fixtures = await data_svc.get_fixtures(date)
    return {"fixtures": fixtures, "count": len(fixtures), "date": date}


@app.post("/api/probability")
async def calculate_probability(confidence: float, ai_count: int = 1, pick: str = ""):
    """Calculate calibrated probability from raw confidence."""
    tip = {"confidence": confidence, "aiCount": ai_count, "pick": pick}
    result = prob_engine.calibrate(tip)
    return {
        "predicted_probability": result["predicted_probability"],
        "fair_odds": result["fair_odds"],
    }


@app.post("/api/value")
async def calculate_value(probability: float, odds: float):
    """Calculate value of a bet."""
    return value_engine.calculate(probability, odds)


@app.post("/api/streak")
async def calculate_streak(probability: float, steps: int):
    """Calculate streak probability."""
    return streak_engine.calculate(probability, steps)


@app.get("/api/risk")
async def get_risk_status():
    """Get current risk management status."""
    return risk_mgr.status()


@app.post("/api/risk/update")
async def update_bankroll(req: BankrollUpdate):
    """Update bankroll after external changes."""
    risk_mgr.bankroll = req.bankroll
    if req.bankroll > risk_mgr.peak_balance:
        risk_mgr.peak_balance = req.bankroll
    return risk_mgr.status()


@app.post("/api/rollover/start")
async def start_rollover(req: RolloverStart):
    """Start a new rollover chain."""
    return rollover_mgr.start_chain(req.bank)


@app.post("/api/rollover/result")
async def rollover_result(req: BetResult):
    """Record rollover bet result."""
    if req.result.upper() == "WIN":
        return rollover_mgr.record_win(req.odds)
    else:
        return rollover_mgr.record_loss()


@app.get("/api/rollover/state")
async def rollover_state():
    """Get current rollover chain state."""
    return rollover_mgr.state()


@app.get("/api/backtest")
async def run_backtest(
    use_synthetic: bool = True,
    start_bankroll: float = 1000.0,
    n_monte_carlo: int = 1000,
    auth: bool = Depends(verify_api_key)
):
    """
    Run full backtesting suite.
    use_synthetic=true uses generated data (fast, always works).
    use_synthetic=false uses real settled predictions from DB.
    """
    try:
        from services.backtester import Backtester
        bt = Backtester()
        results = bt.run_full_backtest(
            start_bankroll=start_bankroll,
            n_monte_carlo=n_monte_carlo,
            use_synthetic=use_synthetic,
        )
        # Remove large bankroll curves from response
        for strat in results.get("strategies", {}).values():
            strat.pop("bankroll_curve", None)
        return results
    except Exception as e:
        print(f"[Backtest] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/summary")
async def backtest_summary():
    """Quick backtest summary — loads saved results if available."""
    import os, json
    path = "logs/backtest_results.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return {"cached": True, "summary": data.get("summary", {}),
                "generated_at": data.get("generated_at")}
    return {"cached": False, "message": "No backtest results yet. Call /api/backtest first."}


@app.get("/api/health")
async def health():
    """Health check."""
    cache = get_cache()
    ml_ready = False
    ml_markets = []
    if ML_AVAILABLE:
        try:
            pred = get_predictor()
            ml_ready = pred.is_ready()
            ml_markets = pred.loaded_markets()
        except Exception:
            pass

    return {
        "status": "ok",
        "version": app.version,
        "services": {
            "data":       "API-Football",
            "ml":         "Claude + Gemini + Groq + XGBoost/LightGBM",
            "decision":   "Value + Rollover + Risk",
            "cache":      cache.stats(),
            "ml_models":  {"ready": ml_ready, "markets": len(ml_markets)},
            "settlement": SETTLEMENT_ENABLED,
        },
        "features": {
            "rate_limiting":  RATE_LIMITING,
            "smart_routing":  SMART_ROUTING,
            "backtesting":    True,
            "structured_logs": True,
        },
        "keys_loaded": {
            "anthropic":   bool(os.getenv("ANTHROPIC_API_KEY")),
            "gemini":      bool(os.getenv("GEMINI_API_KEY")),
            "groq":        bool(os.getenv("GROQ_API_KEY")),
            "api_football":bool(os.getenv("API_FOOTBALL_KEY")),
        }
    }


@app.get("/api/ml/status")
async def ml_status():
    """ML model status and metrics."""
    if not ML_AVAILABLE:
        return {"available": False, "message": "ML pipeline not installed"}
    pred = get_predictor()
    return {
        "available": True,
        "ready":     pred.is_ready(),
        "markets":   pred.loaded_markets(),
        "metrics":   pred.model_metrics(),
    }


@app.get("/api/ml/train")
async def trigger_training():
    """Manually trigger model retraining (runs in background). Call from browser."""
    import asyncio
    try:
        from utils.scheduler import get_scheduler
        scheduler = get_scheduler()
        asyncio.create_task(scheduler._retrain_models())
        return {"status": "Training started in background. Check /api/ml/status in 2-3 minutes."}
    except Exception as e:
        return {"status": "Training triggered", "note": str(e)}


@app.post("/api/settle")
async def settle_predictions(date: Optional[str] = None,
                              auth: bool = Depends(verify_api_key)):
    """Manually trigger result settlement for a date."""
    if not SETTLEMENT_ENABLED:
        return {"error": "Settlement not available"}
    result = settler.settle_predictions(date)
    return result


@app.get("/api/settle/stats")
async def settlement_stats():
    """Get overall prediction performance stats."""
    if not SETTLEMENT_ENABLED:
        return {"error": "Settlement not available"}
    return settler.get_performance_stats()


@app.get("/api/cache/stats")
async def cache_stats():
    """Cache performance stats."""
    return get_cache().stats()


@app.delete("/api/cache/clear")
async def clear_cache(auth: bool = Depends(verify_api_key)):
    """Clear all cached data."""
    cache = get_cache()
    cache.clear_pattern("")
    return {"status": "Cache cleared"}


@app.get("/api/debug/odds")
async def debug_odds():
    """Test The Odds API directly."""
    key = os.getenv("ODDS_API_KEY", "")
    if not key:
        return {"error": "ODDS_API_KEY not set in Railway variables"}

    try:
        import requests as req
        # Test 1: Check available sports
        r = req.get(
            "https://api.the-odds-api.com/v4/sports",
            params={"apiKey": key},
            timeout=8,
        )
        remaining = r.headers.get("x-requests-remaining", "?")
        used = r.headers.get("x-requests-used", "?")

        if r.status_code != 200:
            return {
                "error": f"HTTP {r.status_code}",
                "body": r.text[:300],
                "key_loaded": True,
                "key_starts_with": key[:12] + "...",
            }

        sports = r.json()
        soccer_sports = [s for s in sports if "soccer" in s.get("key", "")]

        # Test 2: Get EPL odds
        r2 = req.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
            params={
                "apiKey": key,
                "regions": "uk,eu",
                "markets": "totals",
                "oddsFormat": "decimal",
            },
            timeout=8,
        )

        return {
            "key_loaded": True,
            "key_starts_with": key[:12] + "...",
            "requests_remaining": remaining,
            "requests_used": used,
            "soccer_leagues_available": len(soccer_sports),
            "epl_odds_status": r2.status_code,
            "epl_matches_found": len(r2.json()) if r2.status_code == 200 else 0,
            "sample_match": r2.json()[0].get("home_team") + " vs " + r2.json()[0].get("away_team") if r2.status_code == 200 and r2.json() else "none",
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    """Test each AI directly with a simple prompt to see exactly what fails."""
    test_prompt = "Reply with only the word: WORKING"
    results = {
        "env_check": {
            "anthropic_key_length": len(os.getenv("ANTHROPIC_API_KEY", "")),
            "gemini_key_length": len(os.getenv("GEMINI_API_KEY", "")),
            "groq_key_length": len(os.getenv("GROQ_API_KEY", "")),
            "anthropic_starts_with": os.getenv("ANTHROPIC_API_KEY", "")[:10] + "...",
            "gemini_starts_with": os.getenv("GEMINI_API_KEY", "")[:10] + "...",
            "groq_starts_with": os.getenv("GROQ_API_KEY", "")[:10] + "...",
        }
    }

    # Test Claude
    try:
        result = await ai_analyzer._call_claude(test_prompt)
        results["claude"] = {
            "status": "success" if result and not result.startswith("ERR:") else "failed",
            "response_length": len(result) if result else 0,
            "response_full": result if result else "EMPTY_STRING_RETURNED",
        }
    except Exception as e:
        results["claude"] = {"status": "exception", "error": f"{type(e).__name__}: {e}"}

    # Test Gemini
    try:
        result = await ai_analyzer._call_gemini(test_prompt)
        results["gemini"] = {
            "status": "success" if result and not result.startswith("ERR:") else "failed",
            "response_length": len(result) if result else 0,
            "response_full": result if result else "EMPTY_STRING_RETURNED",
        }
    except Exception as e:
        results["gemini"] = {"status": "exception", "error": f"{type(e).__name__}: {e}"}

    # Test Groq
    try:
        result = await ai_analyzer._call_groq(test_prompt)
        results["groq"] = {
            "status": "success" if result and not result.startswith("ERR:") else "failed",
            "response_length": len(result) if result else 0,
            "response_full": result if result else "EMPTY_STRING_RETURNED",
        }
    except Exception as e:
        results["groq"] = {"status": "exception", "error": f"{type(e).__name__}: {e}"}

    return results


# ── Run ──────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  ADMIN: keep football.db fresh, then download it to commit to GitHub.
#  Both routes are gated by your INTERNAL_API_KEY (pass ?key=YOUR_KEY).
#  refresh-db is FREE (pulls new international results from a public CSV; no
#  API-Football credits used). Workflow: hit refresh-db -> download-db ->
#  upload the file to GitHub (replacing database/football.db).
# ─────────────────────────────────────────────────────────────────────────────
_DB_FILE = os.path.join(os.path.dirname(__file__), "database", "football.db")


def _db_freshness():
    import sqlite3
    c = sqlite3.connect(_DB_FILE)
    try:
        intl = c.execute("SELECT MAX(date), COUNT(*) FROM fixtures "
                         "WHERE league_id=1 AND home_goals IS NOT NULL").fetchone()
        allr = c.execute("SELECT MAX(date), COUNT(*) FROM fixtures "
                         "WHERE home_goals IS NOT NULL").fetchone()
    finally:
        c.close()
    return {
        "latest_international": intl[0], "international_rows": intl[1],
        "latest_any_result": allr[0], "total_finished_rows": allr[1],
    }


@app.get("/api/admin/refresh-db")
async def admin_refresh_db(key: str = ""):
    """Append newly finished international matches (free). Returns DB freshness."""
    if not INTERNAL_API_KEY or key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Bad or missing ?key=")
    try:
        from services.data_service import live_data
        result = live_data.refresh_results(force=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"refresh failed: {e}")
    # settle any pending calibration predictions now that new results are in
    try:
        from services.ml_service.calibration import settle_pending
        info_cal = settle_pending()
    except Exception as e:
        info_cal = {"error": str(e)}
    info = _db_freshness()
    info["refresh"] = result
    info["calibration"] = info_cal
    info["next_step"] = ("Download the updated DB at "
                         "/api/admin/download-db?key=YOUR_KEY then upload it to "
                         "GitHub, replacing database/football.db, so the change persists.")
    return info


@app.get("/api/admin/download-db")
async def admin_download_db(key: str = ""):
    """Stream football.db so you can download it and commit it to GitHub."""
    if not INTERNAL_API_KEY or key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Bad or missing ?key=")
    if not os.path.exists(_DB_FILE):
        raise HTTPException(status_code=404, detail="football.db not found")
    return FileResponse(_DB_FILE, media_type="application/octet-stream",
                        filename="football.db")


_PARAMS_FILE = os.path.join(os.path.dirname(__file__), "database", "dc_params.json")


@app.get("/api/admin/fit-model")
async def admin_fit_model(key: str = ""):
    """Fit the Dixon-Coles strength model on the current DB and save dc_params.json.
    Run this after refresh-db. Then download-params and commit the file to GitHub."""
    if not INTERNAL_API_KEY or key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Bad or missing ?key=")
    try:
        from services.ml_service import dixon_coles_fit as dc
        info = dc.fit_and_save()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fit failed: {e}")
    info["next_step"] = ("Download the model at /api/admin/download-params?key=YOUR_KEY "
                         "and upload it to GitHub at database/dc_params.json so it persists.")
    return info


@app.get("/api/admin/download-params")
async def admin_download_params(key: str = ""):
    """Stream dc_params.json so you can commit it to GitHub (database/dc_params.json)."""
    if not INTERNAL_API_KEY or key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Bad or missing ?key=")
    if not os.path.exists(_PARAMS_FILE):
        raise HTTPException(status_code=404, detail="dc_params.json not found - run /api/admin/fit-model first")
    return FileResponse(_PARAMS_FILE, media_type="application/json", filename="dc_params.json")


@app.get("/api/calibration")
async def public_calibration():
    """Public, read-only model-accuracy report (no key, no settling)."""
    try:
        from services.ml_service import calibration as cal
        return cal.report()
    except Exception as e:
        return {"settled": 0, "error": str(e)}


@app.get("/api/admin/calibration")
async def admin_calibration(key: str = ""):
    """Settle any pending predictions, then report how well calibrated the model is."""
    if not INTERNAL_API_KEY or key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Bad or missing ?key=")
    try:
        from services.ml_service import calibration as cal
        settle = cal.settle_pending()
        rep = cal.report()
        rep["just_settled"] = settle
        return rep
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"calibration failed: {e}")


@app.get("/api/ai-check")
async def ai_check(key: str = ""):
    """Ping every configured AI in parallel and report which ones actually work.
    Open in a browser: /api/ai-check?key=YOUR_INTERNAL_API_KEY"""
    if not INTERNAL_API_KEY or key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Bad or missing ?key=")

    from concurrent.futures import ThreadPoolExecutor
    from services.ml_service.ai_analyzer import PROVIDERS

    test_prompt = ("Connection test. Reply with a JSON array containing exactly one "
                   "object: [{\"ok\":true}]")

    def classify(name: str, raw: str) -> dict:
        r = raw or ""
        if r.startswith("ERR:no_key"):
            return {"ai": name, "ok": False, "emoji": "⚪",
                    "status": "no key set", "detail": "Add its API key in Railway Variables to enable it."}
        if r.startswith("ERR:HTTP401") or r.startswith("ERR:HTTP403"):
            return {"ai": name, "ok": False, "emoji": "❌",
                    "status": "key rejected", "detail": "The key is wrong/invalid for this provider. " + r[:140]}
        if r.startswith("ERR:HTTP429"):
            return {"ai": name, "ok": True, "emoji": "🟡",
                    "status": "rate-limited", "detail": "Key works but is being throttled right now."}
        low = r.lower()
        if r.startswith("ERR:") and any(p in low for p in
                ("api key not valid", "invalid", "unauthor", "permission", "api_key")):
            return {"ai": name, "ok": False, "emoji": "❌",
                    "status": "key rejected",
                    "detail": "Wrong/invalid key for this provider (e.g. an OpenAI key won't work for Gemini). " + r[:120]}
        if r.startswith("ERR:"):
            return {"ai": name, "ok": False, "emoji": "❌",
                    "status": "error", "detail": r[:160]}
        return {"ai": name, "ok": True, "emoji": "✅",
                "status": "working", "detail": f"Responded ({len(r)} chars)."}

    with ThreadPoolExecutor(max_workers=max(2, len(PROVIDERS))) as ex:
        futures = {name: ex.submit(fn, test_prompt) for name, fn in PROVIDERS}
        results = [classify(name, futures[name].result()) for name, _ in PROVIDERS]

    working = [r["ai"] for r in results if r["status"] in ("working", "rate-limited")]
    summary = "   ".join(f"{r['emoji']} {r['ai']}" for r in results)
    return {
        "summary": summary,
        "working": working,
        "working_count": len(working),
        "total_configured": sum(1 for r in results if r["status"] != "no key set"),
        "providers": results,
        "legend": {
            "✅": "working",
            "🟡": "key works but rate-limited",
            "❌": "key is set but failing (usually wrong key)",
            "⚪": "no key set (provider skipped)",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
