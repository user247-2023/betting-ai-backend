"""
ROLLOVER BETTING AI - MAIN API v4.0
Upgraded with:
  - ML prediction pipeline (XGBoost + LightGBM + RandomForest)
  - APScheduler background jobs
  - In-memory caching (Redis-ready)
  - Auto result settlement
  - PostgreSQL-compatible tracking
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

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

# ML pipeline
try:
    from services.ml_predictor import get_predictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[Main] ML predictor not available")

# Cache
from utils.cache import get_cache, TTL_FIXTURES, TTL_AI_TIPS

# Optional odds integration
try:
    from services.data_service.odds_fetcher import OddsFetcher
    odds_fetcher = OddsFetcher()
    ODDS_ENABLED = True
    print("[Main] OddsFetcher loaded successfully")
except ImportError:
    odds_fetcher = None
    ODDS_ENABLED = False
    print("[Main] OddsFetcher not found - running without real odds")

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


app = FastAPI(
    title="Rollover Betting AI",
    description="ML-powered 3-service betting intelligence system",
    version="4.0.0",
    lifespan=lifespan,
    docs_url=None if os.getenv("ENVIRONMENT") == "production" else "/docs",
    redoc_url=None if os.getenv("ENVIRONMENT") == "production" else "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")

async def verify_api_key(request: Request):
    if request.url.path in ("/api/health", "/api/debug/ai", "/api/debug/odds"):
        return True
    if not INTERNAL_API_KEY:
        return True
    key = request.headers.get("X-API-Key", "")
    if key != INTERNAL_API_KEY:
        print(f"Auth failed. Received key: '{key[:10]}...' Expected: '{INTERNAL_API_KEY[:10]}...'")
        raise HTTPException(status_code=401, detail="Unauthorized - invalid API key")
    return True

# ── App Setup ────────────────────────────────────────────────────
app = FastAPI(
    title="Rollover Betting AI",
    description="3-service betting intelligence system",
    version="3.0.0",
    # Disable public docs in production for security
    docs_url=None if os.getenv("ENVIRONMENT") == "production" else "/docs",
    redoc_url=None if os.getenv("ENVIRONMENT") == "production" else "/redoc",
)

# ── CORS — allow all Vercel deployments + local dev ─────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins — Railway auth key provides security
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# ── API Key Authentication ───────────────────────────────────────
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")

async def verify_api_key(request: Request):
    """Check that requests come with the correct internal API key."""
    # Always allow health check
    if request.url.path == "/api/health":
        return True
    # If no INTERNAL_API_KEY configured, allow all (open mode)
    if not INTERNAL_API_KEY:
        return True
    # Check header
    key = request.headers.get("X-API-Key", "")
    # Also accept from query param as fallback
    if not key:
        key = request.query_params.get("key", "")
    if key != INTERNAL_API_KEY:
        # Log for debugging
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
async def get_tips(req: TipsRequest, auth: bool = Depends(verify_api_key)):
    today = req.date or datetime.utcnow().strftime("%Y-%m-%d")

    # ── STEP 1: Data Service - get real fixtures ─────────────────
    fixtures = await data_svc.get_fixtures(today)
    if not fixtures:
        return {
            "tips": [], "count": 0, "date": today,
            "fixturesFound": 0,
            "message": f"No matches in your leagues for {today}.",
        }

    # ── STEP 2: ML Service - AI analysis (3 AIs in parallel) ────
    fixture_text = data_svc.format_for_prompt(fixtures)
    ai_result = await ai_analyzer.analyse(fixture_text, today)
    tips = ai_result["tips"]
    active_ais = ai_result["activeAIs"]

    if not tips:
        return {
            "tips": [], "count": 0, "date": today,
            "fixturesFound": len(fixtures),
            "activeAIs": active_ais,
            "message": f"AI debug: {ai_result['debug']}",
        }

    # ── STEP 3: ML Service - calibrate probabilities ─────────────
    tips = prob_engine.calibrate_batch(tips)

    # ── STEP 3.5: ML Predictor - blend ML + AI probabilities ─────
    if ML_AVAILABLE:
        try:
            predictor = get_predictor()
            if predictor.is_ready():
                tips = predictor.enrich_tips(tips)
                print(f"[Main] ML enriched {len(tips)} tips")
        except Exception as e:
            print(f"[Main] ML enrichment error: {e}")

    # ── STEP 3.6: Enrich with real odds if available ──────────────
    if ODDS_ENABLED and odds_fetcher and os.getenv("ODDS_API_KEY"):
        tips = odds_fetcher.enrich_tips_with_odds(tips)
        real_odds_count = sum(1 for t in tips if t.get("real_odds_available"))
        print(f"[OddsFetcher] Enriched {real_odds_count}/{len(tips)} tips with real odds")

    # ── STEP 4: Decision Engine - value calculation ──────────────
    tips = value_engine.evaluate_batch(tips)

    # ── STEP 5: Decision Engine - rollover filter ────────────────
    for tip in tips:
        filter_result = rollover_filter.check(tip)
        tip["rollover_approved"] = filter_result["approved"]
        tip["rollover_reason"] = filter_result["reason"]
        tip["approved"] = filter_result["approved"]

    # ── STEP 6: Decision Engine - strategy selection ─────────────
    tips = strategy_engine.batch_decide(tips)

    # ── STEP 7: ML Service - streak probability ─────────────────
    for tip in tips:
        if tip.get("strategy") == "rollover":
            sp = streak_engine.calculate(tip["predicted_probability"], 5)
            tip["streak_probability_5"] = sp["streak_probability"]
            tip["streak_probability_5_pct"] = sp["streak_probability_pct"]
            tip["optimal_chain"] = streak_engine.optimal_chain_length(
                tip["predicted_probability"]
            )["optimal_steps"]

    # ── STEP 8: Risk check ───────────────────────────────────────
    risk_status = risk_mgr.status()
    for tip in tips:
        if risk_status["risk_level"] in ("STOPPED", "SHUTDOWN"):
            tip["strategy"] = "skip"
            tip["strategy_reason"] = f"Risk manager: {risk_status['risk_level']}"

    return {
        "tips": tips,
        "count": len(tips),
        "date": today,
        "fixturesFound": len(fixtures),
        "activeAIs": active_ais,
        "realOddsEnabled": bool(os.getenv("ODDS_API_KEY", "")),
        "risk_status": risk_status,
        "generatedAt": int(datetime.utcnow().timestamp() * 1000),
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
        "version": "4.0.0",
        "services": {
            "data":       "API-Football",
            "ml":         "Claude + Gemini + Groq + XGBoost/LightGBM",
            "decision":   "Value + Rollover + Risk",
            "cache":      cache.stats(),
            "ml_models":  {"ready": ml_ready, "markets": len(ml_markets)},
            "settlement": SETTLEMENT_ENABLED,
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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
