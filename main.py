"""
ROLLOVER BETTING AI - MAIN API
3-Service Architecture on Railway:
  1. Data Service     -> fetches fixtures from API-Football
  2. ML Service       -> probability calibration + AI analysis
  3. Decision Engine  -> value detection + rollover filter + risk management
"""
import os
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

# Service imports
from services.data_service.fetcher import DataService
from services.ml_service.probability_engine import ProbabilityEngine
from services.ml_service.streak_engine import StreakEngine
from services.ml_service.ai_analyzer import AIAnalyzer
from services.decision_engine.value_engine import ValueEngine
from services.decision_engine.rollover_filter import RolloverFilter
from services.decision_engine.rollover_manager import RolloverManager
from services.decision_engine.risk_manager import RiskManager
from services.decision_engine.strategy_engine import StrategyEngine

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


@app.get("/api/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "version": "3.0.0",
        "services": {
            "data": "API-Football",
            "ml": "Claude + Gemini + Groq",
            "decision": "Value + Rollover + Risk",
        },
        "keys_loaded": {
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "api_football": bool(os.getenv("API_FOOTBALL_KEY")),
        }
    }


# ── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
