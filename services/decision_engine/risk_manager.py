"""
risk_manager.py - Professional capital protection system.
Tracks bankroll, drawdown, exposure limits.
"""
from typing import Dict, List
from datetime import datetime, date


class RiskManager:
    """Manages bankroll risk with strict exposure and drawdown limits."""

    def __init__(self, bankroll: float = 100000,
                 max_risk_pct: float = 2.0,
                 daily_cap_pct: float = 10.0,
                 drawdown_reduce_pct: float = 10.0,
                 drawdown_stop_pct: float = 20.0,
                 drawdown_shutdown_pct: float = 50.0):
        self.bankroll = bankroll
        self.peak_balance = bankroll
        self.max_risk_pct = max_risk_pct
        self.daily_cap_pct = daily_cap_pct
        self.drawdown_reduce_pct = drawdown_reduce_pct
        self.drawdown_stop_pct = drawdown_stop_pct
        self.drawdown_shutdown_pct = drawdown_shutdown_pct
        self.daily_exposure = 0.0
        self.today = date.today().isoformat()
        self.bets_today: List[Dict] = []

    def _reset_daily(self):
        """Reset daily counters if new day."""
        today_str = date.today().isoformat()
        if today_str != self.today:
            self.daily_exposure = 0.0
            self.bets_today = []
            self.today = today_str

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak as percentage."""
        if self.peak_balance <= 0:
            return 0
        return round(((self.peak_balance - self.bankroll) / self.peak_balance) * 100, 2)

    @property
    def risk_level(self) -> str:
        dd = self.drawdown
        if dd >= self.drawdown_shutdown_pct:
            return "SHUTDOWN"
        elif dd >= self.drawdown_stop_pct:
            return "STOPPED"
        elif dd >= self.drawdown_reduce_pct:
            return "REDUCED"
        return "NORMAL"

    def calculate_stake(self, recommended_pct: float) -> Dict:
        """Calculate actual stake amount considering risk limits."""
        self._reset_daily()

        dd = self.drawdown
        risk = self.risk_level

        # Shutdown mode - no betting
        if risk == "SHUTDOWN":
            return {
                "stake": 0, "stake_pct": 0,
                "approved": False,
                "reason": f"SHUTDOWN: Drawdown {dd:.1f}% exceeds {self.drawdown_shutdown_pct}%",
                "risk_level": risk,
            }

        # Stopped mode - no betting for 48h
        if risk == "STOPPED":
            return {
                "stake": 0, "stake_pct": 0,
                "approved": False,
                "reason": f"STOPPED: Drawdown {dd:.1f}% exceeds {self.drawdown_stop_pct}%. Wait 48h.",
                "risk_level": risk,
            }

        # Reduced mode - halve stakes
        effective_pct = recommended_pct
        if risk == "REDUCED":
            effective_pct = recommended_pct * 0.5

        # Cap at max risk
        effective_pct = min(effective_pct, self.max_risk_pct)

        # Check daily exposure
        daily_remaining = (self.daily_cap_pct / 100 * self.bankroll) - self.daily_exposure
        stake = min(
            self.bankroll * effective_pct / 100,
            daily_remaining,
        )
        stake = max(0, round(stake, 2))

        if stake <= 0:
            return {
                "stake": 0, "stake_pct": 0,
                "approved": False,
                "reason": f"Daily exposure cap reached ({self.daily_cap_pct}%)",
                "risk_level": risk,
            }

        return {
            "stake": stake,
            "stake_pct": round(stake / self.bankroll * 100, 2),
            "approved": True,
            "reason": f"Stake approved ({risk} mode)",
            "risk_level": risk,
            "drawdown": dd,
            "daily_remaining": round(daily_remaining, 2),
        }

    def record_bet(self, stake: float, result: str, profit: float):
        """Record a bet result and update bankroll."""
        self._reset_daily()
        self.daily_exposure += stake
        self.bankroll += profit
        if self.bankroll > self.peak_balance:
            self.peak_balance = self.bankroll
        self.bets_today.append({
            "stake": stake, "result": result,
            "profit": profit, "bankroll": round(self.bankroll, 2),
        })

    def status(self) -> Dict:
        return {
            "bankroll": round(self.bankroll, 2),
            "peak_balance": round(self.peak_balance, 2),
            "drawdown_pct": self.drawdown,
            "risk_level": self.risk_level,
            "daily_exposure": round(self.daily_exposure, 2),
            "bets_today": len(self.bets_today),
            "daily_cap_remaining": round(
                (self.daily_cap_pct / 100 * self.bankroll) - self.daily_exposure, 2
            ),
        }
