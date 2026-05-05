"""
rollover_manager.py - Manages controlled rollover chains.
Max 3-5 steps, locks profit on completion, restarts cycle.
"""
from typing import Dict, List, Optional


class RolloverManager:
    """Tracks and controls rollover betting chains."""

    def __init__(self, max_steps: int = 5, starting_bank: float = 50000):
        self.max_steps = max_steps
        self.starting_bank = starting_bank
        self.current_step = 0
        self.current_bank = starting_bank
        self.locked_profit = 0.0
        self.chain_history: List[Dict] = []
        self.completed_chains = 0
        self.failed_chains = 0

    def state(self) -> Dict:
        """Get current rollover state."""
        return {
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "current_bank": round(self.current_bank, 2),
            "starting_bank": self.starting_bank,
            "locked_profit": round(self.locked_profit, 2),
            "chain_active": self.current_step > 0,
            "steps_remaining": self.max_steps - self.current_step,
            "chain_gain_pct": round(
                ((self.current_bank - self.starting_bank) / self.starting_bank) * 100, 2
            ) if self.starting_bank > 0 else 0,
            "completed_chains": self.completed_chains,
            "failed_chains": self.failed_chains,
        }

    def start_chain(self, bank: float) -> Dict:
        """Start a new rollover chain."""
        self.current_step = 0
        self.current_bank = bank
        self.starting_bank = bank
        self.chain_history = []
        return {"status": "chain_started", "bank": bank, **self.state()}

    def record_win(self, odds: float) -> Dict:
        """Record a winning bet in the current chain."""
        prev_bank = self.current_bank
        self.current_bank *= odds
        self.current_step += 1

        self.chain_history.append({
            "step": self.current_step,
            "result": "WIN",
            "odds": odds,
            "bank_before": round(prev_bank, 2),
            "bank_after": round(self.current_bank, 2),
        })

        # Check if chain is complete
        if self.current_step >= self.max_steps:
            return self._complete_chain()

        return {
            "status": "chain_continues",
            "step": self.current_step,
            "bank": round(self.current_bank, 2),
            "profit_so_far": round(self.current_bank - self.starting_bank, 2),
            **self.state(),
        }

    def record_loss(self) -> Dict:
        """Record a losing bet - chain breaks, restart."""
        loss_amount = self.current_bank
        self.failed_chains += 1

        result = {
            "status": "chain_broken",
            "lost_at_step": self.current_step + 1,
            "amount_lost": round(loss_amount, 2),
            "chain_history": self.chain_history,
        }

        # Reset
        self.current_step = 0
        self.current_bank = 0
        self.chain_history = []

        return {**result, **self.state()}

    def _complete_chain(self) -> Dict:
        """Chain completed successfully - lock profit."""
        profit = self.current_bank - self.starting_bank
        self.locked_profit += profit
        self.completed_chains += 1

        result = {
            "status": "chain_completed",
            "steps": self.max_steps,
            "starting_bank": round(self.starting_bank, 2),
            "ending_bank": round(self.current_bank, 2),
            "profit_locked": round(profit, 2),
            "total_locked_profit": round(self.locked_profit, 2),
            "chain_history": self.chain_history,
        }

        # Reset for next chain
        self.current_step = 0
        self.chain_history = []

        return {**result, **self.state()}

    def should_continue(self, next_tip: Dict) -> Dict:
        """Decide whether to continue the current chain with the next tip."""
        prob = next_tip.get("predicted_probability", 0)
        approved = next_tip.get("approved", False)
        value = next_tip.get("value", -1)

        if not approved:
            return {"continue": False, "reason": "Tip not approved by rollover filter"}
        if prob < 0.70:
            return {"continue": False, "reason": f"Probability {prob:.2f} too low for rollover"}
        if value < 0:
            return {"continue": False, "reason": "No positive value detected"}

        return {
            "continue": True,
            "reason": f"Step {self.current_step + 1}/{self.max_steps} approved",
            "current_bank": round(self.current_bank, 2),
        }
