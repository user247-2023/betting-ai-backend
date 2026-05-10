"""
services/backtester.py
═══════════════════════════════════════════════════════════════════
Professional Backtesting Engine for Rollover Betting AI
═══════════════════════════════════════════════════════════════════

Simulates thousands of bets using historical data to validate:
  - Rollover strategy profitability
  - Flat betting baseline
  - Kelly staking performance
  - Probability calibration accuracy
  - Drawdown exposure
  - Risk-of-ruin estimation
  - Monte Carlo confidence intervals

Usage:
    from services.backtester import Backtester
    bt = Backtester()
    results = bt.run_full_backtest()
"""

import os
import json
import math
import random
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict


DB_PATH    = os.getenv("DB_PATH", "database/football.db")
MODELS_DIR = os.getenv("MODELS_DIR", "models")


# ══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════

@dataclass
class BetRecord:
    """A single bet result used in backtesting."""
    fixture_id:       int
    date:             str
    match:            str
    market:           str
    pick:             str
    predicted_prob:   float   # Our model's probability
    bookmaker_odds:   float   # Odds at time of bet
    implied_prob:     float   # 1 / bookmaker_odds
    value:            float   # predicted_prob * odds - 1
    actual_outcome:   int     # 1 = won, 0 = lost
    stake:            float   # stake placed
    profit_loss:      float   # net P&L from this bet
    strategy:         str     # "rollover", "flat", "kelly"


@dataclass
class BacktestResult:
    """Complete backtest run results."""
    strategy:         str
    n_bets:           int
    n_wins:           int
    n_losses:         int
    win_rate:         float
    roi:              float
    total_profit:     float
    start_bankroll:   float
    end_bankroll:     float
    peak_bankroll:    float
    max_drawdown:     float
    max_drawdown_pct: float
    sharpe_ratio:     float
    profit_factor:    float
    longest_win_streak:  int
    longest_loss_streak: int
    avg_odds:         float
    avg_predicted_prob: float
    avg_implied_prob:   float
    avg_value:          float
    value_bets_roi:     float
    non_value_bets_roi: float
    calibration_error:  float   # Mean absolute calibration error
    bankroll_curve:   List[float] = field(default_factory=list)
    monthly_roi:      Dict[str, float] = field(default_factory=dict)
    market_breakdown: Dict[str, dict] = field(default_factory=dict)


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation output."""
    n_simulations:    int
    n_bets_per_sim:   int
    median_roi:       float
    p10_roi:          float    # 10th percentile (bad case)
    p25_roi:          float
    p75_roi:          float
    p90_roi:          float    # 90th percentile (good case)
    risk_of_ruin:     float    # % of sims ending bankrupt (<10% start)
    prob_profitable:  float    # % of sims with positive ROI
    expected_roi:     float    # mean ROI
    roi_std:          float    # ROI standard deviation
    median_max_drawdown: float
    bankroll_paths:   List[List[float]] = field(default_factory=list)  # sample paths


# ══════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR
# (Used when real historical data is not yet available)
# ══════════════════════════════════════════════════════════════════

class SyntheticDataGenerator:
    """
    Generates realistic synthetic betting records for backtesting
    when real historical data hasn't been collected yet.

    Uses realistic football statistics:
    - Over 2.5 Goals hits ~52% of EPL matches
    - BTTS Yes hits ~50% of EPL matches
    - Over 1.5 Goals hits ~72% of matches
    """

    MARKET_BASE_RATES = {
        "Over 1.5 Goals":    0.73,
        "Over 2.5 Goals":    0.53,
        "Over 3.5 Goals":    0.25,
        "Under 2.5 Goals":   0.47,
        "BTTS Yes":          0.51,
        "BTTS No":           0.49,
        "Over 9.5 Corners":  0.48,
        "Over 10.5 Corners": 0.32,
        "Over 3.5 Cards":    0.45,
        "Over 4.5 Cards":    0.28,
        "1st Half Over 0.5": 0.68,
        "1st Half Over 1.5": 0.32,
    }

    TYPICAL_ODDS = {
        "Over 1.5 Goals":    (1.30, 1.50),
        "Over 2.5 Goals":    (1.75, 2.05),
        "Over 3.5 Goals":    (2.90, 3.50),
        "Under 2.5 Goals":   (1.80, 2.10),
        "BTTS Yes":          (1.75, 2.00),
        "BTTS No":           (1.80, 2.05),
        "Over 9.5 Corners":  (1.85, 2.10),
        "Over 10.5 Corners": (2.40, 2.80),
        "Over 3.5 Cards":    (1.90, 2.20),
        "Over 4.5 Cards":    (2.60, 3.20),
        "1st Half Over 0.5": (1.35, 1.55),
        "1st Half Over 1.5": (2.70, 3.20),
    }

    def generate(self, n: int = 2000, seed: int = 42) -> List[BetRecord]:
        """Generate n synthetic bet records with realistic distributions."""
        random.seed(seed)
        np.random.seed(seed)

        records = []
        markets = list(self.MARKET_BASE_RATES.keys())
        leagues  = ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1",
                    "UCL", "Europa League", "MLS", "Brazil Serie A"]

        start_date = datetime(2024, 1, 1)

        for i in range(n):
            market  = random.choice(markets)
            league  = random.choice(leagues)
            base_rate = self.MARKET_BASE_RATES[market]
            odds_range = self.TYPICAL_ODDS[market]

            # Bookmaker odds (uniform in typical range)
            bm_odds    = random.uniform(*odds_range)
            implied    = 1.0 / bm_odds

            # Our predicted probability — slightly better than implied
            # (simulating a model with a small edge)
            skill_noise = np.random.normal(0.04, 0.06)  # avg +4% edge
            pred_prob   = max(0.05, min(0.95, implied + skill_noise))

            # Value
            value = pred_prob * bm_odds - 1.0

            # Actual outcome — drawn from true base rate (not our prediction)
            # This is crucial: outcome is NOT based on our model's pred_prob
            actual = 1 if random.random() < base_rate else 0

            # Stake (will be overridden per strategy in backtest)
            stake = 1.0

            profit_loss = stake * (bm_odds - 1) if actual else -stake

            bet_date = (start_date + timedelta(days=random.randint(0, 480))).strftime("%Y-%m-%d")

            records.append(BetRecord(
                fixture_id     = 1000000 + i,
                date           = bet_date,
                match          = f"Team A{i} vs Team B{i} ({league})",
                market         = market,
                pick           = market,
                predicted_prob = round(pred_prob, 4),
                bookmaker_odds = round(bm_odds, 2),
                implied_prob   = round(implied, 4),
                value          = round(value, 4),
                actual_outcome = actual,
                stake          = stake,
                profit_loss    = round(profit_loss, 4),
                strategy       = "synthetic",
            ))

        # Sort by date
        records.sort(key=lambda r: r.date)
        print(f"[Backtester] Generated {len(records)} synthetic bet records")
        return records


# ══════════════════════════════════════════════════════════════════
# REAL DATA LOADER
# ══════════════════════════════════════════════════════════════════

class HistoricalDataLoader:
    """Loads real historical predictions from the SQLite database."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def load_settled_predictions(self) -> List[BetRecord]:
        """Load all settled predictions from the database."""
        if not os.path.exists(self.db_path):
            print(f"[DataLoader] Database not found at {self.db_path}")
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("""
                SELECT
                    p.fixture_id,
                    f.date,
                    f.home_team || ' vs ' || f.away_team AS match_name,
                    p.market,
                    p.market AS pick,
                    p.final_probability,
                    p.bookmaker_odds,
                    p.result,
                    p.profit_loss
                FROM predictions p
                LEFT JOIN fixtures f ON p.fixture_id = f.fixture_id
                WHERE p.result IS NOT NULL
                  AND p.bookmaker_odds > 1.0
                ORDER BY f.date ASC
            """)

            rows = cursor.fetchall()
            conn.close()

            records = []
            for row in rows:
                fid, date, match, market, pick, pred_prob, bm_odds, result, pl = row
                pred_prob  = pred_prob or 0.5
                bm_odds    = bm_odds or 1.90
                implied    = 1.0 / bm_odds
                value      = pred_prob * bm_odds - 1.0
                actual     = 1 if result == "WIN" else 0

                records.append(BetRecord(
                    fixture_id     = fid or 0,
                    date           = date or "2025-01-01",
                    match          = match or "Unknown",
                    market         = market or "Unknown",
                    pick           = pick or "Unknown",
                    predicted_prob = round(pred_prob, 4),
                    bookmaker_odds = round(bm_odds, 2),
                    implied_prob   = round(implied, 4),
                    value          = round(value, 4),
                    actual_outcome = actual,
                    stake          = 1.0,
                    profit_loss    = round(pl or 0, 4),
                    strategy       = "historical",
                ))

            print(f"[DataLoader] Loaded {len(records)} settled predictions from database")
            return records

        except Exception as e:
            print(f"[DataLoader] Error loading predictions: {e}")
            return []


# ══════════════════════════════════════════════════════════════════
# STAKING STRATEGIES
# ══════════════════════════════════════════════════════════════════

class StakingStrategy:
    """Implements different staking approaches for backtesting."""

    @staticmethod
    def flat_stake(bankroll: float, unit: float = 0.02) -> float:
        """Bet a fixed percentage of STARTING bankroll each time."""
        return bankroll * unit

    @staticmethod
    def kelly_stake(bankroll: float, prob: float, odds: float,
                    fraction: float = 0.25, max_pct: float = 0.05) -> float:
        """Quarter-Kelly stake, capped at max_pct of bankroll."""
        b = odds - 1.0
        q = 1.0 - prob
        kelly = (b * prob - q) / b if b > 0 else 0
        kelly = max(0, kelly)
        stk = bankroll * min(kelly * fraction, max_pct)
        return max(0, stk)

    @staticmethod
    def rollover_stake(bankroll: float, target_multiplier: float = 1.10) -> float:
        """
        For a rollover strategy, stake is typically the whole bankroll
        (or a defined portion) since we're compounding step-by-step.
        For backtesting purposes, treat each rollover bet as 100% of current bankroll.
        """
        return bankroll

    @staticmethod
    def fixed_amount(amount: float) -> float:
        """Simple fixed unit staking."""
        return amount


# ══════════════════════════════════════════════════════════════════
# CORE BACKTESTER
# ══════════════════════════════════════════════════════════════════

class Backtester:
    """
    Professional backtesting engine.
    Runs multiple staking strategies against historical or synthetic data.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path     = db_path
        self.data_loader = HistoricalDataLoader(db_path)
        self.synthetic   = SyntheticDataGenerator()
        self.staking     = StakingStrategy()

    # ── Data loading ──────────────────────────────────────────────

    def _load_data(self) -> List[BetRecord]:
        """Load real data if available, else use synthetic."""
        real = self.data_loader.load_settled_predictions()
        if len(real) >= 50:
            print(f"[Backtester] Using {len(real)} real historical records")
            return real
        else:
            n_synth = max(2000, 3000)
            print(f"[Backtester] Insufficient real data ({len(real)} records). "
                  f"Using {n_synth} synthetic records.")
            return self.synthetic.generate(n=n_synth)

    def _filter_value_bets(self, records: List[BetRecord],
                            min_value: float = 0.05) -> List[BetRecord]:
        return [r for r in records if r.value >= min_value]

    def _filter_high_confidence(self, records: List[BetRecord],
                                  min_prob: float = 0.65) -> List[BetRecord]:
        return [r for r in records if r.predicted_prob >= min_prob]

    # ── Single strategy simulation ────────────────────────────────

    def simulate_strategy(self,
                          records: List[BetRecord],
                          strategy: str = "flat",
                          start_bankroll: float = 1000.0,
                          unit_pct: float = 0.02,
                          kelly_fraction: float = 0.25) -> BacktestResult:
        """
        Simulate one staking strategy over all bet records.

        Strategies:
            "flat"      — fixed 2% of starting bankroll per bet
            "kelly"     — quarter-Kelly sizing per bet
            "rollover"  — compound: each bet uses full current bankroll
            "fixed"     — fixed 1 unit per bet (normalised)
        """
        if not records:
            return self._empty_result(strategy, start_bankroll)

        bankroll      = start_bankroll
        peak_bankroll = start_bankroll
        bankroll_curve = [start_bankroll]

        wins = losses = 0
        total_staked = total_profit = 0.0
        total_gross_win = total_gross_loss = 0.0
        cur_win_streak = cur_loss_streak = 0
        max_win_streak = max_loss_streak = 0
        max_drawdown = 0.0
        drawdown_start = start_bankroll

        monthly_pnl: Dict[str, float] = {}
        market_stats: Dict[str, dict] = {}

        for rec in records:
            if bankroll <= 0:
                break

            # Determine stake
            if strategy == "flat":
                stake = start_bankroll * unit_pct
            elif strategy == "kelly":
                stake = self.staking.kelly_stake(
                    bankroll, rec.predicted_prob, rec.bookmaker_odds,
                    fraction=kelly_fraction
                )
            elif strategy == "rollover":
                stake = bankroll  # full bankroll per step
            else:
                stake = unit_pct * start_bankroll

            stake = min(stake, bankroll)  # can't bet more than we have

            # Outcome
            if rec.actual_outcome == 1:
                pnl = stake * (rec.bookmaker_odds - 1.0)
                wins += 1
                cur_win_streak += 1
                cur_loss_streak = 0
                max_win_streak  = max(max_win_streak, cur_win_streak)
                total_gross_win += pnl
            else:
                pnl = -stake
                losses += 1
                cur_loss_streak += 1
                cur_win_streak  = 0
                max_loss_streak = max(max_loss_streak, cur_loss_streak)
                total_gross_loss += abs(pnl)

            bankroll     += pnl
            total_staked += stake
            total_profit += pnl

            # Track peak and drawdown
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
                drawdown_start = bankroll
            else:
                dd = peak_bankroll - bankroll
                if dd > max_drawdown:
                    max_drawdown = dd

            bankroll_curve.append(round(bankroll, 4))

            # Monthly breakdown
            month_key = rec.date[:7]
            monthly_pnl[month_key] = monthly_pnl.get(month_key, 0.0) + pnl

            # Market breakdown
            mkt = rec.market
            if mkt not in market_stats:
                market_stats[mkt] = {
                    "n": 0, "wins": 0, "profit": 0.0, "staked": 0.0
                }
            market_stats[mkt]["n"]      += 1
            market_stats[mkt]["wins"]   += rec.actual_outcome
            market_stats[mkt]["profit"] += pnl
            market_stats[mkt]["staked"] += stake

        # ── Derived metrics ───────────────────────────────────────
        n_bets   = wins + losses
        win_rate = wins / n_bets if n_bets else 0
        roi      = total_profit / total_staked if total_staked > 0 else 0

        max_drawdown_pct = max_drawdown / peak_bankroll if peak_bankroll else 0

        # Sharpe-like ratio (using bet-level P&L)
        bet_returns = []
        for rec in records:
            if strategy == "flat":
                stk = start_bankroll * unit_pct
            elif strategy == "kelly":
                stk = self.staking.kelly_stake(
                    start_bankroll, rec.predicted_prob, rec.bookmaker_odds, kelly_fraction
                )
            else:
                stk = start_bankroll * unit_pct

            if rec.actual_outcome:
                bet_returns.append((stk * (rec.bookmaker_odds - 1.0)) / start_bankroll)
            else:
                bet_returns.append(-stk / start_bankroll)

        if len(bet_returns) > 1:
            ret_std = float(np.std(bet_returns))
            ret_mean = float(np.mean(bet_returns))
            sharpe = ret_mean / ret_std * math.sqrt(len(bet_returns)) if ret_std > 0 else 0
        else:
            sharpe = 0.0

        profit_factor = total_gross_win / total_gross_loss if total_gross_loss > 0 else float("inf")

        avg_odds = float(np.mean([r.bookmaker_odds for r in records]))
        avg_prob = float(np.mean([r.predicted_prob for r in records]))
        avg_impl = float(np.mean([r.implied_prob   for r in records]))
        avg_val  = float(np.mean([r.value          for r in records]))

        # Value bets ROI
        value_bets    = [r for r in records if r.value >= 0.05]
        non_value_bets = [r for r in records if r.value < 0.05]
        vb_roi  = self._simple_roi(value_bets)
        nvb_roi = self._simple_roi(non_value_bets)

        # Calibration error
        cal_err = self._calibration_error(records)

        # Monthly ROI
        monthly_roi = {}
        for month, pnl in monthly_pnl.items():
            month_staked = total_staked / len(monthly_pnl) if monthly_pnl else 1
            monthly_roi[month] = round(pnl / month_staked, 4) if month_staked else 0

        # Market breakdown ROI
        market_breakdown = {}
        for mkt, stats in market_stats.items():
            wr = stats["wins"] / stats["n"] if stats["n"] else 0
            mr = stats["profit"] / stats["staked"] if stats["staked"] else 0
            market_breakdown[mkt] = {
                "n": stats["n"],
                "win_rate": round(wr, 4),
                "roi": round(mr, 4),
                "profit": round(stats["profit"], 2),
            }

        return BacktestResult(
            strategy          = strategy,
            n_bets            = n_bets,
            n_wins            = wins,
            n_losses          = losses,
            win_rate          = round(win_rate, 4),
            roi               = round(roi, 4),
            total_profit      = round(total_profit, 2),
            start_bankroll    = start_bankroll,
            end_bankroll      = round(bankroll, 2),
            peak_bankroll     = round(peak_bankroll, 2),
            max_drawdown      = round(max_drawdown, 2),
            max_drawdown_pct  = round(max_drawdown_pct, 4),
            sharpe_ratio      = round(sharpe, 4),
            profit_factor     = round(profit_factor, 4),
            longest_win_streak  = max_win_streak,
            longest_loss_streak = max_loss_streak,
            avg_odds          = round(avg_odds, 4),
            avg_predicted_prob = round(avg_prob, 4),
            avg_implied_prob  = round(avg_impl, 4),
            avg_value         = round(avg_val, 4),
            value_bets_roi    = round(vb_roi, 4),
            non_value_bets_roi = round(nvb_roi, 4),
            calibration_error = round(cal_err, 4),
            bankroll_curve    = bankroll_curve[::10],  # downsample to every 10th
            monthly_roi       = monthly_roi,
            market_breakdown  = market_breakdown,
        )

    def _simple_roi(self, records: List[BetRecord]) -> float:
        """ROI calculation for a flat-bet scenario."""
        if not records:
            return 0.0
        total_profit = sum(r.bookmaker_odds - 1 if r.actual_outcome else -1 for r in records)
        return total_profit / len(records)

    def _calibration_error(self, records: List[BetRecord]) -> float:
        """
        Mean Absolute Calibration Error (MACE).
        Groups bets into probability buckets and checks if actual
        win rate matches predicted probability.
        Perfect: calibration_error = 0
        """
        if not records:
            return 1.0

        buckets = {}
        for r in records:
            bucket = round(r.predicted_prob, 1)
            if bucket not in buckets:
                buckets[bucket] = {"preds": [], "actuals": []}
            buckets[bucket]["preds"].append(r.predicted_prob)
            buckets[bucket]["actuals"].append(r.actual_outcome)

        errors = []
        for bucket, data in buckets.items():
            if len(data["actuals"]) < 5:
                continue
            mean_pred   = float(np.mean(data["preds"]))
            actual_rate = float(np.mean(data["actuals"]))
            errors.append(abs(mean_pred - actual_rate))

        return float(np.mean(errors)) if errors else 0.0

    def _empty_result(self, strategy: str, bankroll: float) -> BacktestResult:
        return BacktestResult(
            strategy=strategy, n_bets=0, n_wins=0, n_losses=0,
            win_rate=0, roi=0, total_profit=0,
            start_bankroll=bankroll, end_bankroll=bankroll,
            peak_bankroll=bankroll, max_drawdown=0, max_drawdown_pct=0,
            sharpe_ratio=0, profit_factor=0,
            longest_win_streak=0, longest_loss_streak=0,
            avg_odds=0, avg_predicted_prob=0, avg_implied_prob=0, avg_value=0,
            value_bets_roi=0, non_value_bets_roi=0, calibration_error=1.0,
        )

    # ── Monte Carlo simulation ────────────────────────────────────

    def monte_carlo(self,
                    records: List[BetRecord],
                    n_simulations: int = 5000,
                    n_bets_per_sim: int = 200,
                    start_bankroll: float = 1000.0,
                    strategy: str = "flat",
                    unit_pct: float = 0.02,
                    kelly_fraction: float = 0.25,
                    ruin_threshold: float = 0.10) -> MonteCarloResult:
        """
        Monte Carlo simulation: randomly samples from the bet pool
        n_simulations times and runs independent bankroll paths.

        Args:
            ruin_threshold: bankroll fraction below which = "ruin" (default 10%)
        """
        if not records:
            return self._empty_mc(n_simulations, n_bets_per_sim)

        random.seed(None)  # Different seed each run

        end_bankrolls  = []
        max_drawdowns  = []
        ruin_count     = 0
        profit_count   = 0
        sample_paths   = []  # Store a few paths for visualisation

        ruin_level = start_bankroll * ruin_threshold

        for sim_idx in range(n_simulations):
            bankroll     = start_bankroll
            peak         = start_bankroll
            max_dd       = 0.0
            path         = [start_bankroll]

            # Sample with replacement
            sample = random.choices(records, k=n_bets_per_sim)

            for rec in sample:
                if bankroll <= ruin_level:
                    break

                if strategy == "flat":
                    stake = start_bankroll * unit_pct
                elif strategy == "kelly":
                    stake = self.staking.kelly_stake(
                        bankroll, rec.predicted_prob, rec.bookmaker_odds, kelly_fraction
                    )
                else:
                    stake = start_bankroll * unit_pct

                stake = min(stake, bankroll)

                if rec.actual_outcome:
                    pnl = stake * (rec.bookmaker_odds - 1.0)
                else:
                    pnl = -stake

                bankroll += pnl

                if bankroll > peak:
                    peak = bankroll
                else:
                    dd = (peak - bankroll) / peak
                    max_dd = max(max_dd, dd)

                if sim_idx < 50:  # Store first 50 paths for plotting
                    path.append(round(bankroll, 2))

            end_bankrolls.append(bankroll)
            max_drawdowns.append(max_dd)

            if bankroll <= ruin_level:
                ruin_count += 1
            if bankroll > start_bankroll:
                profit_count += 1

            if sim_idx < 50:
                sample_paths.append(path)

        # Convert end bankrolls to ROI
        end_rois = [(b - start_bankroll) / start_bankroll for b in end_bankrolls]

        sorted_rois = sorted(end_rois)
        n = len(sorted_rois)

        return MonteCarloResult(
            n_simulations    = n_simulations,
            n_bets_per_sim   = n_bets_per_sim,
            median_roi       = round(float(np.median(sorted_rois)), 4),
            p10_roi          = round(sorted_rois[int(n * 0.10)], 4),
            p25_roi          = round(sorted_rois[int(n * 0.25)], 4),
            p75_roi          = round(sorted_rois[int(n * 0.75)], 4),
            p90_roi          = round(sorted_rois[int(n * 0.90)], 4),
            risk_of_ruin     = round(ruin_count / n_simulations, 4),
            prob_profitable  = round(profit_count / n_simulations, 4),
            expected_roi     = round(float(np.mean(sorted_rois)), 4),
            roi_std          = round(float(np.std(sorted_rois)), 4),
            median_max_drawdown = round(float(np.median(max_drawdowns)), 4),
            bankroll_paths   = sample_paths[:10],  # Return 10 sample paths
        )

    def _empty_mc(self, n_sims, n_bets) -> MonteCarloResult:
        return MonteCarloResult(
            n_simulations=n_sims, n_bets_per_sim=n_bets,
            median_roi=0, p10_roi=0, p25_roi=0, p75_roi=0, p90_roi=0,
            risk_of_ruin=1.0, prob_profitable=0.0, expected_roi=0,
            roi_std=0, median_max_drawdown=1.0,
        )

    # ── Strategy comparison ───────────────────────────────────────

    def compare_strategies(self,
                           records: List[BetRecord],
                           start_bankroll: float = 1000.0) -> Dict[str, BacktestResult]:
        """Run and compare all staking strategies side by side."""
        strategies = {
            "flat_2pct":     self.simulate_strategy(records, "flat",   start_bankroll, 0.02),
            "kelly_quarter": self.simulate_strategy(records, "kelly",  start_bankroll, 0.02, 0.25),
            "kelly_half":    self.simulate_strategy(records, "kelly",  start_bankroll, 0.02, 0.50),
        }

        # Also run on value bets only
        value_records = self._filter_value_bets(records, 0.05)
        if value_records:
            strategies["flat_value_only"] = self.simulate_strategy(
                value_records, "flat", start_bankroll, 0.02
            )

        return strategies

    # ── Rollover chain simulation ─────────────────────────────────

    def simulate_rollover_chain(self,
                                records: List[BetRecord],
                                plan: str = "alpha",
                                target_multiplier: float = 1.10,
                                start_bankroll: float = 1000.0,
                                n_chains: int = 30) -> Dict:
        """
        Simulate n_chains of the rollover strategy.
        Each chain: bet full bankroll until target multiplier is hit or bust.

        Plans:
            alpha: target 1.10x, odds 1.05-1.15
            beta:  target 1.20x, odds 1.15-1.25
            gamma: target 1.50x, odds 1.40-1.60
        """
        odds_ranges = {
            "alpha": (1.05, 1.15),
            "beta":  (1.15, 1.25),
            "gamma": (1.40, 1.60),
        }

        odds_min, odds_max = odds_ranges.get(plan, (1.05, 1.20))

        # Filter records to plan's odds range
        plan_records = [r for r in records if odds_min <= r.bookmaker_odds <= odds_max]

        if not plan_records:
            return {"error": f"No records in odds range {odds_min}-{odds_max}", "plan": plan}

        chain_results = []
        bankroll = start_bankroll

        for chain_num in range(n_chains):
            chain_start = bankroll
            target      = bankroll * target_multiplier
            step        = 0
            success     = False
            steps_taken = []

            # Sample bets for this chain
            chain_pool = random.choices(plan_records, k=20)

            for rec in chain_pool:
                stake = bankroll
                step += 1

                if rec.actual_outcome:
                    pnl = stake * (rec.bookmaker_odds - 1.0)
                    bankroll += pnl
                    steps_taken.append({"step": step, "result": "WIN", "odds": rec.bookmaker_odds,
                                        "bankroll": round(bankroll, 2)})
                    if bankroll >= target:
                        success = True
                        break
                else:
                    bankroll = 0
                    steps_taken.append({"step": step, "result": "LOSS", "odds": rec.bookmaker_odds,
                                        "bankroll": 0})
                    break

            chain_results.append({
                "chain":      chain_num + 1,
                "success":    success,
                "steps":      step,
                "start":      round(chain_start, 2),
                "end":        round(bankroll, 2),
                "gain":       round(bankroll - chain_start, 2),
                "steps_detail": steps_taken,
            })

            if bankroll <= 0:
                bankroll = 0
                print(f"[Rollover] Chain {chain_num+1}: BUST. Bankroll gone.")
                break

        chains_won  = sum(1 for c in chain_results if c["success"])
        chains_lost = len(chain_results) - chains_won
        final_bankroll = bankroll
        total_return = (final_bankroll - start_bankroll) / start_bankroll

        return {
            "plan":           plan,
            "target_multiplier": target_multiplier,
            "start_bankroll": start_bankroll,
            "final_bankroll": round(final_bankroll, 2),
            "total_return":   round(total_return, 4),
            "n_chains_run":   len(chain_results),
            "chains_won":     chains_won,
            "chains_lost":    chains_lost,
            "chain_win_rate": round(chains_won / len(chain_results), 4) if chain_results else 0,
            "avg_steps_per_win": round(
                float(np.mean([c["steps"] for c in chain_results if c["success"]])), 2
            ) if chains_won else 0,
            "chains": chain_results,
        }

    # ── Full backtest report ──────────────────────────────────────

    def run_full_backtest(self,
                          start_bankroll: float = 1000.0,
                          n_monte_carlo: int = 2000,
                          use_synthetic: bool = False) -> Dict:
        """
        Run the complete backtesting suite and return a full report.

        Returns:
            {
                "data_source": "real" | "synthetic",
                "n_records": int,
                "strategies": {strategy: BacktestResult},
                "monte_carlo": {strategy: MonteCarloResult},
                "rollover_chains": {plan: result},
                "calibration": {...},
                "summary": {...},
                "generated_at": timestamp,
            }
        """
        print("[Backtester] Starting full backtest run...")

        # Load data
        if use_synthetic:
            records = self.synthetic.generate(n=3000)
            source  = "synthetic"
        else:
            records = self._load_data()
            source  = "real" if len(self.data_loader.load_settled_predictions()) >= 50 else "synthetic"

        n = len(records)
        print(f"[Backtester] Running on {n} records...")

        # ── Strategy comparison ─────────────────────────────────
        strategies = self.compare_strategies(records, start_bankroll)

        # ── Monte Carlo simulations ─────────────────────────────
        print("[Backtester] Running Monte Carlo simulations...")
        mc_flat = self.monte_carlo(
            records, n_simulations=n_monte_carlo, n_bets_per_sim=min(200, n),
            start_bankroll=start_bankroll, strategy="flat"
        )
        mc_kelly = self.monte_carlo(
            records, n_simulations=n_monte_carlo, n_bets_per_sim=min(200, n),
            start_bankroll=start_bankroll, strategy="kelly"
        )

        # ── Rollover chain sims ─────────────────────────────────
        print("[Backtester] Simulating rollover chains...")
        rollover_chains = {}
        for plan, mult in [("alpha", 1.10), ("beta", 1.20), ("gamma", 1.50)]:
            rollover_chains[plan] = self.simulate_rollover_chain(
                records, plan=plan, target_multiplier=mult,
                start_bankroll=start_bankroll, n_chains=30
            )

        # ── Calibration analysis ────────────────────────────────
        calibration = self._full_calibration_report(records)

        # ── Summary ─────────────────────────────────────────────
        flat_result = strategies.get("flat_2pct")
        summary = {
            "data_source":        source,
            "n_total_bets":       n,
            "overall_win_rate":   flat_result.win_rate if flat_result else 0,
            "overall_roi_flat":   flat_result.roi      if flat_result else 0,
            "avg_predicted_prob": flat_result.avg_predicted_prob if flat_result else 0,
            "avg_bookmaker_odds": flat_result.avg_odds if flat_result else 0,
            "avg_value_per_bet":  flat_result.avg_value if flat_result else 0,
            "value_bets_roi":     flat_result.value_bets_roi if flat_result else 0,
            "calibration_error":  flat_result.calibration_error if flat_result else 1.0,
            "best_strategy":      max(strategies, key=lambda k: strategies[k].roi),
            "mc_risk_of_ruin_flat":  mc_flat.risk_of_ruin,
            "mc_median_roi_flat":    mc_flat.median_roi,
            "mc_prob_profitable":    mc_flat.prob_profitable,
        }

        print(f"[Backtester] ✅ Backtest complete. Overall ROI (flat): {summary['overall_roi_flat']:.2%}")
        print(f"[Backtester] Win rate: {summary['overall_win_rate']:.1%} | "
              f"Best strategy: {summary['best_strategy']} | "
              f"Risk of ruin: {summary['mc_risk_of_ruin_flat']:.1%}")

        return {
            "generated_at":  datetime.utcnow().isoformat(),
            "data_source":   source,
            "n_records":     n,
            "strategies":    {k: asdict(v) for k, v in strategies.items()},
            "monte_carlo":   {
                "flat":  asdict(mc_flat),
                "kelly": asdict(mc_kelly),
            },
            "rollover_chains": rollover_chains,
            "calibration":   calibration,
            "summary":       summary,
        }

    def _full_calibration_report(self, records: List[BetRecord]) -> Dict:
        """Detailed probability calibration analysis."""
        buckets = {}
        for r in records:
            bucket = round(r.predicted_prob * 10) / 10  # round to nearest 0.1
            if bucket not in buckets:
                buckets[bucket] = {"predicted": [], "actuals": []}
            buckets[bucket]["predicted"].append(r.predicted_prob)
            buckets[bucket]["actuals"].append(r.actual_outcome)

        calibration_points = []
        for bucket in sorted(buckets.keys()):
            data = buckets[bucket]
            if len(data["actuals"]) < 5:
                continue
            mean_pred   = float(np.mean(data["predicted"]))
            actual_rate = float(np.mean(data["actuals"]))
            n           = len(data["actuals"])
            calibration_points.append({
                "predicted_prob":  round(mean_pred, 3),
                "actual_win_rate": round(actual_rate, 3),
                "error":           round(abs(mean_pred - actual_rate), 3),
                "n_bets":          n,
                "well_calibrated": abs(mean_pred - actual_rate) < 0.05,
            })

        overall_cal_error = float(np.mean([p["error"] for p in calibration_points])) \
            if calibration_points else 1.0

        # Brier score
        brier = float(np.mean([(r.predicted_prob - r.actual_outcome)**2 for r in records])) \
            if records else 1.0

        return {
            "calibration_points":    calibration_points,
            "overall_mace":          round(overall_cal_error, 4),
            "brier_score":           round(brier, 4),
            "interpretation": {
                "mace":   "0.00 = perfect, 0.10 = acceptable, >0.15 = needs recalibration",
                "brier":  "0.00 = perfect, 0.25 = random, <0.20 = acceptable for sports betting",
                "status": "GOOD" if overall_cal_error < 0.08 else
                          "ACCEPTABLE" if overall_cal_error < 0.15 else "NEEDS_RECALIBRATION",
            }
        }

    def save_results(self, results: Dict, path: str = "logs/backtest_results.json"):
        """Save backtest results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Remove bankroll curves (too large to store fully)
        results_clean = json.loads(json.dumps(results))
        for strat_data in results_clean.get("strategies", {}).values():
            strat_data.pop("bankroll_curve", None)

        with open(path, "w") as f:
            json.dump(results_clean, f, indent=2)
        print(f"[Backtester] Results saved to {path}")
        return path


# ══════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bt      = Backtester()
    results = bt.run_full_backtest(
        start_bankroll = 1000.0,
        n_monte_carlo  = 2000,
        use_synthetic  = True,  # Set False when real data is available
    )

    # Print summary
    s = results["summary"]
    print("\n" + "═"*60)
    print("  BACKTEST SUMMARY")
    print("═"*60)
    print(f"  Data source:       {results['data_source']}")
    print(f"  Total bets:        {s['n_total_bets']}")
    print(f"  Win rate:          {s['overall_win_rate']:.1%}")
    print(f"  ROI (flat):        {s['overall_roi_flat']:.2%}")
    print(f"  Value bets ROI:    {s['value_bets_roi']:.2%}")
    print(f"  Avg value per bet: {s['avg_value_per_bet']:.2%}")
    print(f"  Calibration error: {s['calibration_error']:.3f}")
    print(f"  Best strategy:     {s['best_strategy']}")
    print(f"  MC Risk of ruin:   {s['mc_risk_of_ruin_flat']:.1%}")
    print(f"  MC Prob. profit:   {s['mc_prob_profitable']:.1%}")
    print(f"  MC Median ROI:     {s['mc_median_roi_flat']:.2%}")
    print("═"*60)

    # Rollover chains
    print("\n  ROLLOVER CHAIN RESULTS")
    print("─"*60)
    for plan, chain in results["rollover_chains"].items():
        if "error" in chain:
            print(f"  {plan.upper()}: {chain['error']}")
        else:
            print(f"  {plan.upper()}: {chain['chains_won']}/{chain['n_chains_run']} chains won "
                  f"({chain['chain_win_rate']:.0%}) | Total return: {chain['total_return']:.1%}")

    bt.save_results(results)
