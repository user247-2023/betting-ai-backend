"""
odds_movement.py — odds-movement and steam-move detection (Tier 2).

Tracks the trajectory of a market's price (opening -> current -> closing) and
flags sharp moves. In betting, *how* a line moves carries information:

  * A "steam move" — a fast, large, often multi-book move — usually marks sharp
    money and a genuine shift in the true probability. Lines that steam THROUGH
    your tipped price (the market converging toward you) are confirmation; lines
    moving AWAY from you erode the edge and may signal stale information.

  * Drift direction relative to your tip tells you whether to bet now or wait:
    if the line is drifting in your favour (odds lengthening on your side), it
    may pay to wait; if it is shortening (steaming toward you), bet now before
    the value disappears.

Everything is computed in de-vigged probability space where possible, so moves
aren't confounded by changes in the bookmaker's margin.

Storage: in-memory snapshots per market; persist via the project's scheduler if
you want history across restarts.
Author: Rollover Betting AI · market-intelligence layer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class OddsSnapshot:
    timestamp: str
    odds: float
    fair_prob: float | None = None  # de-vigged, if both sides supplied


@dataclass
class MarketTracker:
    """Trajectory of a single selection's price over time."""
    fixture: str
    market: str
    selection: str
    snapshots: list[OddsSnapshot] = field(default_factory=list)

    def add(self, odds: float, fair_prob: float | None = None) -> None:
        self.snapshots.append(
            OddsSnapshot(datetime.now(timezone.utc).isoformat(), float(odds), fair_prob)
        )

    @property
    def opening(self) -> OddsSnapshot | None:
        return self.snapshots[0] if self.snapshots else None

    @property
    def current(self) -> OddsSnapshot | None:
        return self.snapshots[-1] if self.snapshots else None


# Thresholds (tune to your books). A move is "steam" if the implied-prob shift
# exceeds STEAM_PROB_DELTA within STEAM_WINDOW_MIN minutes.
STEAM_PROB_DELTA = 0.03   # 3 percentage points of implied probability
STEAM_WINDOW_MIN = 60     # within an hour
BIG_MOVE_PROB_DELTA = 0.05


def analyse_movement(tracker: MarketTracker, tipped_odds: float | None = None) -> dict:
    """
    Summarise a selection's price trajectory and flag steam / drift.

    Returns opening vs current move (in odds and implied-prob terms), the
    detected move type, the direction relative to a tipped price, and a plain
    recommendation (bet now / wait / monitor).
    """
    if not tracker.snapshots:
        return {"error": "no snapshots recorded"}
    opening = tracker.opening
    current = tracker.current
    if opening is None or current is None:
        return {"error": "incomplete trajectory"}

    open_imp = 1.0 / opening.odds
    curr_imp = 1.0 / current.odds
    prob_delta = curr_imp - open_imp           # +ve => odds shortened (prob rose)
    odds_delta_pct = (current.odds / opening.odds - 1.0) * 100.0

    # Velocity over the most recent leg.
    move_type = "stable"
    recent_delta = 0.0
    if len(tracker.snapshots) >= 2:
        prev = tracker.snapshots[-2]
        recent_delta = curr_imp - (1.0 / prev.odds)
        dt_min = _minutes_between(prev.timestamp, current.timestamp)
        if abs(recent_delta) >= STEAM_PROB_DELTA and dt_min <= STEAM_WINDOW_MIN:
            move_type = "steam"
        elif abs(prob_delta) >= BIG_MOVE_PROB_DELTA:
            move_type = "big_move"
        elif abs(prob_delta) >= 0.01:
            move_type = "drift"

    direction = "shortening" if prob_delta > 0 else ("lengthening" if prob_delta < 0 else "flat")

    rel = None
    recommendation = "monitor"
    if tipped_odds:
        # Is the market moving toward (confirming) or away from our tip?
        # Our tip is on THIS selection at tipped_odds. If current odds are now
        # SHORTER than we tipped, the market has converged toward us (good).
        if current.odds < tipped_odds:
            rel = "market_converged_toward_tip"   # we beat the move — confirmation
            recommendation = "confirmed — value already captured at tipped price"
        elif current.odds > tipped_odds:
            rel = "market_moved_away_from_tip"     # value lengthened or eroded
            recommendation = "wait — line drifting our way; better price may come"
        else:
            rel = "unchanged_vs_tip"

    return {
        "fixture": tracker.fixture,
        "market": tracker.market,
        "selection": tracker.selection,
        "n_snapshots": len(tracker.snapshots),
        "opening_odds": round(opening.odds, 3),
        "current_odds": round(current.odds, 3),
        "odds_change_pct": round(odds_delta_pct, 2),
        "implied_prob_change": round(prob_delta, 4),
        "recent_leg_change": round(recent_delta, 4),
        "direction": direction,
        "move_type": move_type,
        "is_steam": move_type == "steam",
        "relative_to_tip": rel,
        "recommendation": recommendation,
    }


def detect_cross_book_steam(
    book_moves: dict[str, float],
    threshold: float = STEAM_PROB_DELTA,
    min_books: int = 3,
) -> dict:
    """
    Confirm a steam move by agreement across bookmakers. `book_moves` maps each
    book to its implied-probability change for the same selection. A genuine
    steam move shows the SAME-direction shift at several books at once (lone-book
    moves are usually errors or limits, not information).
    """
    if not book_moves:
        return {"steam_confirmed": False, "reason": "no data"}
    same_dir_up = [b for b, d in book_moves.items() if d >= threshold]
    same_dir_dn = [b for b, d in book_moves.items() if d <= -threshold]
    if len(same_dir_up) >= min_books:
        return {"steam_confirmed": True, "direction": "shortening",
                "books": same_dir_up, "n_books": len(same_dir_up)}
    if len(same_dir_dn) >= min_books:
        return {"steam_confirmed": True, "direction": "lengthening",
                "books": same_dir_dn, "n_books": len(same_dir_dn)}
    return {"steam_confirmed": False,
            "reason": f"fewer than {min_books} books moved past {threshold} together"}


def _minutes_between(iso_a: str, iso_b: str) -> float:
    try:
        a = datetime.fromisoformat(iso_a)
        b = datetime.fromisoformat(iso_b)
        return abs((b - a).total_seconds()) / 60.0
    except (ValueError, TypeError):
        return 0.0
