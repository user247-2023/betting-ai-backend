"""
clv_tracker.py — Closing Line Value (CLV) tracking.

CLV is the single most reliable proof that a tipping model has genuine edge.
Win/loss over a few hundred bets is mostly noise; beating the CLOSING line
consistently is not. The closing line (the price right before kick-off, after
all sharp money is in) is the market's best estimate of true probability. If
you repeatedly take prices BETTER than the close, you are, by definition,
finding value — regardless of whether any individual bet won.

Two CLV measures are computed:
    odds CLV   = tipped_odds / closing_odds - 1
                 (+ve => you got a better price than the close)
    prob CLV   = closing_fair_prob - tipped_fair_prob
                 (de-vigged on both ends; cleaner, removes margin distortion)

Aggregate "beat-close rate" (% of tips with positive CLV) and mean CLV are the
headline KPIs. A model holding > ~52-55% beat-close at fair margins is very
likely profitable long-term, even before settling a single result.

Storage: lightweight SQLite (own table); decoupled from the main football.db.
Author: Rollover Betting AI · edge-verification
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from services.decision_engine.devig import devig


@dataclass
class CLVRecord:
    tip_id: str
    fixture: str
    market: str
    selection: str
    tipped_odds: float
    tipped_at: str
    closing_odds: float | None = None
    closed_at: str | None = None
    # de-vigged fair probs (need the opposite side's odds to compute)
    tipped_fair_prob: float | None = None
    closing_fair_prob: float | None = None

    @property
    def odds_clv(self) -> float | None:
        if self.closing_odds and self.closing_odds > 0:
            return self.tipped_odds / self.closing_odds - 1.0
        return None

    @property
    def prob_clv(self) -> float | None:
        if self.tipped_fair_prob is not None and self.closing_fair_prob is not None:
            return self.closing_fair_prob - self.tipped_fair_prob
        return None

    @property
    def beat_close(self) -> bool | None:
        c = self.odds_clv
        return None if c is None else c > 0


class CLVTracker:
    def __init__(self, db_path: str = "database/clv.db") -> None:
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS clv (
                    tip_id            TEXT PRIMARY KEY,
                    fixture           TEXT NOT NULL,
                    market            TEXT NOT NULL,
                    selection         TEXT NOT NULL,
                    tipped_odds       REAL NOT NULL,
                    tipped_at         TEXT NOT NULL,
                    tipped_fair_prob  REAL,
                    closing_odds      REAL,
                    closed_at         TEXT,
                    closing_fair_prob REAL
                )
                """
            )

    # ------------------------------------------------------------------ #
    #  Record a tip at the moment it is issued                           #
    # ------------------------------------------------------------------ #
    def record_tip(
        self,
        tip_id: str,
        fixture: str,
        market: str,
        selection: str,
        tipped_odds: float,
        all_outcome_odds: Sequence[float] | None = None,
        devig_method: str = "power",
        selection_index: int | None = None,
    ) -> CLVRecord:
        """
        Store a tip's price now. If you pass the full set of outcome odds
        (e.g. both O/U sides) and the selection's index, the de-vigged fair
        probability at tip time is computed and stored too.
        """
        tipped_fair = None
        if all_outcome_odds and selection_index is not None:
            tipped_fair = devig(all_outcome_odds, devig_method).fair_probabilities[selection_index]

        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO clv
                   (tip_id, fixture, market, selection, tipped_odds, tipped_at, tipped_fair_prob)
                   VALUES (?,?,?,?,?,?,?)""",
                (tip_id, fixture, market, selection, float(tipped_odds), now, tipped_fair),
            )
        return CLVRecord(tip_id, fixture, market, selection, float(tipped_odds), now,
                         tipped_fair_prob=tipped_fair)

    # ------------------------------------------------------------------ #
    #  Capture the closing line (call from the scheduler at kick-off)    #
    # ------------------------------------------------------------------ #
    def record_close(
        self,
        tip_id: str,
        closing_odds: float,
        all_outcome_odds: Sequence[float] | None = None,
        devig_method: str = "power",
        selection_index: int | None = None,
    ) -> CLVRecord | None:
        closing_fair = None
        if all_outcome_odds and selection_index is not None:
            closing_fair = devig(all_outcome_odds, devig_method).fair_probabilities[selection_index]

        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            cur = c.execute("SELECT * FROM clv WHERE tip_id = ?", (tip_id,))
            row = cur.fetchone()
            if row is None:
                return None
            c.execute(
                "UPDATE clv SET closing_odds=?, closed_at=?, closing_fair_prob=? WHERE tip_id=?",
                (float(closing_odds), now, closing_fair, tip_id),
            )
        return self.get(tip_id)

    def get(self, tip_id: str) -> CLVRecord | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM clv WHERE tip_id = ?", (tip_id,)).fetchone()
        return self._row_to_record(row) if row else None

    # ------------------------------------------------------------------ #
    #  Aggregate KPIs                                                    #
    # ------------------------------------------------------------------ #
    def summary(self, market: str | None = None) -> dict:
        sql = "SELECT * FROM clv WHERE closing_odds IS NOT NULL"
        params: tuple = ()
        if market:
            sql += " AND market = ?"
            params = (market,)
        with self._conn() as c:
            rows = [self._row_to_record(r) for r in c.execute(sql, params).fetchall()]

        if not rows:
            return {"n_closed": 0, "message": "No tips with a captured closing line yet."}

        odds_clvs = [r.odds_clv for r in rows if r.odds_clv is not None]
        prob_clvs = [r.prob_clv for r in rows if r.prob_clv is not None]
        beats = [r.beat_close for r in rows if r.beat_close is not None]

        beat_rate = (sum(beats) / len(beats)) if beats else None
        return {
            "n_closed": len(rows),
            "beat_close_rate": round(beat_rate, 4) if beat_rate is not None else None,
            "mean_odds_clv_pct": round(sum(odds_clvs) / len(odds_clvs) * 100, 3) if odds_clvs else None,
            "median_odds_clv_pct": round(_median(odds_clvs) * 100, 3) if odds_clvs else None,
            "mean_prob_clv_pct": round(sum(prob_clvs) / len(prob_clvs) * 100, 3) if prob_clvs else None,
            "verdict": _clv_verdict(beat_rate),
            "by_market": self._by_market() if market is None else None,
        }

    def _by_market(self) -> dict:
        with self._conn() as c:
            rows = [self._row_to_record(r) for r in
                    c.execute("SELECT * FROM clv WHERE closing_odds IS NOT NULL").fetchall()]
        out: dict[str, dict] = {}
        markets = {r.market for r in rows}
        for m in markets:
            sub = [r for r in rows if r.market == m]
            beats = [r.beat_close for r in sub if r.beat_close is not None]
            clvs = [r.odds_clv for r in sub if r.odds_clv is not None]
            out[m] = {
                "n": len(sub),
                "beat_close_rate": round(sum(beats) / len(beats), 4) if beats else None,
                "mean_clv_pct": round(sum(clvs) / len(clvs) * 100, 3) if clvs else None,
            }
        return out

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> CLVRecord:
        return CLVRecord(
            tip_id=row["tip_id"], fixture=row["fixture"], market=row["market"],
            selection=row["selection"], tipped_odds=row["tipped_odds"],
            tipped_at=row["tipped_at"], closing_odds=row["closing_odds"],
            closed_at=row["closed_at"], tipped_fair_prob=row["tipped_fair_prob"],
            closing_fair_prob=row["closing_fair_prob"],
        )


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _clv_verdict(beat_rate: float | None) -> str:
    if beat_rate is None:
        return "insufficient data"
    if beat_rate >= 0.55:
        return "strong edge — consistently beating the close"
    if beat_rate >= 0.52:
        return "positive edge — likely profitable long-term"
    if beat_rate >= 0.48:
        return "break-even — no demonstrable edge yet"
    return "negative — taking worse-than-closing prices (review the model)"
