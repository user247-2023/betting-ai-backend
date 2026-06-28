"""
calibration.py — Track whether the model's probabilities are HONEST.

Every tip we surface carries a predicted_probability. This module:
  1. logs each prediction (pending) into a `calibration` table in football.db,
  2. auto-settles pending rows once the real match result is in the DB,
  3. reports calibration: when the model says 65%, do those actually hit ~65%?

No external calls. Settlement reads the actual final score from `fixtures`, so it
works automatically after you refresh the database.
"""
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "database", "football.db")

try:
    from services.data_service.form_provider import _db_name  # reuse name resolver
except Exception:
    _db_name = None


def _conn(db_path=_DB_PATH):
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    return c


def init_db(db_path=_DB_PATH):
    with _conn(db_path) as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS calibration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT, tip_date TEXT,
            home_team TEXT, away_team TEXT,
            market TEXT, pick TEXT,
            settle_type TEXT, side TEXT, line REAL,
            predicted_probability REAL, prob_source TEXT,
            outcome INTEGER, settled_at TEXT,
            UNIQUE(tip_date, home_team, away_team, settle_type, side, line)
        )""")
        c.commit()


# settle_types we can auto-grade from a final score
_AUTO = {"goals_ou", "btts", "result", "double_chance", "dnb",
         "win_to_nil", "clean_sheet", "odd_even", "team_goals"}


def log_predictions(tips: List[Dict], tip_date: str, db_path=_DB_PATH) -> int:
    """Insert pending calibration rows for tips that carry a probability."""
    init_db(db_path)
    rows = 0
    with _conn(db_path) as c:
        for t in tips:
            p = t.get("predicted_probability")
            st = (t.get("settle_type") or "").lower()
            if p is None or st not in _AUTO:
                continue
            try:
                c.execute(
                    "INSERT OR IGNORE INTO calibration "
                    "(created_at,tip_date,home_team,away_team,market,pick,settle_type,"
                    "side,line,predicted_probability,prob_source,outcome,settled_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)",
                    (datetime.now(timezone.utc).isoformat(), tip_date,
                     t.get("home_team", ""), t.get("away_team", ""),
                     t.get("market", ""), t.get("pick", ""), st,
                     (t.get("side") or ""), t.get("line"),
                     float(p), t.get("prob_source", "")))
                rows += c.total_changes and 1 or 0
            except Exception:
                continue
        c.commit()
    return rows


def _won(settle_type, side, line, pick, home, away, hg, ag):
    """Return 1 win / 0 loss / None void for a bet given the final score."""
    st = settle_type; side = (side or "").lower(); tot = hg + ag
    try:
        if st == "goals_ou":
            L = float(line)
            return int(tot > L) if side == "over" else int(tot < L)
        if st == "btts":
            yes = hg > 0 and ag > 0
            return int(yes) if side == "yes" else int(not yes)
        if st == "result":
            return int({"home": hg > ag, "draw": hg == ag, "away": ag > hg}.get(side, False))
        if st == "double_chance":
            if side == "1x": return int(hg >= ag)
            if side == "12": return int(hg != ag)
            if side == "x2": return int(ag >= hg)
        if st == "dnb":
            if hg == ag: return None  # void
            if side == "home": return int(hg > ag)
            if side == "away": return int(ag > hg)
        if st == "win_to_nil":
            if side == "home": return int(hg > ag and ag == 0)
            if side == "away": return int(ag > hg and hg == 0)
        if st == "clean_sheet":
            if side == "home": return int(ag == 0)
            if side == "away": return int(hg == 0)
        if st == "odd_even":
            return int(tot % 2 == 1) if side == "odd" else int(tot % 2 == 0)
        if st == "team_goals":
            L = float(line); pl = (pick or "").lower()
            who = "home" if home.lower() in pl else ("away" if away.lower() in pl else None)
            g = hg if who == "home" else ag if who == "away" else None
            if g is None: return None
            return int(g > L) if side == "over" else int(g < L)
    except Exception:
        return None
    return None


def settle_pending(db_path=_DB_PATH, days=4) -> Dict:
    """Grade pending predictions against real results now in `fixtures`."""
    init_db(db_path)
    settled = 0
    with _conn(db_path) as c:
        pend = c.execute(
            "SELECT * FROM calibration WHERE outcome IS NULL").fetchall()
        for r in pend:
            home, away = r["home_team"], r["away_team"]
            if _db_name:
                try:
                    home = _db_name(c, r["home_team"], 0) or r["home_team"]
                    away = _db_name(c, r["away_team"], 0) or r["away_team"]
                except Exception:
                    pass
            row = c.execute(
                "SELECT home_goals, away_goals FROM fixtures "
                "WHERE home_team=? AND away_team=? AND home_goals IS NOT NULL "
                "AND ABS(julianday(substr(date,1,10)) - julianday(?)) <= ? "
                "ORDER BY ABS(julianday(substr(date,1,10)) - julianday(?)) LIMIT 1",
                (home, away, r["tip_date"], days, r["tip_date"])).fetchone()
            if not row:
                continue
            out = _won(r["settle_type"], r["side"], r["line"], r["pick"],
                       r["home_team"], r["away_team"], row["home_goals"], row["away_goals"])
            if out is None:
                # void -> remove from calibration (doesn't count)
                c.execute("DELETE FROM calibration WHERE id=?", (r["id"],))
                continue
            c.execute("UPDATE calibration SET outcome=?, settled_at=? WHERE id=?",
                      (out, datetime.now(timezone.utc).isoformat(), r["id"]))
            settled += 1
        c.commit()
        pending_left = c.execute(
            "SELECT COUNT(*) FROM calibration WHERE outcome IS NULL").fetchone()[0]
    return {"newly_settled": settled, "still_pending": pending_left}


def report(db_path=_DB_PATH) -> Dict:
    """Calibration report: predicted vs actual by probability band + Brier score."""
    init_db(db_path)
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT predicted_probability p, outcome o, prob_source s "
            "FROM calibration WHERE outcome IS NOT NULL").fetchall()
    n = len(rows)
    if n == 0:
        return {"settled": 0, "message": "No settled predictions yet. Generate tips, "
                "refresh the DB after the matches finish, then check again."}

    bands = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    out_bands = []
    for lo, hi in bands:
        grp = [r for r in rows if lo <= r["p"] < hi]
        if not grp:
            continue
        pred = sum(r["p"] for r in grp) / len(grp)
        act = sum(r["o"] for r in grp) / len(grp)
        out_bands.append({
            "band": f"{int(lo*100)}-{int(hi*100)}%",
            "tips": len(grp),
            "predicted": round(pred * 100, 1),
            "actual_hit": round(act * 100, 1),
            "gap": round((act - pred) * 100, 1),
        })

    brier = sum((r["p"] - r["o"]) ** 2 for r in rows) / n
    hit = sum(r["o"] for r in rows) / n
    avg_pred = sum(r["p"] for r in rows) / n
    cal_err = (sum(abs(b["actual_hit"] - b["predicted"]) * b["tips"] for b in out_bands)
               / sum(b["tips"] for b in out_bands)) if out_bands else None

    def src(name):
        g = [r for r in rows if r["s"] == name]
        if not g: return None
        return {"tips": len(g),
                "predicted": round(sum(r["p"] for r in g)/len(g)*100, 1),
                "actual_hit": round(sum(r["o"] for r in g)/len(g)*100, 1)}

    return {
        "settled": n,
        "overall": {"avg_predicted": round(avg_pred*100, 1),
                    "actual_hit": round(hit*100, 1),
                    "brier_score": round(brier, 4),
                    "calibration_error_pts": round(cal_err, 1) if cal_err is not None else None},
        "by_band": out_bands,
        "by_source": {"model+ai": src("model+ai") or src("model"), "ai": src("ai")},
        "guide": "Good calibration = 'actual_hit' close to 'predicted' in each band "
                 "(small gap). Brier < 0.25 is decent; lower is better.",
    }
