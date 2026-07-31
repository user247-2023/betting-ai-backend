"""
analysis_builder.py — Deep, structured, GROUNDED analysis for every tip.

This is the honest implementation of a professional betting-analyst review. It
covers every section a serious analyst would write **that our data can actually
support**, and it states plainly which sections it cannot support rather than
inventing them.

Grounded here (real, from the database / model):
    • match context, venue, kickoff
    • recent form: W-D-L, goals for/against, clean sheets, Over 2.5 rate, BTTS rate
    • head-to-head record and scoring pattern
    • model expected goals, model probability, fair odds
    • market comparison: implied probability, edge, +EV / -EV verdict
    • correlation warnings between legs on the same fixture
    • fractional-Kelly stake guidance, only when the bet is genuinely +EV

NOT available, and therefore never asserted (previous versions of this app
hallucinated several of these, which is exactly what this module exists to stop):
    • xG / shots / possession / corners / cards  — those DB columns are NULL
    • injuries, suspensions, confirmed line-ups   — needs a paid feed
    • weather, pitch, referee assignment          — no source wired
    • market movement / sharp money               — needs odds history

Every number below can be traced to a row in football.db or to the fitted
Dixon-Coles model. If a section has no data, it says so.
"""
import os
import sqlite3
from typing import Dict, List, Optional

_DB = os.getenv("FOOTBALL_DB", os.path.join(
    os.path.dirname(__file__), "..", "..", "database", "football.db"))

try:
    from services.data_service.form_provider import _db_name
except Exception:
    _db_name = None

RECENT_N = 6
H2H_N = 6

# Factors we deliberately do not model. Shown to the user so the analysis is
# honest about its own blind spots instead of implying total coverage.
DATA_GAPS = [
    "Injuries, suspensions and confirmed line-ups",
    "Expected goals (xG) and shot-level data",
    "Weather, pitch condition and referee assignment",
    "Market movement / sharp money flow",
]


def _conn():
    c = sqlite3.connect(_DB)
    c.row_factory = sqlite3.Row
    return c


def _resolve(conn, name: str, league_id: int = 0) -> str:
    if _db_name:
        try:
            return _db_name(conn, name, league_id) or name
        except Exception:
            return name
    return name


def _recent(conn, team: str, before: str, n: int = RECENT_N) -> List[Dict]:
    rows = conn.execute(
        "SELECT date, home_team, away_team, home_goals, away_goals FROM fixtures "
        "WHERE (home_team=? OR away_team=?) AND home_goals IS NOT NULL "
        "AND substr(date,1,10) < ? ORDER BY date DESC LIMIT ?",
        (team, team, before, n)).fetchall()
    out = []
    for r in rows:
        at_home = r["home_team"] == team
        gf = r["home_goals"] if at_home else r["away_goals"]
        ga = r["away_goals"] if at_home else r["home_goals"]
        out.append({
            "date": r["date"][:10], "opponent": r["away_team"] if at_home else r["home_team"],
            "venue": "H" if at_home else "A", "gf": gf, "ga": ga,
            "result": "W" if gf > ga else ("D" if gf == ga else "L"),
            "total": gf + ga, "btts": gf > 0 and ga > 0,
        })
    return out


def _form_summary(games: List[Dict]) -> Optional[Dict]:
    if not games:
        return None
    n = len(games)
    w = sum(1 for g in games if g["result"] == "W")
    d = sum(1 for g in games if g["result"] == "D")
    return {
        "matches": n,
        "form_string": "".join(g["result"] for g in games),
        "record": f"{w}W-{d}D-{n-w-d}L",
        "scored_pg": round(sum(g["gf"] for g in games) / n, 2),
        "conceded_pg": round(sum(g["ga"] for g in games) / n, 2),
        "clean_sheets": sum(1 for g in games if g["ga"] == 0),
        "failed_to_score": sum(1 for g in games if g["gf"] == 0),
        "over25": f"{sum(1 for g in games if g['total'] > 2.5)}/{n}",
        "btts": f"{sum(1 for g in games if g['btts'])}/{n}",
        "last5": [f"{g['result']} {g['gf']}-{g['ga']} vs {g['opponent']} ({g['venue']})"
                  for g in games[:5]],
    }


def _h2h(conn, home: str, away: str, before: str) -> Optional[Dict]:
    rows = conn.execute(
        "SELECT date, home_team, away_team, home_goals, away_goals FROM fixtures "
        "WHERE ((home_team=? AND away_team=?) OR (home_team=? AND away_team=?)) "
        "AND home_goals IS NOT NULL AND substr(date,1,10) < ? "
        "ORDER BY date DESC LIMIT ?", (home, away, away, home, before, H2H_N)).fetchall()
    if not rows:
        return None
    n = len(rows)
    tot = sum(r["home_goals"] + r["away_goals"] for r in rows)
    return {
        "meetings": n,
        "avg_goals": round(tot / n, 2),
        "over25": f"{sum(1 for r in rows if r['home_goals']+r['away_goals'] > 2.5)}/{n}",
        "btts": f"{sum(1 for r in rows if r['home_goals'] > 0 and r['away_goals'] > 0)}/{n}",
        "results": [f"{r['date'][:10]} {r['home_team']} {r['home_goals']}-{r['away_goals']} {r['away_team']}"
                    for r in rows[:4]],
    }


def _verdict(prob, value, has_odds, grounded) -> Dict:
    """Honest overall rating. Without a real market price we explicitly refuse
    to call anything a value bet — probability alone is not an edge."""
    if not grounded:
        return {"label": "AVOID", "color": "red",
                "reason": "Not enough recent data on these teams to price this honestly."}
    if not has_odds:
        return {"label": "NO PRICE", "color": "grey",
                "reason": ("Model probability only — no live market price for this "
                           "market, so no edge can be claimed. Check your bookmaker's "
                           "odds against the fair odds below before staking.")}
    if value is None:
        return {"label": "AVOID", "color": "red",
                "reason": "Odds could not be matched to this exact market."}
    if value >= 0.08:
        return {"label": "STRONG", "color": "green",
                "reason": f"Model rates this {prob*100:.0f}% vs a market implying less — a genuine {value*100:.1f}% edge."}
    if value >= 0.03:
        return {"label": "DECENT", "color": "green",
                "reason": f"Small but real positive edge of {value*100:.1f}%."}
    if value >= 0:
        return {"label": "MARGINAL", "color": "amber",
                "reason": f"Edge of just {value*100:.1f}% — inside the model's own margin of error."}
    return {"label": "AVOID", "color": "red",
            "reason": f"Negative edge ({value*100:.1f}%). The price is worse than the true chance."}


def build_analysis(tip: Dict, fixture: Dict, date: str) -> Dict:
    """Full grounded analysis for one tip. Never raises."""
    try:
        home = tip.get("home_team") or fixture.get("home", "")
        away = tip.get("away_team") or fixture.get("away", "")
        lid = int(fixture.get("league_id") or 0)
        conn = _conn()
        try:
            h_db, a_db = _resolve(conn, home, lid), _resolve(conn, away, lid)
            hf = _form_summary(_recent(conn, h_db, date))
            af = _form_summary(_recent(conn, a_db, date))
            h2h = _h2h(conn, h_db, a_db, date)
        finally:
            conn.close()

        prob = tip.get("predicted_probability")
        value = tip.get("value")
        odds = tip.get("bookmaker_odds")
        has_odds = bool(tip.get("real_odds_available")) and isinstance(odds, (int, float))
        grounded = bool(hf and af and hf["matches"] >= 3 and af["matches"] >= 3)

        market = {"has_live_price": has_odds}
        if has_odds:
            market.update({
                "bookmaker_odds": odds,
                "implied_probability": round(100.0 / odds, 1),
                "model_probability": round((prob or 0) * 100, 1),
                "edge_pct": round((value or 0) * 100, 1) if value is not None else None,
                "positive_ev": bool(value is not None and value > 0),
            })
        else:
            market["note"] = ("No live price for this market in the odds feed "
                              "(it carries goal totals only). Compare your "
                              "bookmaker's odds to the fair odds yourself.")

        # Fractional-Kelly guidance, only for a genuinely +EV bet
        stake = {"recommend": False,
                 "note": "No stake recommended without a confirmed positive edge."}
        if has_odds and prob and value and value > 0.02:
            b = odds - 1.0
            kelly = (b * prob - (1 - prob)) / b if b > 0 else 0
            pct = max(0.0, min(kelly * 0.25, 0.02)) * 100
            if pct >= 0.1:
                stake = {"recommend": True, "bankroll_pct": round(pct, 2),
                         "note": (f"Quarter-Kelly suggests {pct:.2f}% of bankroll. "
                                  "Cap any single bet at 2%.")}

        return {
            "verdict": _verdict(prob, value, has_odds, grounded),
            "context": {
                "match": f"{home} vs {away}",
                "league": tip.get("league", ""),
                "kickoff": tip.get("time", "TBD"),
                "venue": "Neutral venue" if fixture.get("neutral") else f"{home} at home",
                "pick": tip.get("pick", ""), "market": tip.get("market", ""),
            },
            "form": {"home": hf, "away": af,
                     "note": None if grounded else
                     "Limited recent data — treat this read with caution."},
            "h2h": h2h or {"note": "No previous meetings in the database."},
            "model": {
                "expected_goals": tip.get("expected_goals"),
                "model_probability": round((prob or 0) * 100, 1),
                "fair_odds": tip.get("fair_odds"),
                "calibrated": bool(tip.get("calibrated")),
                "note": ("Probability from a Dixon-Coles model fitted on ~14,600 "
                         "matches, time-weighted toward recent form."),
            },
            "market": market,
            "stake": stake,
            "data_gaps": DATA_GAPS,
        }
    except Exception as e:
        return {"verdict": {"label": "N/A", "color": "grey",
                            "reason": f"Analysis unavailable: {e}"},
                "data_gaps": DATA_GAPS}


def attach(tips: List[Dict], fixtures: List[Dict], date: str) -> List[Dict]:
    """Attach `analysis` to every tip, plus a correlation warning for tips that
    share a fixture (they are NOT independent legs on an accumulator)."""
    meta = {}
    for f in fixtures or []:
        h, a = (f.get("home") or "").strip(), (f.get("away") or "").strip()
        if h and a:
            meta[f"{h} vs {a}"] = f
    counts: Dict[str, int] = {}
    for t in tips or []:
        counts[t.get("match", "")] = counts.get(t.get("match", ""), 0) + 1
    for t in tips or []:
        fx = meta.get(t.get("match", ""), {})
        t["analysis"] = build_analysis(t, fx, date)
        same = counts.get(t.get("match", ""), 1)
        if same > 1:
            t["analysis"]["correlation"] = (
                f"{same} tips on this same match. They are correlated, not "
                "independent — combining them on one slip multiplies the odds "
                "but not the safety, because one bad result can sink several legs.")
    return tips
