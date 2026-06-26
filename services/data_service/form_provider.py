"""
form_provider.py — Builds REAL pre-match data blocks from the local fixtures DB.

This is the ground-truth layer for AI tip generation. It pulls each team's actual
recent results, goal numbers and head-to-head history straight from football.db
(zero external API calls) so the AI analyses real data instead of inventing stats.

The DB reliably contains GOALS and RESULTS only (xG, corners, cards, shots,
possession and odds columns are empty), so this module exposes goals/results-based
signals exclusively. The prompt restricts tips to markets these can actually support.
"""
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "database", "football.db")

# Reuse the project's team-name resolver (live API name -> DB name). Optional.
try:
    from services.data_service.team_matcher import resolve as _resolve_name
except Exception:  # pragma: no cover
    _resolve_name = None

_RECENT_N = 6          # matches of form per team
_H2H_N = 6             # head-to-head meetings
_BULLETS = 5           # recent matches listed per team (aggregates always use all _RECENT_N)
_MAX_FIXTURES = 30     # cap matches sent to the AI (keeps prompt bounded)


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _db_name(conn: sqlite3.Connection, src: str, league_id: int) -> Optional[str]:
    """Map a live-API team name to the name used in the fixtures DB. None if unknown."""
    if not src:
        return None
    # 1) already an exact DB name
    if conn.execute(
        "SELECT 1 FROM fixtures WHERE home_team=? OR away_team=? LIMIT 1", (src, src)
    ).fetchone():
        return src
    # 2) alias table (prefer same league, then any)
    row = conn.execute(
        "SELECT model_name FROM team_aliases WHERE src_name=? AND league_id=? LIMIT 1",
        (src, league_id),
    ).fetchone()
    if not row:
        row = conn.execute(
            "SELECT model_name FROM team_aliases WHERE src_name=? LIMIT 1", (src,)
        ).fetchone()
    if row and row[0]:
        return row[0]
    # 3) fuzzy resolver
    if _resolve_name:
        try:
            name, _method = _resolve_name(src, league_id)
            if name:
                return name
        except Exception:
            pass
    return None


def _recent(conn: sqlite3.Connection, team: str, before: str, n: int) -> List[Dict]:
    """Last n finished matches for `team` before date `before`, from team's perspective."""
    rows = conn.execute(
        "SELECT date, home_team, away_team, home_goals, away_goals "
        "FROM fixtures "
        "WHERE (home_team=? OR away_team=?) AND home_goals IS NOT NULL "
        "AND substr(date,1,10) < ? "
        "ORDER BY date DESC LIMIT ?",
        (team, team, before, n),
    ).fetchall()
    out = []
    for r in rows:
        is_home = (r["home_team"] == team)
        gf = r["home_goals"] if is_home else r["away_goals"]
        ga = r["away_goals"] if is_home else r["home_goals"]
        opp = r["away_team"] if is_home else r["home_team"]
        res = "W" if gf > ga else ("D" if gf == ga else "L")
        out.append({
            "date": (r["date"] or "")[:10], "opp": opp, "gf": gf, "ga": ga,
            "res": res, "ha": "H" if is_home else "A", "total": gf + ga,
        })
    return out


def _aggregate(matches: List[Dict]) -> Optional[Dict]:
    n = len(matches)
    if n == 0:
        return None
    gf = sum(m["gf"] for m in matches)
    ga = sum(m["ga"] for m in matches)
    w = sum(1 for m in matches if m["res"] == "W")
    d = sum(1 for m in matches if m["res"] == "D")
    return {
        "n": n,
        "form": "".join(m["res"] for m in matches),
        "w": w, "d": d, "l": n - w - d,
        "avg_gf": round(gf / n, 2),
        "avg_ga": round(ga / n, 2),
        "clean_sheets": sum(1 for m in matches if m["ga"] == 0),
        "failed_to_score": sum(1 for m in matches if m["gf"] == 0),
        "over25": sum(1 for m in matches if m["total"] >= 3),
        "btts": sum(1 for m in matches if m["gf"] > 0 and m["ga"] > 0),
    }


def _h2h(conn: sqlite3.Connection, a: str, b: str, before: str, n: int) -> List[Dict]:
    rows = conn.execute(
        "SELECT date, home_team, away_team, home_goals, away_goals FROM fixtures "
        "WHERE ((home_team=? AND away_team=?) OR (home_team=? AND away_team=?)) "
        "AND home_goals IS NOT NULL AND substr(date,1,10) < ? "
        "ORDER BY date DESC LIMIT ?",
        (a, b, b, a, before, n),
    ).fetchall()
    return [{
        "date": (r["date"] or "")[:10], "h": r["home_team"], "a": r["away_team"],
        "hg": r["home_goals"], "ag": r["away_goals"], "total": r["home_goals"] + r["away_goals"],
    } for r in rows]


def _team_block(label: str, name: str, agg: Dict, matches: List[Dict]) -> str:
    head = (
        f"{label} ({name}) — last {agg['n']}: form {agg['form']} | "
        f"{agg['w']}W-{agg['d']}D-{agg['l']}L | "
        f"scored {agg['avg_gf']}/game, conceded {agg['avg_ga']}/game | "
        f"{agg['clean_sheets']} clean sheets, {agg['failed_to_score']} blank | "
        f"{agg['over25']}/{agg['n']} games Over 2.5 | {agg['btts']}/{agg['n']} BTTS"
    )
    bullets = [
        f"    - {m['date']} {m['res']} {m['gf']}-{m['ga']} vs {m['opp']} ({m['ha']})"
        for m in matches[:_BULLETS]
    ]
    return head + "\n" + "\n".join(bullets)


def _match_block(conn: sqlite3.Connection, idx: int, fx: Dict) -> Optional[str]:
    league_id = fx.get("league_id", 0)
    home_name = _db_name(conn, fx.get("home", ""), league_id)
    away_name = _db_name(conn, fx.get("away", ""), league_id)
    if not home_name or not away_name:
        return None

    before = fx.get("date") or "9999-99-99"
    hm = _recent(conn, home_name, before, _RECENT_N)
    am = _recent(conn, away_name, before, _RECENT_N)
    ha = _aggregate(hm)
    aa = _aggregate(am)
    # Need real form on BOTH sides, otherwise we cannot ground a tip — skip it.
    if not ha or not aa or ha["n"] < 3 or aa["n"] < 3:
        return None

    parts = [
        f"=== MATCH {idx} ===",
        f"{fx.get('home','')} vs {fx.get('away','')} | "
        f"{fx.get('league','')} ({fx.get('country','')}) | {fx.get('time','')}",
        _team_block("HOME", fx.get("home", ""), ha, hm),
        _team_block("AWAY", fx.get("away", ""), aa, am),
    ]

    h2h = _h2h(conn, home_name, away_name, before, _H2H_N)
    if h2h:
        tot = sum(m["total"] for m in h2h)
        over = sum(1 for m in h2h if m["total"] >= 3)
        lines = [
            f"    - {m['date']} {m['h']} {m['hg']}-{m['ag']} {m['a']}"
            for m in h2h
        ]
        parts.append(
            f"HEAD-TO-HEAD — last {len(h2h)}: avg {round(tot/len(h2h),2)} goals | "
            f"{over}/{len(h2h)} Over 2.5\n" + "\n".join(lines)
        )
    else:
        parts.append("HEAD-TO-HEAD — no prior meetings in database.")

    return "\n".join(parts)


def build_prompt_data(fixtures: List[Dict], date: str, max_fixtures: int = _MAX_FIXTURES) -> str:
    """
    Build the real-data fixture text for the AI prompt.

    Returns a string containing one DATA block per fixture for which we have
    genuine recent results on both teams. Fixtures without enough data are omitted
    (the prompt instructs the AI to tip only listed matches), so the AI can never
    fabricate stats for an unknown team.
    """
    try:
        conn = _conn()
    except Exception:
        return "No match data available."

    blocks: List[str] = []
    try:
        idx = 1
        for fx in fixtures:
            if len(blocks) >= max_fixtures:
                break
            try:
                block = _match_block(conn, idx, fx)
            except Exception:
                block = None
            if block:
                blocks.append(block)
                idx += 1
    finally:
        conn.close()

    if not blocks:
        return (
            "No fixtures today have sufficient real data in the database. "
            "Do NOT generate any tips."
        )
    return "\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Model layer: expose each fixture's REAL scoring rates so a statistical model
# (Dixon-Coles) can compute true market probabilities instead of relying on the
# AI's self-reported confidence. Same data source as above — zero API calls.
# ─────────────────────────────────────────────────────────────────────────────

# Competitions played at neutral venues (no home advantage applied).
_NEUTRAL_HINTS = ("world cup", "friendl", "club world", "olympic", "nations league final")


def normalize_key(name: str) -> str:
    """Stable key for a team name (case/punctuation-insensitive)."""
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def fixture_key(home: str, away: str) -> str:
    """Join key for a fixture, matching how AI tips identify home/away teams."""
    return normalize_key(home) + "|" + normalize_key(away)


def _is_neutral(fx: Dict) -> bool:
    blob = f"{fx.get('league','')} {fx.get('country','')}".lower()
    return any(h in blob for h in _NEUTRAL_HINTS)


def fixture_rates(fixtures: List[Dict], date: str) -> Dict[str, Dict]:
    """
    For each fixture with real data, return the scoring rates the model needs:
      { fixture_key: {home_team, away_team, home_gf, home_ga, home_n,
                      away_gf, away_ga, away_n, neutral} }
    Fixtures without enough data on both sides are omitted.
    """
    try:
        conn = _conn()
    except Exception:
        return {}

    out: Dict[str, Dict] = {}
    try:
        for fx in fixtures:
            try:
                league_id = fx.get("league_id", 0)
                home = _db_name(conn, fx.get("home", ""), league_id)
                away = _db_name(conn, fx.get("away", ""), league_id)
                if not home or not away:
                    continue
                before = fx.get("date") or "9999-99-99"
                ha = _aggregate(_recent(conn, home, before, _RECENT_N))
                aa = _aggregate(_recent(conn, away, before, _RECENT_N))
                if not ha or not aa or ha["n"] < 3 or aa["n"] < 3:
                    continue
                out[fixture_key(fx.get("home", ""), fx.get("away", ""))] = {
                    "home_team": fx.get("home", ""),
                    "away_team": fx.get("away", ""),
                    "home_gf": ha["avg_gf"], "home_ga": ha["avg_ga"], "home_n": ha["n"],
                    "away_gf": aa["avg_gf"], "away_ga": aa["avg_ga"], "away_n": aa["n"],
                    "neutral": _is_neutral(fx),
                }
            except Exception:
                continue
    finally:
        conn.close()
    return out
