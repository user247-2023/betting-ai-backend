"""
services/data_service/live_data.py

Live match data on a strict budget. Three jobs:

1. SNAPSHOTS  — one call to The Odds API returns every upcoming match for a
   competition (kickoff times + multi-bookmaker 1X2 and Over/Under odds).
   Each snapshot is stored, timestamped, in odds_events — today's fuel for
   /api/live, tomorrow's food for steam detection.

2. BUDGET     — every response's x-requests-remaining / x-requests-used
   headers are recorded, a daily snapshot cap is enforced, and a reserve is
   kept so the key can never be silently drained. The free plan is 500
   credits/month; one snapshot here costs 2 (h2h + totals, one region).

3. RESULTS    — finished international matches flow in for free from the
   martj42 results.csv on GitHub (same source football.db was built from).
   A delta refresh appends only new rows, so the model can re-learn after
   every World Cup matchday without any paid API at all.

No new dependencies: urllib + sqlite3 only.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

DB_PATH = os.getenv("FOOTBALL_DB", "database/football.db")

ODDS_BASE = "https://api.the-odds-api.com/v4"
RESULTS_CSV = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# The Odds API sport keys -> our league_id convention (API-Football numbering)
SPORT_TO_LEAGUE: dict[str, int] = {
    "soccer_fifa_world_cup": 1,
    "soccer_epl": 39,
    "soccer_efl_champ": 40,
    "soccer_spain_la_liga": 140,
    "soccer_spain_segunda_division": 141,
    "soccer_italy_serie_a": 135,
    "soccer_italy_serie_b": 136,
    "soccer_germany_bundesliga": 78,
    "soccer_germany_bundesliga2": 79,
    "soccer_france_ligue_one": 61,
    "soccer_france_ligue_two": 62,
    "soccer_netherlands_eredivisie": 88,
    "soccer_portugal_primeira_liga": 94,
    "soccer_belgium_first_div": 144,
    "soccer_turkey_super_league": 203,
    "soccer_greece_super_league": 197,
    "soccer_austria_bundesliga": 218,
    "soccer_spl": 179,
}

# Budget knobs (env-tunable, safe defaults)
DAILY_SNAPSHOT_CAP = int(os.getenv("ODDS_DAILY_SNAPSHOTS", "6"))
MIN_CREDIT_RESERVE = int(os.getenv("ODDS_MIN_RESERVE", "20"))
RESULTS_TTL_MIN = int(os.getenv("RESULTS_TTL_MIN", "360"))

_UNMATCHED_CAP = 50


# --------------------------------------------------------------------------- #
#  Storage                                                                    #
# --------------------------------------------------------------------------- #

def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS odds_events (
            event_id   TEXT NOT NULL,
            sport_key  TEXT,
            snap_at    TEXT NOT NULL,
            commence   TEXT,
            home_src   TEXT,
            away_src   TEXT,
            home_model TEXT,
            away_model TEXT,
            league_id  INTEGER,
            books_json TEXT,
            PRIMARY KEY (event_id, snap_at)
        );
        CREATE TABLE IF NOT EXISTS api_usage (
            meter     TEXT NOT NULL,
            day       TEXT NOT NULL,
            used      INTEGER DEFAULT 0,
            remaining INTEGER,
            PRIMARY KEY (meter, day)
        );
        CREATE TABLE IF NOT EXISTS team_aliases (
            src_name   TEXT NOT NULL,
            league_id  INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            method     TEXT,
            PRIMARY KEY (src_name, league_id)
        );
        CREATE TABLE IF NOT EXISTS live_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS predictions (
            event_id    TEXT PRIMARY KEY,
            logged_at   TEXT,
            league_id   INTEGER,
            kickoff     TEXT,
            home_model  TEXT,
            away_model  TEXT,
            model_h REAL, model_d REAL, model_a REAL,
            blend_h REAL, blend_d REAL, blend_a REAL,
            market_h REAL, market_d REAL, market_a REAL,
            pick        TEXT,      -- 'home' / 'draw' / 'away' (blended argmax)
            pick_prob   REAL,
            confidence  TEXT,
            settled     INTEGER DEFAULT 0,
            actual      TEXT,
            actual_score TEXT,
            correct     INTEGER,
            mkt_correct INTEGER,   -- did the market favourite hit (benchmark)
            brier_model  REAL,
            brier_blend  REAL,
            brier_market REAL,
            settled_at  TEXT
        );
        """
    )
    return con


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _today() -> str:
    return _now().strftime("%Y-%m-%d")


def meta_get(key: str, default: str | None = None) -> str | None:
    con = _conn()
    try:
        row = con.execute("SELECT value FROM live_meta WHERE key=?", (key,)).fetchone()
    finally:
        con.close()
    return row[0] if row else default


def meta_set(key: str, value: str) -> None:
    con = _conn()
    con.execute(
        "INSERT OR REPLACE INTO live_meta (key, value) VALUES (?,?)", (key, value)
    )
    con.commit()
    con.close()


# --------------------------------------------------------------------------- #
#  Budget meter                                                               #
# --------------------------------------------------------------------------- #

def _snapshots_today() -> int:
    con = _conn()
    try:
        row = con.execute(
            "SELECT used FROM api_usage WHERE meter='odds_snapshots' AND day=?",
            (_today(),),
        ).fetchone()
    finally:
        con.close()
    return int(row[0]) if row else 0


def _record_snapshot(headers) -> dict:
    """Store usage headers + bump today's snapshot counter."""
    remaining = used = None
    try:
        remaining = int(float(headers.get("x-requests-remaining", "")))
    except (TypeError, ValueError):
        pass
    try:
        used = int(float(headers.get("x-requests-used", "")))
    except (TypeError, ValueError):
        pass

    con = _conn()
    con.execute(
        "INSERT INTO api_usage (meter, day, used, remaining) VALUES ('odds_snapshots', ?, 1, NULL) "
        "ON CONFLICT(meter, day) DO UPDATE SET used = used + 1",
        (_today(),),
    )
    if remaining is not None or used is not None:
        con.execute(
            "INSERT OR REPLACE INTO api_usage (meter, day, used, remaining) "
            "VALUES ('odds_credits', ?, ?, ?)",
            (_today(), used or 0, remaining),
        )
    con.commit()
    con.close()
    return {"credits_remaining": remaining, "credits_used": used}


def budget_status() -> dict:
    snaps = _snapshots_today()
    con = _conn()
    try:
        row = con.execute(
            "SELECT used, remaining, day FROM api_usage WHERE meter='odds_credits' "
            "ORDER BY day DESC LIMIT 1"
        ).fetchone()
    finally:
        con.close()
    credits_used, credits_remaining, seen_day = (row if row else (None, None, None))
    ok, reason = _can_snapshot_check(snaps, credits_remaining)
    return {
        "key_set": bool(os.getenv("ODDS_API_KEY")),
        "snapshots_today": snaps,
        "daily_snapshot_cap": DAILY_SNAPSHOT_CAP,
        "credits_remaining": credits_remaining,
        "credits_used": credits_used,
        "credits_last_seen": seen_day,
        "min_credit_reserve": MIN_CREDIT_RESERVE,
        "can_snapshot": ok,
        "reason": reason,
    }


def _can_snapshot_check(snaps: int, remaining: int | None) -> tuple[bool, str]:
    if not os.getenv("ODDS_API_KEY"):
        return False, "ODDS_API_KEY is not set"
    if snaps >= DAILY_SNAPSHOT_CAP:
        return False, f"daily snapshot cap reached ({snaps}/{DAILY_SNAPSHOT_CAP})"
    if remaining is not None and remaining <= MIN_CREDIT_RESERVE:
        return False, f"credit reserve protection ({remaining} left, reserve {MIN_CREDIT_RESERVE})"
    return True, "ok"


def can_snapshot() -> tuple[bool, str]:
    b = budget_status()
    return b["can_snapshot"], b["reason"]


# --------------------------------------------------------------------------- #
#  Snapshots from The Odds API                                                #
# --------------------------------------------------------------------------- #

def _log_unmatched(src: str, league_id: int, sport: str) -> None:
    try:
        items = json.loads(meta_get("unmatched", "[]"))
    except Exception:
        items = []
    entry = {"src": src, "league_id": league_id, "sport": sport, "at": _now().isoformat()}
    items = [i for i in items if i.get("src") != src][-(_UNMATCHED_CAP - 1):] + [entry]
    meta_set("unmatched", json.dumps(items))


def _parse_books(ev: dict, home_src: str, away_src: str) -> dict:
    """Bookmaker odds -> {book: {"1x2": [h,d,a], "ou_2_5": [over,under]}}."""
    out: dict[str, dict] = {}
    for bm in ev.get("bookmakers", []) or []:
        bk = bm.get("key") or bm.get("title") or "book"
        entry: dict[str, list[float]] = {}
        for mk in bm.get("markets", []) or []:
            if mk.get("key") == "h2h":
                h = d = a = None
                for o in mk.get("outcomes", []) or []:
                    nm, price = o.get("name"), o.get("price")
                    if price is None:
                        continue
                    if nm == home_src:
                        h = float(price)
                    elif nm == away_src:
                        a = float(price)
                    elif nm == "Draw":
                        d = float(price)
                if h and d and a:
                    entry["1x2"] = [h, d, a]
            elif mk.get("key") == "totals":
                over = under = None
                for o in mk.get("outcomes", []) or []:
                    if o.get("point") != 2.5 or o.get("price") is None:
                        continue
                    if o.get("name") == "Over":
                        over = float(o["price"])
                    elif o.get("name") == "Under":
                        under = float(o["price"])
                if over and under:
                    entry["ou_2_5"] = [over, under]
        if entry:
            out[bk] = entry
    return out


def snapshot(sport: str, enforce_budget: bool = True) -> dict:
    """
    One call -> every upcoming match for `sport`, with multi-book odds,
    stored timestamped in odds_events. Respects the daily cap and reserve.
    """
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY is not set")
    league_id = SPORT_TO_LEAGUE.get(sport)
    if league_id is None:
        raise ValueError(f"unknown sport key: {sport}")

    if enforce_budget:
        ok, reason = can_snapshot()
        if not ok:
            return {"skipped": True, "sport": sport, "reason": reason}

    qs = urllib.parse.urlencode(
        {
            "apiKey": key,
            "regions": "eu",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
    )
    url = f"{ODDS_BASE}/sports/{sport}/odds?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "rollover-app/1.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        usage = _record_snapshot(resp.headers)
        events = json.loads(resp.read().decode())

    from services.data_service import team_matcher

    snap_at = _now().isoformat()
    stored = matched = 0
    unmatched: list[str] = []
    con = _conn()
    try:
        for ev in events:
            home_src = ev.get("home_team") or ""
            away_src = ev.get("away_team") or ""
            hm, _m1 = team_matcher.resolve(home_src, league_id)
            am, _m2 = team_matcher.resolve(away_src, league_id)
            for src, model in ((home_src, hm), (away_src, am)):
                if model is None:
                    unmatched.append(src)
                    _log_unmatched(src, league_id, sport)
            if hm and am:
                matched += 1
            books = _parse_books(ev, home_src, away_src)
            con.execute(
                "INSERT OR REPLACE INTO odds_events "
                "(event_id, sport_key, snap_at, commence, home_src, away_src, "
                " home_model, away_model, league_id, books_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    ev.get("id"),
                    sport,
                    snap_at,
                    ev.get("commence_time"),
                    home_src,
                    away_src,
                    hm,
                    am,
                    league_id,
                    json.dumps(books),
                ),
            )
            stored += 1
        con.commit()
    finally:
        con.close()

    meta_set(f"last_snapshot:{sport}", snap_at)
    return {
        "sport": sport,
        "events_stored": stored,
        "fully_matched": matched,
        "unmatched_names": sorted(set(unmatched)),
        **usage,
    }


# --------------------------------------------------------------------------- #
#  Reading stored events                                                      #
# --------------------------------------------------------------------------- #

def _row_to_event(row) -> dict:
    (event_id, sport_key, snap_at, commence, home_src, away_src,
     home_model, away_model, league_id, books_json) = row
    try:
        books = json.loads(books_json or "{}")
    except Exception:
        books = {}
    age_min = None
    try:
        age_min = round(
            (_now() - datetime.fromisoformat(snap_at)).total_seconds() / 60, 1
        )
    except Exception:
        pass
    started = None
    try:
        started = datetime.fromisoformat(commence.replace("Z", "+00:00")) <= _now()
    except Exception:
        pass
    return {
        "event_id": event_id,
        "sport": sport_key,
        "league_id": league_id,
        "commence": commence,
        "started": started,
        "home_src": home_src,
        "away_src": away_src,
        "home_model": home_model,
        "away_model": away_model,
        "books": books,
        "books_count": len(books),
        "snapshot_at": snap_at,
        "snapshot_age_min": age_min,
    }


_LATEST_SQL = (
    "SELECT event_id, sport_key, snap_at, commence, home_src, away_src, "
    "home_model, away_model, league_id, books_json FROM odds_events o "
    "WHERE snap_at = (SELECT MAX(snap_at) FROM odds_events i "
    "                 WHERE i.event_id = o.event_id)"
)


def latest_events(hours_ahead: int = 96, include_started: bool = False) -> list[dict]:
    """Latest stored snapshot of every event kicking off within the window."""
    horizon = (_now() + timedelta(hours=hours_ahead)).isoformat()
    con = _conn()
    try:
        rows = con.execute(
            _LATEST_SQL + " AND commence <= ? ORDER BY commence", (horizon,)
        ).fetchall()
    finally:
        con.close()
    events = [_row_to_event(r) for r in rows]
    if not include_started:
        events = [e for e in events if not e["started"]]
    return events


def get_event(event_id: str) -> dict | None:
    con = _conn()
    try:
        row = con.execute(
            _LATEST_SQL + " AND o.event_id = ?", (event_id,)
        ).fetchone()
    finally:
        con.close()
    return _row_to_event(row) if row else None


# --------------------------------------------------------------------------- #
#  Free results river (martj42 CSV)                                           #
# --------------------------------------------------------------------------- #

def refresh_results(force: bool = False) -> dict:
    """
    Delta-append newly finished international matches to fixtures.
    Free, no key, no credits. Returns how many rows were added.
    """
    last = meta_get("results_last_refresh")
    if last and not force:
        try:
            age_min = (_now() - datetime.fromisoformat(last)).total_seconds() / 60
            if age_min < RESULTS_TTL_MIN:
                return {"skipped": True, "reason": f"fresh ({age_min:.0f} min old)", "added": 0}
        except Exception:
            pass

    con = _conn()
    try:
        row = con.execute(
            "SELECT MAX(date) FROM fixtures WHERE league_id=1 AND home_goals IS NOT NULL"
        ).fetchone()
        latest_in_db = row[0] or "2021-07-01"
        floor = (
            datetime.fromisoformat(latest_in_db) - timedelta(days=3)
        ).strftime("%Y-%m-%d")

        req = urllib.request.Request(
            RESULTS_CSV, headers={"User-Agent": "rollover-app/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()

        added = 0
        for r in csv.DictReader(io.StringIO(raw)):
            d = r.get("date", "")
            if d < floor:
                continue
            hs, as_ = r.get("home_score", ""), r.get("away_score", "")
            if hs == "" or as_ == "" or hs is None or as_ is None:
                continue
            try:
                hg, ag = int(float(hs)), int(float(as_))
            except ValueError:
                continue
            cur = con.execute(
                "SELECT 1 FROM fixtures WHERE league_id=1 AND date=? "
                "AND home_team=? AND away_team=? LIMIT 1",
                (d, r["home_team"], r["away_team"]),
            ).fetchone()
            if cur:
                continue
            con.execute(
                "INSERT INTO fixtures (league_id, league_name, country, season, date, "
                "home_team, away_team, home_goals, away_goals, total_goals, status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    1, "International / World Cup", r.get("tournament", ""),
                    int(d[:4]), d, r["home_team"], r["away_team"],
                    hg, ag, hg + ag, "FT",
                ),
            )
            added += 1
        con.commit()
    finally:
        con.close()

    meta_set("results_last_refresh", _now().isoformat())
    return {"skipped": False, "added": added, "scanned_from": floor}


# --------------------------------------------------------------------------- #
#  Honest probe for his existing API-Football key                             #
# --------------------------------------------------------------------------- #

def apifootball_test() -> dict:
    """
    One real call against API-Football (league=1, season=2026). Reports the
    truth about what the free plan returns — including its season limits.
    Costs exactly 1 of the key's ~100 daily calls.
    """
    key = os.getenv("API_FOOTBALL_KEY")
    if not key:
        return {"configured": False,
                "note": "API_FOOTBALL_KEY is not set on the server."}
    url = "https://v3.football.api-sports.io/fixtures?league=1&season=2026&next=10"
    req = urllib.request.Request(
        url, headers={"x-apisports-key": key, "User-Agent": "rollover-app/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return {"configured": True, "ok": False, "error": str(e)}
    errors = data.get("errors") or {}
    fixtures = data.get("response") or []
    sample = [
        {
            "date": (f.get("fixture") or {}).get("date"),
            "home": ((f.get("teams") or {}).get("home") or {}).get("name"),
            "away": ((f.get("teams") or {}).get("away") or {}).get("name"),
        }
        for f in fixtures[:5]
    ]
    return {
        "configured": True,
        "ok": not errors and bool(fixtures),
        "results": data.get("results", 0),
        "errors": errors,
        "sample": sample,
        "note": (
            "If 'errors' mentions your plan or season access, the free tier "
            "does not cover the 2026 season — exactly why The Odds API is the "
            "primary live source here."
        ),
    }


# --------------------------------------------------------------------------- #
#  Predictions ledger — the scorecard ("does this actually work?")            #
# --------------------------------------------------------------------------- #
#
#  log_predictions()    freezes each pre-kickoff prediction (model / blended /
#                       market probabilities + the pick) keyed by event.
#  settle_predictions() matches finished fixtures (fed for free by the results
#                       river) back to logged predictions and grades them:
#                       did the blended pick hit, and what Brier score did the
#                       model, the blend, and the market each earn.
#  scorecard()          aggregates the graded rows into an honest report —
#                       crucially, whether the blend beats the raw market.

def _argmax_outcome(h: float, d: float, a: float) -> str:
    return "home" if h >= d and h >= a else "draw" if d >= a else "away"


def _brier(probs: dict, actual: str) -> float:
    y = {"home": 0.0, "draw": 0.0, "away": 0.0}
    y[actual] = 1.0
    return sum((probs[k] - y[k]) ** 2 for k in y)


def log_predictions(matches: list[dict]) -> int:
    """Freeze the current pre-kickoff prediction for each matched fixture.

    Cheap and local (no API). Safe to call on every /matches request — rows are
    upserted by event, so a fixture's stored prediction is its latest read
    before kickoff. Once a match starts it stops appearing in /matches, so the
    row naturally freezes. Never overwrites a row already settled.
    """
    logged = 0
    con = _conn()
    try:
        for m in matches:
            if not m.get("matched") or m.get("started"):
                continue
            p = m.get("prediction") or {}
            blend = p.get("prediction_1x2") or p.get("model_1x2")
            model = p.get("model_1x2")
            market = p.get("market_1x2")
            if not blend or not model:
                continue
            pick = _argmax_outcome(blend["home"], blend["draw"], blend["away"])
            pick_prob = blend[pick]
            conf = (p.get("tips") or {}).get("result_pick", {}).get("confidence")
            con.execute(
                """INSERT INTO predictions
                   (event_id, logged_at, league_id, kickoff, home_model, away_model,
                    model_h, model_d, model_a, blend_h, blend_d, blend_a,
                    market_h, market_d, market_a, pick, pick_prob, confidence, settled)
                   VALUES (?,?,?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, 0)
                   ON CONFLICT(event_id) DO UPDATE SET
                     logged_at=excluded.logged_at, kickoff=excluded.kickoff,
                     model_h=excluded.model_h, model_d=excluded.model_d, model_a=excluded.model_a,
                     blend_h=excluded.blend_h, blend_d=excluded.blend_d, blend_a=excluded.blend_a,
                     market_h=excluded.market_h, market_d=excluded.market_d, market_a=excluded.market_a,
                     pick=excluded.pick, pick_prob=excluded.pick_prob, confidence=excluded.confidence
                   WHERE predictions.settled = 0""",
                (
                    m["event_id"], _now().isoformat(), m["league_id"], m.get("kickoff"),
                    m["home"], m["away"],
                    model["home"], model["draw"], model["away"],
                    blend["home"], blend["draw"], blend["away"],
                    (market or {}).get("home"), (market or {}).get("draw"), (market or {}).get("away"),
                    pick, pick_prob, conf,
                ),
            )
            logged += 1
        con.commit()
    finally:
        con.close()
    return logged


def settle_predictions() -> int:
    """Grade pending predictions whose result has since landed. Returns #graded."""
    graded = 0
    con = _conn()
    try:
        pending = con.execute(
            "SELECT event_id, league_id, kickoff, home_model, away_model, "
            "model_h, model_d, model_a, blend_h, blend_d, blend_a, "
            "market_h, market_d, market_a FROM predictions WHERE settled = 0"
        ).fetchall()
        for row in pending:
            (event_id, league_id, kickoff, home, away,
             mh, md, ma, bh, bd, ba, kh, kd, ka) = row
            day = (kickoff or "")[:10]
            if not day:
                continue
            fx = con.execute(
                "SELECT home_goals, away_goals FROM fixtures "
                "WHERE league_id=? AND home_team=? AND away_team=? "
                "AND date BETWEEN date(?, '-1 day') AND date(?, '+1 day') "
                "AND home_goals IS NOT NULL ORDER BY date LIMIT 1",
                (league_id, home, away, day, day),
            ).fetchone()
            if not fx:
                continue
            hg, ag = int(fx[0]), int(fx[1])
            actual = "home" if hg > ag else "away" if ag > hg else "draw"
            model = {"home": mh, "draw": md, "away": ma}
            blend = {"home": bh, "draw": bd, "away": ba}
            pick = _argmax_outcome(bh, bd, ba)
            b_model = _brier(model, actual)
            b_blend = _brier(blend, actual)
            b_market = mkt_correct = None
            if kh is not None and kd is not None and ka is not None:
                market = {"home": kh, "draw": kd, "away": ka}
                b_market = _brier(market, actual)
                mkt_correct = 1 if _argmax_outcome(kh, kd, ka) == actual else 0
            con.execute(
                "UPDATE predictions SET settled=1, actual=?, actual_score=?, correct=?, "
                "mkt_correct=?, brier_model=?, brier_blend=?, brier_market=?, settled_at=? "
                "WHERE event_id=?",
                (actual, f"{hg}-{ag}", 1 if pick == actual else 0, mkt_correct,
                 b_model, b_blend, b_market, _now().isoformat(), event_id),
            )
            graded += 1
        con.commit()
    finally:
        con.close()
    return graded


def scorecard() -> dict:
    """Honest track record: accuracy, sharpness, and blend-vs-market."""
    con = _conn()
    try:
        tracked = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        rows = con.execute(
            "SELECT league_id, correct, mkt_correct, brier_model, brier_blend, "
            "brier_market, confidence FROM predictions WHERE settled=1"
        ).fetchall()
        recent = con.execute(
            "SELECT home_model, away_model, pick, actual, actual_score, correct, kickoff "
            "FROM predictions WHERE settled=1 ORDER BY settled_at DESC LIMIT 10"
        ).fetchall()
    finally:
        con.close()

    n = len(rows)
    base = {
        "tracked": tracked,
        "settled": n,
        "pending": tracked - n,
        "recent": [
            {
                "match": f"{r[0]} v {r[1]}", "pick": r[2], "result": r[3],
                "score": r[4], "hit": bool(r[5]), "kickoff": r[6],
            }
            for r in recent
        ],
    }
    if n == 0:
        base["message"] = (
            "No graded predictions yet. The scorecard fills in as logged matches "
            "finish and their results arrive — check back after the next kickoffs."
        )
        return base

    def _rate(items):
        items = [x for x in items if x is not None]
        return round(sum(items) / len(items), 4) if items else None

    correct = [r[1] for r in rows]
    mkt_correct = [r[2] for r in rows]
    b_model = [r[3] for r in rows]
    b_blend = [r[4] for r in rows]
    b_market = [r[5] for r in rows]

    intl = [r for r in rows if r[0] == 1]
    club = [r for r in rows if r[0] != 1]

    bb, bm = _rate(b_blend), _rate(b_market)
    base.update({
        "hit_rate": _rate(correct),
        "market_favourite_hit_rate": _rate(mkt_correct),
        "brier": {"model": _rate(b_model), "blend": bb, "market": bm},
        "blend_beats_market": (bb is not None and bm is not None and bb < bm),
        "by_competition": {
            "internationals": {
                "settled": len(intl),
                "hit_rate": _rate([r[1] for r in intl]),
                "brier_blend": _rate([r[4] for r in intl]),
            },
            "clubs": {
                "settled": len(club),
                "hit_rate": _rate([r[1] for r in club]),
                "brier_blend": _rate([r[4] for r in club]),
            },
        },
        "note": (
            "Hit rate = how often the blended top pick was correct. Brier is "
            "sharpness (lower is better). 'blend_beats_market' compares the blended "
            "forecast's Brier against the bookmakers' own — the real test of edge."
        ),
    })
    return base
