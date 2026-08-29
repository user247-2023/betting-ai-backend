"""
content.py — The two tip streams the app serves.

  1. FREE  — the AI/model pipeline is run ONCE per day and the result cached.
             Every free user that day gets the identical batch. This is the
             difference between one set of AI calls per day and one set per
             user per tap, which is where token budgets die.

  2. PAID  — tips the admin uploads by hand for each rollover tier
             (1.10 / 1.20 / 1.50), and later marks Won or Lost.

Both live in Firestore so they survive Railway redeploys (the local disk is
wiped on every deploy). If Firestore isn't configured yet, everything falls
back to an on-disk JSON cache so nothing breaks — you simply lose the cache
on redeploy.
"""
import json
import os
import secrets
from datetime import datetime, timezone
from typing import Dict, List, Optional

from services.auth_service import identity

_LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "database", "cache")

TIERS = ["1.10", "1.20", "1.50"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalise_tier(value) -> str:
    """Map a plan's target odds onto the nearest supported tier."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "1.20"
    return min(TIERS, key=lambda t: abs(float(t) - f))


# ── local fallback ──────────────────────────────────────────────────────
def _local_path(key: str) -> str:
    os.makedirs(_LOCAL_DIR, exist_ok=True)
    return os.path.join(_LOCAL_DIR, key.replace("/", "_") + ".json")


def _read(path_key: str) -> Optional[Dict]:
    if identity.configured():
        try:
            doc = identity.fs_get(path_key)
            if doc and doc.get("payload"):
                return json.loads(doc["payload"])
        except Exception:
            pass
    try:
        with open(_local_path(path_key)) as f:
            return json.load(f)
    except Exception:
        return None


def _write(path_key: str, data: Dict) -> None:
    blob = json.dumps(data)
    if identity.configured():
        try:
            identity.fs_set(path_key, {"payload": blob, "updated": _now()})
            return
        except Exception:
            pass
    try:
        with open(_local_path(path_key), "w") as f:
            f.write(blob)
    except Exception:
        pass


# ── 1. free daily tip cache ─────────────────────────────────────────────
def cache_key(date: str) -> str:
    return f"daily_tips/{date}"


def get_daily(date: str) -> Optional[Dict]:
    """Today's cached free tips, or None if not generated yet."""
    return _read(cache_key(date))


def save_daily(date: str, payload: Dict) -> None:
    slim = dict(payload)
    slim["cached_at"] = _now()
    slim["cached"] = True
    _write(cache_key(date), slim)


def clear_daily(date: str) -> None:
    _write(cache_key(date), {})
    try:
        os.remove(_local_path(cache_key(date)))
    except Exception:
        pass


# ── 2. admin-curated tips per tier ──────────────────────────────────────
def curated_key(date: str, tier: str) -> str:
    return f"curated_tips/{date}_{tier}"


def get_curated(date: str, tier: str) -> Dict:
    data = _read(curated_key(date, tier)) or {}
    return {"date": date, "tier": tier, "tips": data.get("tips", []),
            "note": data.get("note", ""), "updated": data.get("updated")}


def add_curated(date: str, tier: str, tip: Dict) -> Dict:
    data = get_curated(date, tier)
    rec = {
        "id": secrets.token_hex(4),
        "match": (tip.get("match") or "").strip(),
        "league": (tip.get("league") or "").strip(),
        "market": (tip.get("market") or "").strip(),
        "pick": (tip.get("pick") or "").strip(),
        "odds": tip.get("odds"),
        "kickoff": (tip.get("kickoff") or "").strip(),
        "note": (tip.get("note") or "").strip(),
        "status": "PENDING",
        "created": _now(),
    }
    if not rec["match"] or not rec["pick"]:
        raise ValueError("match and pick are required")
    data["tips"].append(rec)
    _write(curated_key(date, tier),
           {"tips": data["tips"], "note": data.get("note", ""), "updated": _now()})
    return rec


def settle_curated(date: str, tier: str, tip_id: str, status: str) -> bool:
    """Mark one tip WON or LOST (or back to PENDING)."""
    status = (status or "").upper()
    if status not in ("WON", "LOST", "PENDING", "VOID"):
        raise ValueError("status must be WON, LOST, VOID or PENDING")
    data = get_curated(date, tier)
    hit = False
    for t in data["tips"]:
        if t.get("id") == tip_id:
            t["status"] = status
            t["settled_at"] = _now() if status != "PENDING" else None
            hit = True
    if hit:
        _write(curated_key(date, tier),
               {"tips": data["tips"], "note": data.get("note", ""), "updated": _now()})
    return hit


def settle_all(date: str, tier: str, status: str) -> int:
    """Mark every pending tip of the day/tier at once (the slip won or lost)."""
    status = (status or "").upper()
    if status not in ("WON", "LOST", "VOID"):
        raise ValueError("status must be WON, LOST or VOID")
    data = get_curated(date, tier)
    n = 0
    for t in data["tips"]:
        if t.get("status") == "PENDING":
            t["status"] = status
            t["settled_at"] = _now()
            n += 1
    if n:
        _write(curated_key(date, tier),
               {"tips": data["tips"], "note": data.get("note", ""), "updated": _now()})
    return n


def delete_curated(date: str, tier: str, tip_id: str) -> bool:
    data = get_curated(date, tier)
    before = len(data["tips"])
    tips = [t for t in data["tips"] if t.get("id") != tip_id]
    if len(tips) == before:
        return False
    _write(curated_key(date, tier),
           {"tips": tips, "note": data.get("note", ""), "updated": _now()})
    return True


def set_note(date: str, tier: str, note: str) -> None:
    data = get_curated(date, tier)
    _write(curated_key(date, tier),
           {"tips": data["tips"], "note": note or "", "updated": _now()})


def curated_record(tier: str, days: List[str]) -> Dict:
    """Won/lost record across the given dates — the honest track record."""
    won = lost = pending = 0
    for d in days:
        for t in get_curated(d, tier)["tips"]:
            st = t.get("status")
            if st == "WON":
                won += 1
            elif st == "LOST":
                lost += 1
            elif st == "PENDING":
                pending += 1
    total = won + lost
    return {"tier": tier, "won": won, "lost": lost, "pending": pending,
            "settled": total,
            "win_rate": round(won / total * 100, 1) if total else None}
