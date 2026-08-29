"""
subscriptions.py — Access control for the tips feed.

Every /api/tips call must carry a valid access code. Codes are issued by the
owner through key-guarded admin endpoints, carry an expiry date, and track how
many distinct devices use them (so a shared code is visible, not silent).

Storage is a small JSON file next to the database. Railway's filesystem is
ephemeral, so a code only survives a redeploy once it is committed to GitHub —
the admin endpoints take `commit=1` to do that via the existing
github_commit module.

Design notes:
  * The check is enforced SERVER-SIDE. Anything done in the React app is public
    JavaScript and can be bypassed by opening devtools; this file is the real gate.
  * Codes are compared with a constant-time comparison to avoid timing leaks.
  * The owner code never expires and is exempt from device limits.
"""
import json
import os
import secrets
import string
from datetime import datetime, timedelta, timezone
from hmac import compare_digest
from typing import Dict, List, Optional

_FILE = os.getenv("SUBS_FILE", os.path.join(
    os.path.dirname(__file__), "..", "..", "database", "subscriptions.json"))

DEFAULT_MAX_DEVICES = 2
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no confusable 0/O/1/I


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> Dict:
    try:
        with open(_FILE) as f:
            data = json.load(f)
            if isinstance(data, dict) and "codes" in data:
                return data
    except Exception:
        pass
    return {"codes": {}}


def _save(data: Dict) -> None:
    os.makedirs(os.path.dirname(_FILE), exist_ok=True)
    tmp = _FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, _FILE)


def _gen_code() -> str:
    part = lambda n: "".join(secrets.choice(_ALPHABET) for _ in range(n))
    return f"ROLL-{part(4)}-{part(4)}"


def create_code(label: str = "", days: int = 30, owner: bool = False,
                max_devices: int = DEFAULT_MAX_DEVICES) -> Dict:
    """Issue a new access code. `days<=0` means it never expires."""
    data = _load()
    code = _gen_code()
    while code in data["codes"]:
        code = _gen_code()
    expires = None if days <= 0 else (
        datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
    rec = {
        "label": label or ("owner" if owner else "subscriber"),
        "created": _now(), "expires": expires, "active": True,
        "owner": bool(owner), "max_devices": 0 if owner else max_devices,
        "devices": [], "uses": 0, "last_used": None,
    }
    data["codes"][code] = rec
    _save(data)
    return {"code": code, **rec}


def revoke(code: str) -> bool:
    data = _load()
    rec = data["codes"].get(code)
    if not rec:
        return False
    rec["active"] = False
    rec["revoked_at"] = _now()
    _save(data)
    return True


def list_codes() -> List[Dict]:
    data = _load()
    out = []
    for code, rec in data["codes"].items():
        status = "active"
        if not rec.get("active"):
            status = "revoked"
        elif rec.get("expires") and rec["expires"] < _now():
            status = "expired"
        out.append({"code": code, "status": status, "label": rec.get("label"),
                    "expires": rec.get("expires"), "uses": rec.get("uses", 0),
                    "devices": len(rec.get("devices", [])),
                    "max_devices": rec.get("max_devices"),
                    "owner": rec.get("owner", False),
                    "last_used": rec.get("last_used")})
    return sorted(out, key=lambda r: (r["status"] != "active", r["label"] or ""))


def validate(code: str, device_id: str = "") -> Dict:
    """Return {'ok': bool, 'reason': str, 'label': str}. Records usage on success."""
    if not code:
        return {"ok": False, "reason": "no_code"}
    data = _load()

    # constant-time match so a wrong code can't be discovered by timing
    match_key = None
    for k in data["codes"]:
        if compare_digest(k, code):
            match_key = k
            break
    if match_key is None:
        return {"ok": False, "reason": "invalid"}

    rec = data["codes"][match_key]
    if not rec.get("active"):
        return {"ok": False, "reason": "revoked"}
    if rec.get("expires") and rec["expires"] < _now():
        return {"ok": False, "reason": "expired", "expires": rec["expires"]}

    # device limit (owner exempt; max_devices 0 = unlimited)
    devs = rec.setdefault("devices", [])
    limit = rec.get("max_devices") or 0
    if device_id:
        if device_id not in devs:
            if limit and len(devs) >= limit:
                return {"ok": False, "reason": "device_limit",
                        "max_devices": limit}
            devs.append(device_id)
    rec["uses"] = rec.get("uses", 0) + 1
    rec["last_used"] = _now()
    _save(data)
    return {"ok": True, "label": rec.get("label"), "owner": rec.get("owner", False),
            "expires": rec.get("expires")}


def stats() -> Dict:
    rows = list_codes()
    return {"total": len(rows),
            "active": sum(1 for r in rows if r["status"] == "active"),
            "expired": sum(1 for r in rows if r["status"] == "expired"),
            "revoked": sum(1 for r in rows if r["status"] == "revoked")}
