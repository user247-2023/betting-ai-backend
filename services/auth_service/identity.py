"""
identity.py — Real user accounts and subscriptions.

Replaces the shared-access-code gate with proper per-user accounts:

    Firebase Auth (phone OTP or email)  ->  ID token  ->  this module verifies it
    Firestore                           ->  stores the user + subscription state

Why Firestore and not the local disk: Railway's filesystem is wiped on every
redeploy, and signups/payments arrive continuously — committing each one to
GitHub would trigger a redeploy per signup. Firestore is already used by this
app, is free at this scale, and is the natural pair for Firebase Auth.

SECURITY — the two rules that matter:
  1. Subscription documents are written ONLY by this backend, using a service
     account (which bypasses Firestore security rules). Your rules must DENY
     client writes to `subscriptions/**`, otherwise any user could grant
     themselves a subscription from the browser devtools.
  2. Every request is authenticated by verifying the Firebase ID token's RSA
     signature against Google's published certificates — never by trusting a
     uid sent from the client.

Required Railway env vars:
    FIREBASE_PROJECT_ID
    FIREBASE_CLIENT_EMAIL      (from the service-account JSON)
    FIREBASE_PRIVATE_KEY       (from the same JSON; \\n escapes are handled)
"""
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import requests

# PyJWT + cryptography power token verification. If they are unavailable the
# app must still boot and serve everything else — only account features degrade.
try:
    import jwt
    _CRYPTO_OK = True
except Exception:  # pragma: no cover
    jwt = None
    _CRYPTO_OK = False

_CERT_URL = ("https://www.googleapis.com/robot/v1/metadata/x509/"
             "securetoken@system.gserviceaccount.com")
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_FS_BASE = "https://firestore.googleapis.com/v1"

_certs_cache = {"at": 0, "data": None}
_sa_token = {"at": 0, "token": None}


# ───────────────────────── config ────────────────────────────────────────
def project_id() -> str:
    return os.getenv("FIREBASE_PROJECT_ID", "").strip()


def _private_key() -> str:
    k = os.getenv("FIREBASE_PRIVATE_KEY", "")
    return k.replace("\\n", "\n").strip()


def configured() -> bool:
    return bool(_CRYPTO_OK and project_id()
                and os.getenv("FIREBASE_CLIENT_EMAIL") and _private_key())


# ───────────────────── Firebase ID token verification ────────────────────
def _google_certs() -> Dict[str, str]:
    now = time.time()
    if _certs_cache["data"] and now - _certs_cache["at"] < 3600:
        return _certs_cache["data"]
    r = requests.get(_CERT_URL, timeout=10)
    r.raise_for_status()
    _certs_cache["data"] = r.json()
    _certs_cache["at"] = now
    return _certs_cache["data"]


def verify_id_token(id_token: str) -> Optional[Dict]:
    """Verify a Firebase ID token. Returns its claims, or None if invalid.

    Checks the RSA signature against Google's certs plus issuer, audience and
    expiry. A forged or expired token fails here — the uid is never taken on
    trust from the client.
    """
    if not id_token or not project_id() or not _CRYPTO_OK:
        return None
    try:
        from cryptography.x509 import load_pem_x509_certificate
        header = jwt.get_unverified_header(id_token)
        kid = header.get("kid")
        certs = _google_certs()
        pem = certs.get(kid)
        if not pem:
            return None
        pub = load_pem_x509_certificate(pem.encode()).public_key()
        claims = jwt.decode(
            id_token, pub, algorithms=["RS256"],
            audience=project_id(),
            issuer=f"https://securetoken.google.com/{project_id()}",
            options={"require": ["exp", "iat", "sub"]},
        )
        if not claims.get("sub"):
            return None
        return claims
    except Exception:
        return None


# ─────────────────── Firestore access (service account) ──────────────────
def _access_token() -> Optional[str]:
    """OAuth token for the service account, cached until shortly before expiry."""
    now = time.time()
    if _sa_token["token"] and now < _sa_token["at"] - 60:
        return _sa_token["token"]
    email, key = os.getenv("FIREBASE_CLIENT_EMAIL", "").strip(), _private_key()
    if not (email and key and _CRYPTO_OK):
        return None
    iat = int(now)
    assertion = jwt.encode(
        {"iss": email, "scope": "https://www.googleapis.com/auth/datastore",
         "aud": _TOKEN_URL, "iat": iat, "exp": iat + 3600},
        key, algorithm="RS256")
    r = requests.post(_TOKEN_URL, timeout=15, data={
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "assertion": assertion})
    r.raise_for_status()
    data = r.json()
    _sa_token["token"] = data["access_token"]
    _sa_token["at"] = now + int(data.get("expires_in", 3600))
    return _sa_token["token"]


def _encode(value):
    if value is None:                   return {"nullValue": None}
    if isinstance(value, bool):         return {"booleanValue": value}
    if isinstance(value, int):          return {"integerValue": str(value)}
    if isinstance(value, float):        return {"doubleValue": value}
    if isinstance(value, dict):
        return {"mapValue": {"fields": {k: _encode(v) for k, v in value.items()}}}
    if isinstance(value, list):
        return {"arrayValue": {"values": [_encode(v) for v in value]}}
    return {"stringValue": str(value)}


def _decode(field):
    if not isinstance(field, dict):
        return field
    for k, v in field.items():
        if k == "nullValue":    return None
        if k == "integerValue": return int(v)
        if k == "doubleValue":  return float(v)
        if k == "booleanValue": return bool(v)
        if k == "mapValue":     return {kk: _decode(vv)
                                        for kk, vv in (v.get("fields") or {}).items()}
        if k == "arrayValue":   return [_decode(x) for x in (v.get("values") or [])]
        return v
    return None


def _doc_url(path: str) -> str:
    return f"{_FS_BASE}/projects/{project_id()}/databases/(default)/documents/{path}"


def fs_get(path: str) -> Optional[Dict]:
    tok = _access_token()
    if not tok:
        return None
    r = requests.get(_doc_url(path), timeout=15,
                     headers={"Authorization": f"Bearer {tok}"})
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return {k: _decode(v) for k, v in (r.json().get("fields") or {}).items()}


def fs_set(path: str, data: Dict) -> bool:
    """Create/overwrite the listed fields of a document."""
    tok = _access_token()
    if not tok:
        return False
    params = "&".join(f"updateMask.fieldPaths={k}" for k in data)
    r = requests.patch(f"{_doc_url(path)}?{params}", timeout=15,
                       headers={"Authorization": f"Bearer {tok}",
                                "Content-Type": "application/json"},
                       data=json.dumps({"fields": {k: _encode(v)
                                                   for k, v in data.items()}}))
    r.raise_for_status()
    return True


def fs_list(collection: str, page_size: int = 200) -> list:
    """List documents in a collection. Used by the admin payments queue."""
    tok = _access_token()
    if not tok:
        return []
    url = (f"{_FS_BASE}/projects/{project_id()}/databases/(default)/documents/"
           f"{collection}?pageSize={int(page_size)}")
    out = []
    try:
        r = requests.get(url, timeout=20, headers={"Authorization": f"Bearer {tok}"})
        if r.status_code == 404:
            return []
        r.raise_for_status()
        for doc in (r.json().get("documents") or []):
            rec = {k: _decode(v) for k, v in (doc.get("fields") or {}).items()}
            rec["_id"] = doc.get("name", "").rsplit("/", 1)[-1]
            out.append(rec)
    except Exception:
        return []
    return out


# ─────────────────────────── subscriptions ───────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_subscription(uid: str) -> Dict:
    """Current subscription state for a user. Never raises."""
    try:
        doc = fs_get(f"subscriptions/{uid}") or {}
    except Exception as e:
        return {"active": False, "reason": f"lookup_failed: {e}"}
    expires = doc.get("expires")
    if not expires:
        return {"active": False, "reason": "none"}
    if expires < _now():
        return {"active": False, "reason": "expired", "expires": expires,
                "plan": doc.get("plan")}
    return {"active": True, "expires": expires, "plan": doc.get("plan"),
            "since": doc.get("since")}


def grant_subscription(uid: str, days: int, plan: str = "monthly",
                       payment_ref: str = "") -> Dict:
    """Activate or extend a subscription. Extends from the later of now/expiry
    so a user renewing early never loses days."""
    current = get_subscription(uid)
    base = datetime.now(timezone.utc)
    if current.get("active") and current.get("expires"):
        try:
            cur_exp = datetime.fromisoformat(current["expires"])
            if cur_exp > base:
                base = cur_exp
        except Exception:
            pass
    expires = (base + timedelta(days=days)).isoformat()
    fs_set(f"subscriptions/{uid}", {
        "plan": plan, "expires": expires, "updated": _now(),
        "since": current.get("since") or _now(),
        "last_payment_ref": payment_ref, "days_added": days,
    })
    return {"active": True, "expires": expires, "plan": plan}


def upsert_user(uid: str, claims: Dict) -> None:
    """Record who this user is (phone or email), best-effort."""
    try:
        existing = fs_get(f"users/{uid}") or {}
        fs_set(f"users/{uid}", {
            "uid": uid,
            "phone": claims.get("phone_number") or existing.get("phone") or "",
            "email": claims.get("email") or existing.get("email") or "",
            "created": existing.get("created") or _now(),
            "last_seen": _now(),
        })
    except Exception:
        pass
