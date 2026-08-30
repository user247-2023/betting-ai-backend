"""
selcom.py — Selcom C2B collection integration (Tanzania).

Built against https://developers.selcommobile.com — the C2B/Collection Services
flow, which is Selcom's documented way to COLLECT money from customers.

── Outbound (us -> Selcom), signed with HMAC ───────────────────────────
  POST /v1/wallet/pushussd     push a PIN prompt to the payer's handset
  GET  /v1/c2b/query-status    check a transaction if a callback is missed

  Headers on every outbound call:
      Authorization : "SELCOM " + base64(API_KEY)
      Digest-Method : HS256
      Digest        : base64( HMAC_SHA256(signing_string, API_SECRET) )
      Timestamp     : ISO-8601, e.g. 2019-02-26T09:30:46+03:00
      Signed-Fields : comma-separated field names, in signing order
  signing_string = "timestamp=<ts>&field1=v1&field2=v2..."
  (timestamp is ALWAYS first, even though it is not listed in Signed-Fields)

── Inbound (Selcom -> us), authenticated with a Bearer token ───────────
  Selcom calls three endpoints on our side. Per the docs these use a
  DIFFERENT scheme from the outbound calls: a bearer token that WE choose
  and hand to Selcom during onboarding.

      POST /lookup        verify the reference before charging
      POST /validation    final check - a failure or timeout AUTO-REVERSES
                          the customer's money, so it must answer quickly
      POST /notification  payment confirmed; this is where we grant access

  Each must reply with Selcom's result envelope:
      {"reference": <same as request>, "resultcode": "000",
       "result": "SUCCESS", "message": "..."}

  Result codes we may return (from the docs):
      000 success | 010 invalid reference | 012 invalid amount
      014 amount too high | 015 amount too low | 4XX other failure

Env vars:
    SELCOM_API_KEY, SELCOM_API_SECRET, SELCOM_VENDOR_ID
    SELCOM_C2B_TOKEN     the bearer token you give Selcom for callbacks
    SELCOM_BASE_URL      defaults to live; use the test URL while onboarding
"""
import base64
import hashlib
import hmac
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import requests

LIVE_BASE = "https://apigw.selcommobile.com"
TEST_BASE = "https://apigwtest.selcommobile.com"

# Documented Selcom result codes
OK = "000"
ERR_BAD_REFERENCE = "010"
ERR_BAD_AMOUNT = "012"
ERR_AMOUNT_HIGH = "014"
ERR_AMOUNT_LOW = "015"
ERR_GENERAL = "400"

# 000 = done, 111/927 = in progress, 999 = ambiguous (poll later, never retry)
ACCEPTED_CODES = {OK, "111", "927"}


def _cfg() -> Tuple[str, str, str, str]:
    return (os.getenv("SELCOM_API_KEY", "").strip(),
            os.getenv("SELCOM_API_SECRET", "").strip(),
            os.getenv("SELCOM_VENDOR_ID", "").strip(),
            os.getenv("SELCOM_BASE_URL", LIVE_BASE).rstrip("/"))


def configured() -> bool:
    key, secret, vendor, _ = _cfg()
    return bool(key and secret and vendor)


def _timestamp() -> str:
    """ISO-8601 with offset, e.g. 2026-08-30T02:15:00+03:00 (the docs' format)."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _digest(data: Dict, signed_fields: List[str], timestamp: str, secret: str) -> str:
    """Build the signing string exactly as documented, then HMAC-SHA256 it."""
    parts = [f"timestamp={timestamp}"]
    parts += [f"{f}={data.get(f, '')}" for f in signed_fields]
    signing_string = "&".join(parts)
    mac = hmac.new(secret.encode(), signing_string.encode(), hashlib.sha256).digest()
    return base64.b64encode(mac).decode()


def build_headers(data: Dict, signed_fields: List[str]) -> Dict[str, str]:
    key, secret, _v, _b = _cfg()
    ts = _timestamp()
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "SELCOM " + base64.b64encode(key.encode()).decode(),
        "Digest-Method": "HS256",
        "Digest": _digest(data, signed_fields, ts, secret),
        "Timestamp": ts,
        "Signed-Fields": ",".join(signed_fields),
    }


def transid_for(ref: str) -> str:
    """Selcom transids are short alphanumerics; our references contain hyphens."""
    return re.sub(r"[^A-Za-z0-9]", "", ref or "")[:20]


# ── Outbound: prompt the customer to pay ────────────────────────────────
def push_ussd(ref: str, amount: int, msisdn: str, timeout: int = 30) -> Dict:
    """Trigger the wallet PIN prompt on the customer's phone.

    Per the docs, success here only means the wallet provider accepted the
    prompt - NOT that money moved. Confirmation arrives on /notification.
    """
    key, secret, vendor, base = _cfg()
    if not (key and secret and vendor):
        return {"ok": False, "message": "Selcom is not configured."}

    data = {
        "transid": transid_for(ref),
        "utilityref": ref,          # our reference - echoed back in callbacks
        "amount": int(amount),
        "vendor": vendor,
        "msisdn": msisdn,
    }
    signed = ["transid", "utilityref", "amount", "vendor", "msisdn"]
    try:
        r = requests.post(f"{base}/v1/wallet/pushussd",
                          headers=build_headers(data, signed),
                          data=json.dumps(data), timeout=timeout)
        body = r.json() if r.content else {}
    except Exception as e:
        return {"ok": False, "message": f"Could not reach Selcom: {e}"}

    code = str(body.get("resultcode", ""))
    if code in ACCEPTED_CODES:
        return {"ok": True,
                "message": "Check your phone and enter your PIN to confirm.",
                "selcom_reference": body.get("reference", ""), "raw": body}
    return {"ok": False,
            "message": body.get("message") or f"Selcom error {code}", "raw": body}


def query_status(ref: str = "", selcom_reference: str = "", timeout: int = 20) -> Dict:
    """Check a transaction when no callback arrived.

    The docs are explicit: wait ~3 minutes, then poll - never re-send the
    charge, or the customer can be debited twice.
    """
    _k, _s, _v, base = _cfg()
    data = {"reference": selcom_reference} if selcom_reference else {"transid": transid_for(ref)}
    fields = list(data.keys())
    try:
        r = requests.get(f"{base}/v1/c2b/query-status",
                         headers=build_headers(data, fields),
                         params=data, timeout=timeout)
        return r.json() if r.content else {}
    except Exception as e:
        return {"error": str(e)}


# ── Inbound: authenticate Selcom's callbacks ────────────────────────────
def verify_callback(headers: Dict[str, str]) -> Tuple[bool, str]:
    """C2B callbacks carry a bearer token that WE issue to Selcom.

    Without this check anyone who found the URL could POST a fake
    'payment received' and hand themselves a free subscription.
    """
    expected = os.getenv("SELCOM_C2B_TOKEN", "").strip()
    if not expected:
        return False, "c2b_token_not_configured"
    lower = {k.lower(): v for k, v in headers.items()}
    auth = (lower.get("authorization") or "").strip()
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if token and hmac.compare_digest(token, expected):
        return True, "bearer"
    return False, "bad_token"


def reply(reference: str, resultcode: str = OK, message: str = "Success",
          **extra) -> Dict:
    """Selcom's expected response envelope."""
    out = {"reference": reference, "resultcode": resultcode,
           "result": "SUCCESS" if resultcode == OK else "FAILED",
           "message": message}
    out.update(extra)
    return out
