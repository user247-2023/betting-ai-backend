"""
gateway.py — Mobile-money subscription payments.

Flow (identical across every Tanzanian aggregator — ClickPesa, Azampay, Selcom):

    1. User picks a plan and enters their mobile-money number.
    2. We create a PENDING payment record, then ask the provider to send a
       USSD push to that number.
    3. The user confirms on their handset with their PIN.
    4. The provider calls our webhook with the result.
    5. We VERIFY the webhook, confirm the payment, and activate the
       subscription in Firestore.

SECURITY — three rules, each of which has burned real products that skipped it:
  * The webhook is authenticated (shared secret and/or HMAC signature). Without
    this, anyone who finds the URL can POST "payment successful" and get a free
    subscription forever.
  * The AMOUNT is checked against the plan price. Otherwise a user can pay
    TZS 100 for a TZS 30,000 plan.
  * Activation is IDEMPOTENT, keyed on the provider's reference. A webhook
    retried five times must not grant five months.

Provider specifics live in one small adapter below and are env-configurable,
because the exact field names differ per aggregator and are only handed to you
once your merchant account is approved. Set PAYMENTS_PROVIDER=manual until then.
"""
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import requests

from services.auth_service import identity

# ── Plans (price in TZS) ────────────────────────────────────────────────
PLANS = {
    "weekly":  {"days": 7,   "price": int(os.getenv("PRICE_WEEKLY",  "5000")),  "label": "1 Week"},
    "monthly": {"days": 30,  "price": int(os.getenv("PRICE_MONTHLY", "15000")), "label": "1 Month"},
    "quarter": {"days": 90,  "price": int(os.getenv("PRICE_QUARTER", "40000")), "label": "3 Months"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def plans_public() -> Dict:
    return {k: {"label": v["label"], "days": v["days"], "price": v["price"],
                "currency": "TZS"} for k, v in PLANS.items()}


# ── Payment records (Firestore, so they survive redeploys) ──────────────
def _payment_path(ref: str) -> str:
    return f"payments/{ref}"


def create_pending(uid: str, plan: str, phone: str) -> Dict:
    if plan not in PLANS:
        raise ValueError("unknown plan")
    ref = "RP-" + secrets.token_hex(8).upper()
    rec = {"ref": ref, "uid": uid, "plan": plan,
           "amount": PLANS[plan]["price"], "currency": "TZS",
           "phone": phone, "status": "PENDING", "created": _now(),
           "provider": os.getenv("PAYMENTS_PROVIDER", "manual")}
    identity.fs_set(_payment_path(ref), rec)
    return rec


def get_payment(ref: str) -> Optional[Dict]:
    try:
        return identity.fs_get(_payment_path(ref))
    except Exception:
        return None


# ── Provider adapter ────────────────────────────────────────────────────
def _provider() -> str:
    return os.getenv("PAYMENTS_PROVIDER", "manual").lower()


def _clickpesa_token() -> Optional[str]:
    """Exchange the client id/secret for a short-lived bearer token."""
    cid, secret = os.getenv("CLICKPESA_CLIENT_ID"), os.getenv("CLICKPESA_API_KEY")
    if not (cid and secret):
        return None
    url = os.getenv("CLICKPESA_TOKEN_URL",
                    "https://api.clickpesa.com/third-parties/generate-token")
    r = requests.post(url, timeout=20,
                      headers={"client-id": cid, "api-key": secret})
    r.raise_for_status()
    data = r.json()
    return (data.get("token") or data.get("access_token") or "").replace("Bearer ", "")


def initiate(rec: Dict) -> Dict:
    """Ask the provider to push a payment prompt to the user's phone.

    Returns {'ok': bool, 'message': str}. On a `manual` provider this simply
    tells the user where to send the money — which is a perfectly valid way to
    run before your merchant account is approved.
    """
    prov = _provider()
    if prov == "manual":
        return {"ok": True, "manual": True,
                "message": (f"Send TZS {rec['amount']:,} to "
                            f"{os.getenv('MANUAL_PAY_NUMBER', '(set MANUAL_PAY_NUMBER)')} "
                            f"and use reference {rec['ref']}. Your subscription is "
                            "activated once the owner confirms the payment.")}

    if prov == "clickpesa":
        try:
            tok = _clickpesa_token()
            if not tok:
                return {"ok": False, "message": "Payment provider not configured."}
            url = os.getenv("CLICKPESA_USSD_URL",
                            "https://api.clickpesa.com/third-parties/payments/initiate-ussd-push-request")
            r = requests.post(url, timeout=30,
                              headers={"Authorization": f"Bearer {tok}",
                                       "Content-Type": "application/json"},
                              data=json.dumps({
                                  "amount": str(rec["amount"]),
                                  "currency": rec["currency"],
                                  "orderReference": rec["ref"],
                                  "phoneNumber": rec["phone"],
                              }))
            if r.status_code >= 300:
                return {"ok": False, "message": f"Provider error {r.status_code}."}
            return {"ok": True, "message": "Check your phone and enter your PIN to confirm."}
        except Exception as e:
            return {"ok": False, "message": f"Could not start payment: {e}"}

    return {"ok": False, "message": f"Unsupported provider '{prov}'."}


# ── Webhook ─────────────────────────────────────────────────────────────
def verify_webhook(headers: Dict[str, str], raw_body: bytes) -> Tuple[bool, str]:
    """Authenticate an incoming webhook.

    Supports a shared secret header and/or an HMAC-SHA256 signature of the raw
    body. At least one MUST be configured — an unauthenticated webhook is a
    free-subscription button for anyone who finds the URL.
    """
    secret = os.getenv("PAYMENTS_WEBHOOK_SECRET", "")
    hmac_key = os.getenv("PAYMENTS_WEBHOOK_HMAC_KEY", "")
    if not (secret or hmac_key):
        return False, "webhook_not_configured"

    lower = {k.lower(): v for k, v in headers.items()}
    if secret:
        sent = lower.get("x-webhook-secret") or lower.get("x-api-key") or ""
        if hmac.compare_digest(sent, secret):
            return True, "secret"
    if hmac_key:
        sig = (lower.get("x-signature") or lower.get("x-hub-signature-256") or "")
        sig = sig.split("=")[-1].strip()
        expected = hmac.new(hmac_key.encode(), raw_body, hashlib.sha256).hexdigest()
        if sig and hmac.compare_digest(sig, expected):
            return True, "hmac"
    return False, "bad_signature"


_SUCCESS = {"success", "successful", "completed", "settled", "paid", "confirmed"}


def handle_webhook(payload: Dict) -> Dict:
    """Process a verified webhook: confirm the payment and grant the plan.

    Idempotent — a repeated delivery for an already-confirmed reference is
    acknowledged without granting a second period.
    """
    ref = str(payload.get("orderReference") or payload.get("reference")
              or payload.get("ref") or "").strip()
    status = str(payload.get("status") or payload.get("paymentStatus") or "").lower()
    if not ref:
        return {"ok": False, "reason": "no_reference"}

    rec = get_payment(ref)
    if not rec:
        return {"ok": False, "reason": "unknown_reference"}

    if rec.get("status") == "CONFIRMED":
        return {"ok": True, "idempotent": True, "reason": "already_confirmed"}

    if status not in _SUCCESS:
        identity.fs_set(_payment_path(ref),
                        {"status": "FAILED", "provider_status": status,
                         "updated": _now()})
        return {"ok": False, "reason": f"not_successful:{status}"}

    # amount check — never activate a plan that wasn't actually paid for
    try:
        paid = float(str(payload.get("amount") or payload.get("collectedAmount") or 0))
    except Exception:
        paid = 0.0
    expected = float(rec.get("amount") or 0)
    if expected and paid and paid + 0.5 < expected:
        identity.fs_set(_payment_path(ref),
                        {"status": "UNDERPAID", "paid": paid, "updated": _now()})
        return {"ok": False, "reason": "underpaid", "paid": paid, "expected": expected}

    plan = rec.get("plan", "monthly")
    days = PLANS.get(plan, PLANS["monthly"])["days"]
    sub = identity.grant_subscription(rec["uid"], days, plan, payment_ref=ref)
    identity.fs_set(_payment_path(ref),
                    {"status": "CONFIRMED", "confirmed_at": _now(), "paid": paid})
    return {"ok": True, "uid": rec["uid"], "plan": plan, "expires": sub["expires"]}
