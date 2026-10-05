from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.api.v1.auth import current_identity, require_verified


router = APIRouter(prefix="/v1/billing", tags=["billing"])


class CheckoutRequest(BaseModel):
    plan: str


def verify_paddle_signature(body: bytes, header: str, secret: str, now: int | None = None) -> bool:
    parts: Dict[str, list[str]] = {}
    for part in header.split(";"):
        key, separator, value = part.strip().partition("=")
        if separator:
            parts.setdefault(key, []).append(value)
    timestamp = parts.get("ts", [""])[0]
    if not timestamp.isdigit() or not parts.get("h1") or not secret:
        return False
    current = int(time.time()) if now is None else now
    if abs(current - int(timestamp)) > 5:
        return False
    signed = timestamp.encode() + b":" + body
    digest = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return any(hmac.compare_digest(digest, candidate) for candidate in parts["h1"])


def _api_base(environment: str) -> str:
    return "https://sandbox-api.paddle.com" if environment == "sandbox" else "https://api.paddle.com"


def _request_paddle(settings, method: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not settings.paddle_api_key:
        raise HTTPException(503, "Billing is not configured")
    try:
        with httpx.Client(timeout=15) as client:
            response = client.request(
                method, _api_base(settings.paddle_environment) + path,
                headers={"Authorization": "Bearer " + settings.paddle_api_key,
                         "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            return response.json()["data"]
    except (httpx.HTTPError, ValueError, KeyError):
        raise HTTPException(502, "Billing provider is unavailable") from None


@router.post("/checkout")
def checkout(payload: CheckoutRequest, request: Request,
             identity: Dict[str, Any] = Depends(current_identity)):
    settings = request.app.state.settings
    price = {"developer": settings.paddle_developer_price_id,
             "team": settings.paddle_team_price_id}.get(payload.plan)
    if price is None:
        raise HTTPException(422, "Unknown paid plan")
    if not settings.billing_ready(payload.plan):
        raise HTTPException(503, "Billing is not configured")
    user = identity["user"]
    require_verified(identity)
    if user["plan"] != "free":
        raise HTTPException(409, "Manage an existing subscription in the billing portal")
    transaction = _request_paddle(settings, "POST", "/transactions", {
        "items": [{"price_id": price, "quantity": 1}],
        "collection_mode": "automatic",
        "custom_data": {"visioninsight_user_id": user["id"]},
    })
    transaction_id = transaction.get("id")
    if not isinstance(transaction_id, str) or not transaction_id.startswith("txn_"):
        raise HTTPException(502, "Billing provider returned an invalid transaction")
    request.app.state.store.record_checkout(transaction_id, user["id"], payload.plan)
    checkout_url = (transaction.get("checkout") or {}).get("url")
    if not checkout_url or not checkout_url.startswith("https://"):
        raise HTTPException(503, "Configure the approved default payment link in Paddle")
    return {"checkout_url": checkout_url}


@router.post("/portal")
def portal(request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    customer_id = identity["user"].get("paddle_customer_id")
    if not customer_id:
        raise HTTPException(404, "No billing account")
    result = _request_paddle(request.app.state.settings, "POST",
                             f"/customers/{customer_id}/portal-sessions", {})
    url = (((result.get("urls") or {}).get("general") or {}).get("overview"))
    if not url or not url.startswith("https://"):
        raise HTTPException(502, "Billing portal is unavailable")
    return {"portal_url": url}


@router.post("/webhooks/paddle")
async def paddle_webhook(request: Request):
    settings = request.app.state.settings
    body_buffer = bytearray()
    async for chunk in request.stream():
        if len(body_buffer) + len(chunk) > 1024 * 1024:
            raise HTTPException(413, "Webhook too large")
        body_buffer.extend(chunk)
    body = bytes(body_buffer)
    if not verify_paddle_signature(body, request.headers.get("paddle-signature", ""),
                                   settings.paddle_webhook_secret):
        raise HTTPException(401, "Invalid webhook signature")
    try:
        event = json.loads(body)
        event_id = event["event_id"]
        event_type = event["event_type"]
        data = event["data"]
        occurred_at = datetime.fromisoformat(event["occurred_at"].replace("Z", "+00:00"))
        occurred_at = occurred_at.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except (ValueError, TypeError, KeyError):
        raise HTTPException(422, "Invalid webhook payload") from None
    if not event_type.startswith("subscription."):
        return {"accepted": True, "ignored": True}

    subscription_id = data.get("id")
    transaction_id = data.get("transaction_id")
    existing_user = request.app.state.store.get_user_by_subscription(subscription_id) if subscription_id else None
    checkout = request.app.state.store.get_checkout(transaction_id) if transaction_id else None
    if existing_user is None and checkout is None:
        raise HTTPException(422, "Unknown subscription")
    user_id = existing_user["id"] if existing_user else checkout["user_id"]
    custom = data.get("custom_data") or {}
    custom_user_id = custom.get("visioninsight_user_id")
    if custom_user_id is not None and custom_user_id != user_id:
        raise HTTPException(422, "Subscription account mismatch")
    items = data.get("items") or []
    prices = {((item.get("price") or {}).get("id")) for item in items}
    mapped = {settings.paddle_developer_price_id: "developer",
              settings.paddle_team_price_id: "team"}
    mapped.pop("", None)
    plan = next((mapped[price] for price in prices if price in mapped), None)
    if not plan:
        raise HTTPException(422, "Unknown subscription price")
    if existing_user is None and checkout is not None and checkout["plan"] != plan:
        raise HTTPException(422, "Subscription price does not match checkout")
    status = data.get("status")
    if status not in {"active", "trialing", "past_due", "paused", "canceled"}:
        raise HTTPException(422, "Unknown subscription status")
    request.app.state.store.apply_billing_event(
        event_id=event_id, user_id=user_id,
        plan=plan if status == "active" else "free",
        status=status, customer_id=data.get("customer_id"),
        subscription_id=subscription_id, occurred_at=occurred_at,
    )
    return {"accepted": True}
