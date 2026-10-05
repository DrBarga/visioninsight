from __future__ import annotations

import secrets
import logging
import smtplib
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from app.core.security import hash_password, hash_secret, new_api_key, new_session_token, verify_password
from app.core.email import email_ready, send_password_reset, send_verification


router = APIRouter(prefix="/v1", tags=["account"])
COOKIE_NAME = "vi_session"
logger = logging.getLogger(__name__)


class Credentials(BaseModel):
    email: str = Field(min_length=5, max_length=320)
    password: str = Field(min_length=12, max_length=256)


class KeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class VerificationRequest(BaseModel):
    token: str = Field(min_length=20, max_length=200)


class EmailRequest(BaseModel):
    email: str = Field(min_length=5, max_length=320)


class PasswordResetRequest(BaseModel):
    token: str = Field(min_length=20, max_length=200)
    password: str = Field(min_length=12, max_length=256)


def public_user(user: Dict[str, Any], csrf_token: str | None = None) -> Dict[str, Any]:
    result = {"id": user["id"], "email": user["email"], "plan": user["plan"],
              "email_verified": user.get("email_verified", False)}
    if csrf_token is not None:
        result["csrf_token"] = csrf_token
    return result


def _authenticate(request: Request) -> Dict[str, Any]:
    store = request.app.state.store
    authorization = request.headers.get("authorization", "")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.startswith("vi_live_"):
            raise HTTPException(401, "Invalid API key")
        key = store.get_api_key(hash_secret(token))
        if key is None:
            raise HTTPException(401, "Invalid API key")
        user = store.get_user(key["user_id"])
        method = "api_key"
        csrf = None
    else:
        token = request.cookies.get(COOKIE_NAME)
        session = store.get_session(hash_secret(token)) if token else None
        if session is None:
            raise HTTPException(401, "Sign in required")
        user = store.get_user(session["user_id"])
        method = "session"
        csrf = session["csrf_token"]
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            supplied = request.headers.get("x-csrf-token", "")
            if not secrets.compare_digest(supplied, csrf):
                raise HTTPException(403, "CSRF token missing or invalid")
    if user is None:
        raise HTTPException(401, "Account unavailable")
    if method == "api_key" and user["plan"] == "free":
        raise HTTPException(403, "API access requires a Developer or Team plan")
    if not store.allow_request("user:" + user["id"], 120):
        raise HTTPException(429, "Too many requests")
    return {"user": user, "method": method, "csrf_token": csrf}


def current_identity(request: Request) -> Dict[str, Any]:
    return _authenticate(request)


def require_verified(identity: Dict[str, Any]) -> None:
    if not identity["user"].get("email_verified", False):
        raise HTTPException(403, "Verify your email address before continuing")


def _set_session(response: Response, request: Request, user_id: str) -> str:
    token = new_session_token()
    csrf = secrets.token_urlsafe(24)
    request.app.state.store.create_session(user_id, hash_secret(token), csrf)
    response.set_cookie(
        COOKIE_NAME, token, httponly=True, secure=request.app.state.settings.cookie_secure,
        samesite="lax", max_age=14 * 24 * 60 * 60,
    )
    return csrf


def _deliver_verification(settings, address: str, token: str) -> None:
    try:
        send_verification(settings, address, token)
    except (OSError, smtplib.SMTPException):
        logger.exception("Could not send verification email")


@router.post("/auth/register", status_code=201)
def register(credentials: Credentials, request: Request, response: Response,
             background_tasks: BackgroundTasks):
    address = credentials.email.strip().lower()
    if "@" not in address or address.startswith("@") or address.endswith("@"):
        raise HTTPException(422, "Invalid email address")
    client = request.client.host if request.client else "unknown"
    if not request.app.state.store.allow_request("register:" + client, 5):
        raise HTTPException(429, "Too many registration attempts")
    settings = request.app.state.settings
    if settings.environment == "production" and not email_ready(settings):
        raise HTTPException(503, "Registration is unavailable until email delivery is configured")
    try:
        user = request.app.state.store.create_user(
            address, hash_password(credentials.password),
            email_verified=settings.environment != "production",
        )
    except IntegrityError:
        if settings.environment == "production":
            response.status_code = 202
            return {"message": "If registration can proceed, check your email for a verification link"}
        raise HTTPException(409, "Account already exists") from None
    if not user["email_verified"]:
        token = secrets.token_urlsafe(32)
        request.app.state.store.set_email_token(user["id"], hash_secret(token))
        background_tasks.add_task(_deliver_verification, settings, address, token)
        response.status_code = 202
        return {"message": "If registration can proceed, check your email for a verification link"}
    csrf = _set_session(response, request, user["id"])
    return public_user(user, csrf)


@router.post("/auth/login")
def login(credentials: Credentials, request: Request, response: Response):
    client = request.client.host if request.client else "unknown"
    if not request.app.state.store.allow_request("login:" + client, 10):
        raise HTTPException(429, "Too many sign-in attempts")
    user = request.app.state.store.get_user_by_email(credentials.email.strip().lower())
    if user is None or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")
    csrf = _set_session(response, request, user["id"])
    return public_user(user, csrf)


@router.post("/auth/verify-email")
def verify_email(payload: VerificationRequest, request: Request):
    if not request.app.state.store.verify_email(hash_secret(payload.token)):
        raise HTTPException(422, "Verification link is invalid or expired")
    return {"verified": True}


@router.post("/auth/resend-verification")
def resend_verification(request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    user = identity["user"]
    if user["email_verified"]:
        return {"sent": False, "already_verified": True}
    settings = request.app.state.settings
    if not email_ready(settings):
        raise HTTPException(503, "Email delivery is unavailable")
    if not request.app.state.store.allow_request("verify:" + user["id"], 3):
        raise HTTPException(429, "Too many verification emails")
    token = secrets.token_urlsafe(32)
    request.app.state.store.set_email_token(user["id"], hash_secret(token))
    try:
        send_verification(settings, user["email"], token)
    except (OSError, smtplib.SMTPException):
        logger.exception("Could not resend verification email")
        raise HTTPException(502, "Verification email could not be sent") from None
    return {"sent": True}


@router.post("/auth/request-password-reset")
def request_password_reset(payload: EmailRequest, request: Request):
    settings = request.app.state.settings
    if not email_ready(settings):
        raise HTTPException(503, "Password recovery is unavailable")
    client = request.client.host if request.client else "unknown"
    if not request.app.state.store.allow_request("password-reset:" + client, 5):
        raise HTTPException(429, "Too many recovery requests")
    user = request.app.state.store.get_user_by_email(payload.email.strip().lower())
    if user is not None:
        token = secrets.token_urlsafe(32)
        request.app.state.store.set_password_reset_token(user["id"], hash_secret(token))
        try:
            send_password_reset(settings, user["email"], token)
        except (OSError, smtplib.SMTPException):
            logger.exception("Could not send password recovery email")
    return {"message": "If the address is registered, a recovery link has been sent"}


@router.post("/auth/reset-password")
def reset_password(payload: PasswordResetRequest, request: Request, response: Response):
    client = request.client.host if request.client else "unknown"
    if not request.app.state.store.allow_request("password-change:" + client, 10):
        raise HTTPException(429, "Too many recovery attempts")
    if not request.app.state.store.reset_password(hash_secret(payload.token), hash_password(payload.password)):
        raise HTTPException(422, "Recovery link is invalid or expired")
    response.delete_cookie(COOKIE_NAME)
    return {"reset": True}


@router.get("/auth/me")
def me(identity: Dict[str, Any] = Depends(current_identity)):
    return public_user(identity["user"], identity["csrf_token"])


@router.post("/auth/logout")
def logout(request: Request, response: Response, identity: Dict[str, Any] = Depends(current_identity)):
    if identity["method"] == "session":
        request.app.state.store.delete_session(hash_secret(request.cookies[COOKIE_NAME]))
        response.delete_cookie(COOKIE_NAME)
    return {"ok": True}


@router.get("/account/usage")
def usage(request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    from app.api.v1.analyses import PLANS

    user = identity["user"]
    return {**request.app.state.store.usage(user["id"]), "plan": user["plan"],
            "monthly_limit": PLANS[user["plan"]]["monthly_minutes"]}


@router.get("/account/api-keys")
def list_keys(request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    return {"keys": request.app.state.store.list_api_keys(identity["user"]["id"])}


@router.post("/account/api-keys", status_code=201)
def create_key(payload: KeyRequest, request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    require_verified(identity)
    user = identity["user"]
    if user["plan"] == "free":
        raise HTTPException(403, "API keys require a Developer or Team plan")
    token = new_api_key()
    created = request.app.state.store.create_api_key(user["id"], payload.name.strip(),
                                                     hash_secret(token), token[:16])
    return {**created, "key": token}


@router.delete("/account/api-keys/{key_id}")
def revoke_key(key_id: str, request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    if not request.app.state.store.revoke_api_key(identity["user"]["id"], key_id):
        raise HTTPException(404, "API key not found")
    return {"ok": True}
