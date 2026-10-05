from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, Column, Float, Integer, MetaData, String, Table, Text
from sqlalchemy import and_, create_engine, delete, insert, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


metadata = MetaData()
users = Table(
    "users", metadata,
    Column("id", String(36), primary_key=True),
    Column("email", String(320), unique=True, nullable=False),
    Column("password_hash", Text, nullable=False),
    Column("plan", String(32), nullable=False, default="free"),
    Column("paddle_customer_id", String(80)),
    Column("paddle_subscription_id", String(80)),
    Column("subscription_status", String(32)),
    Column("created_at", String(40), nullable=False),
)
sessions = Table(
    "sessions", metadata,
    Column("token_hash", String(64), primary_key=True),
    Column("user_id", String(36), nullable=False, index=True),
    Column("csrf_token", String(64), nullable=False),
    Column("expires_at", String(40), nullable=False),
)
email_verifications = Table(
    "email_verifications", metadata,
    Column("user_id", String(36), primary_key=True),
    Column("token_hash", String(64), unique=True),
    Column("expires_at", String(40)),
    Column("verified_at", String(40)),
)
password_resets = Table(
    "password_resets", metadata,
    Column("user_id", String(36), primary_key=True),
    Column("token_hash", String(64), unique=True, nullable=False),
    Column("expires_at", String(40), nullable=False),
    Column("used_at", String(40)),
)
api_keys = Table(
    "api_keys", metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), nullable=False, index=True),
    Column("name", String(100), nullable=False),
    Column("prefix", String(16), nullable=False),
    Column("key_hash", String(64), unique=True, nullable=False),
    Column("created_at", String(40), nullable=False),
    Column("revoked", Boolean, nullable=False, default=False),
)
analyses = Table(
    "analyses", metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), nullable=False, index=True),
    Column("recipe", String(32), nullable=False),
    Column("status", String(20), nullable=False, index=True),
    Column("stage", String(40), nullable=False),
    Column("progress", Integer, nullable=False),
    Column("options_json", Text, nullable=False),
    Column("created_at", String(40), nullable=False),
    Column("updated_at", String(40), nullable=False),
    Column("started_at", String(40)),
    Column("completed_at", String(40)),
    Column("error_code", String(80)),
    Column("error_message", Text),
    Column("duration_sec", Float),
    Column("reserved_minutes", Integer, nullable=False),
    Column("usage_month", String(7), nullable=False),
    Column("cancel_requested", Boolean, nullable=False, default=False),
)
usage_periods = Table(
    "usage_periods", metadata,
    Column("user_id", String(36), primary_key=True),
    Column("month", String(7), primary_key=True),
    Column("used_minutes", Integer, nullable=False, default=0),
    Column("reserved_minutes", Integer, nullable=False, default=0),
)
storage_usage = Table(
    "storage_usage", metadata,
    Column("user_id", String(36), primary_key=True),
    Column("used_bytes", Integer, nullable=False, default=0),
)
analysis_storage = Table(
    "analysis_storage", metadata,
    Column("analysis_id", String(36), primary_key=True),
    Column("user_id", String(36), nullable=False, index=True),
    Column("input_bytes", Integer, nullable=False),
)
billing_events = Table(
    "billing_events", metadata,
    Column("id", String(100), primary_key=True),
    Column("processed_at", String(40), nullable=False),
)
billing_checkouts = Table(
    "billing_checkouts", metadata,
    Column("transaction_id", String(80), primary_key=True),
    Column("user_id", String(36), nullable=False, index=True),
    Column("plan", String(32), nullable=False),
    Column("created_at", String(40), nullable=False),
)
billing_state = Table(
    "billing_state", metadata,
    Column("user_id", String(36), primary_key=True),
    Column("event_at", String(40), nullable=False),
)
request_counters = Table(
    "request_counters", metadata,
    Column("principal", String(80), primary_key=True),
    Column("window", String(16), primary_key=True),
    Column("count", Integer, nullable=False, default=0),
)


class QuotaExceeded(ValueError):
    pass


class Store:
    def __init__(self, database_url: Optional[str] = None):
        if database_url is None:
            database_url = os.getenv("VISIONINSIGHT_DATABASE_URL")
        if not database_url:
            local_path = Path(os.getenv("VISIONINSIGHT_DATA_DIR", "backend/data")).resolve() / "visioninsight.sqlite3"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            database_url = "sqlite:///" + local_path.as_posix()
        options: Dict[str, Any] = {"pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            options["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if database_url in ("sqlite://", "sqlite:///:memory:"):
                options["poolclass"] = StaticPool
        self.engine = create_engine(database_url, **options)
        with self.engine.begin() as connection:
            if self.engine.dialect.name == "postgresql":
                connection.execute(text("SELECT pg_advisory_xact_lock(713981004)"))
            metadata.create_all(connection)

    @staticmethod
    def _row(row: Any) -> Optional[Dict[str, Any]]:
        return dict(row._mapping) if row is not None else None

    def create_user(self, email: str, password_hash: str, email_verified: bool = True) -> Dict[str, Any]:
        user_id = str(uuid.uuid4())
        with self.engine.begin() as connection:
            connection.execute(insert(users).values(
                id=user_id, email=email.lower(), password_hash=password_hash,
                plan="free", created_at=utc_now(),
            ))
            connection.execute(insert(email_verifications).values(
                user_id=user_id, verified_at=utc_now() if email_verified else None,
            ))
        return self.get_user(user_id)

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as connection:
            row = self._row(connection.execute(select(users).where(users.c.id == user_id)).first())
            if row is not None:
                row["email_verified"] = connection.execute(select(email_verifications.c.verified_at).where(
                    email_verifications.c.user_id == user_id,
                )).scalar_one_or_none() is not None
            return row

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as connection:
            row = self._row(connection.execute(select(users).where(users.c.email == email.lower())).first())
        return self.get_user(row["id"]) if row is not None else None

    def set_email_token(self, user_id: str, token_hash: str) -> None:
        expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        with self.engine.begin() as connection:
            try:
                with connection.begin_nested():
                    connection.execute(insert(email_verifications).values(user_id=user_id))
            except IntegrityError:
                pass
            connection.execute(update(email_verifications).where(and_(
                email_verifications.c.user_id == user_id,
                email_verifications.c.verified_at.is_(None),
            )).values(token_hash=token_hash, expires_at=expires))

    def verify_email(self, token_hash: str) -> bool:
        with self.engine.begin() as connection:
            updated = connection.execute(update(email_verifications).where(and_(
                email_verifications.c.token_hash == token_hash,
                email_verifications.c.expires_at > utc_now(),
                email_verifications.c.verified_at.is_(None),
            )).values(verified_at=utc_now(), token_hash=None, expires_at=None))
            return updated.rowcount == 1

    def set_password_reset_token(self, user_id: str, token_hash: str) -> None:
        expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        with self.engine.begin() as connection:
            connection.execute(delete(password_resets).where(password_resets.c.user_id == user_id))
            connection.execute(insert(password_resets).values(
                user_id=user_id, token_hash=token_hash, expires_at=expires,
            ))

    def reset_password(self, token_hash: str, password_hash: str) -> bool:
        with self.engine.begin() as connection:
            result = connection.execute(update(password_resets).where(and_(
                password_resets.c.token_hash == token_hash,
                password_resets.c.expires_at > utc_now(),
                password_resets.c.used_at.is_(None),
            )).values(used_at=utc_now()).returning(password_resets.c.user_id))
            user_id = result.scalar_one_or_none()
            if user_id is None:
                return False
            connection.execute(update(users).where(users.c.id == user_id).values(password_hash=password_hash))
            connection.execute(delete(sessions).where(sessions.c.user_id == user_id))
            connection.execute(update(api_keys).where(api_keys.c.user_id == user_id).values(revoked=True))
            return True

    def create_session(self, user_id: str, token_hash: str, csrf_token: str) -> None:
        expires = (datetime.now(timezone.utc) + timedelta(days=14)).isoformat()
        with self.engine.begin() as connection:
            connection.execute(insert(sessions).values(
                token_hash=token_hash, user_id=user_id, csrf_token=csrf_token, expires_at=expires,
            ))

    def get_session(self, token_hash: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as connection:
            return self._row(connection.execute(select(sessions).where(and_(
                sessions.c.token_hash == token_hash, sessions.c.expires_at > utc_now(),
            ))).first())

    def delete_session(self, token_hash: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(sessions).where(sessions.c.token_hash == token_hash))

    def create_api_key(self, user_id: str, name: str, key_hash: str, prefix: str) -> Dict[str, Any]:
        key_id = str(uuid.uuid4())
        with self.engine.begin() as connection:
            connection.execute(insert(api_keys).values(
                id=key_id, user_id=user_id, name=name, prefix=prefix,
                key_hash=key_hash, created_at=utc_now(), revoked=False,
            ))
        return {"id": key_id, "name": name, "prefix": prefix}

    def get_api_key(self, key_hash: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as connection:
            return self._row(connection.execute(select(api_keys).where(and_(
                api_keys.c.key_hash == key_hash, api_keys.c.revoked.is_(False),
            ))).first())

    def list_api_keys(self, user_id: str) -> list[Dict[str, Any]]:
        with self.engine.connect() as connection:
            return [dict(row._mapping) for row in connection.execute(select(
                api_keys.c.id, api_keys.c.name, api_keys.c.prefix, api_keys.c.created_at,
            ).where(and_(api_keys.c.user_id == user_id, api_keys.c.revoked.is_(False))))]

    def revoke_api_key(self, user_id: str, key_id: str) -> bool:
        with self.engine.begin() as connection:
            result = connection.execute(update(api_keys).where(and_(
                api_keys.c.id == key_id, api_keys.c.user_id == user_id,
            )).values(revoked=True))
            return result.rowcount > 0

    def _ensure_usage_period(self, connection: Any, user_id: str, month: str) -> None:
        try:
            with connection.begin_nested():
                connection.execute(insert(usage_periods).values(
                    user_id=user_id, month=month, used_minutes=0, reserved_minutes=0,
                ))
        except IntegrityError:
            pass

    def create_analysis(
        self, user_id: str, recipe: str, options: Dict[str, Any],
        reserved_minutes: int, monthly_limit: int,
        input_bytes: int, storage_limit_bytes: int,
    ) -> Dict[str, Any]:
        analysis_id = str(uuid.uuid4())
        month = current_month()
        now = utc_now()
        with self.engine.begin() as connection:
            try:
                with connection.begin_nested():
                    connection.execute(insert(storage_usage).values(user_id=user_id, used_bytes=0))
            except IntegrityError:
                pass
            stored = connection.execute(update(storage_usage).where(and_(
                storage_usage.c.user_id == user_id,
                storage_usage.c.used_bytes + input_bytes <= storage_limit_bytes,
            )).values(used_bytes=storage_usage.c.used_bytes + input_bytes))
            if stored.rowcount != 1:
                raise QuotaExceeded("Storage limit reached; delete old analyses or upgrade")
            self._ensure_usage_period(connection, user_id, month)
            reserved = connection.execute(update(usage_periods).where(and_(
                usage_periods.c.user_id == user_id,
                usage_periods.c.month == month,
                usage_periods.c.used_minutes + usage_periods.c.reserved_minutes + reserved_minutes <= monthly_limit,
            )).values(reserved_minutes=usage_periods.c.reserved_minutes + reserved_minutes))
            if reserved.rowcount != 1:
                raise QuotaExceeded("Monthly video-minute limit reached")
            connection.execute(insert(analyses).values(
                id=analysis_id, user_id=user_id, recipe=recipe, status="uploading",
                stage="uploading", progress=0, options_json=json.dumps(options),
                created_at=now, updated_at=now, reserved_minutes=reserved_minutes,
                usage_month=month, cancel_requested=False,
            ))
            connection.execute(insert(analysis_storage).values(
                analysis_id=analysis_id, user_id=user_id, input_bytes=input_bytes,
            ))
        return self.get_analysis(analysis_id)

    def enqueue(self, analysis_id: str) -> None:
        with self.engine.begin() as connection:
            updated = connection.execute(update(analyses).where(and_(
                analyses.c.id == analysis_id, analyses.c.status == "uploading",
            )).values(status="queued", stage="queued", updated_at=utc_now()))
            if updated.rowcount != 1:
                raise ValueError("Analysis is not ready for the queue")

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as connection:
            row = self._row(connection.execute(select(analyses).where(analyses.c.id == analysis_id)).first())
        if row is not None:
            row["options"] = json.loads(row.pop("options_json"))
        return row

    def list_analyses(self, user_id: str) -> list[Dict[str, Any]]:
        with self.engine.connect() as connection:
            ids = connection.execute(select(analyses.c.id).where(analyses.c.user_id == user_id).order_by(
                analyses.c.created_at.desc()
            ).limit(100)).scalars().all()
        return [item for analysis_id in ids if (item := self.get_analysis(analysis_id)) is not None]

    def claim_next(self) -> Optional[Dict[str, Any]]:
        with self.engine.begin() as connection:
            next_id = select(analyses.c.id).where(analyses.c.status == "queued").order_by(
                analyses.c.created_at
            ).limit(1).scalar_subquery()
            result = connection.execute(update(analyses).where(and_(
                analyses.c.id == next_id, analyses.c.status == "queued",
            )).values(status="processing", stage="starting", progress=1,
                     started_at=utc_now(), updated_at=utc_now()).returning(analyses.c.id))
            claimed = result.scalar_one_or_none()
        return self.get_analysis(claimed) if claimed else None

    def recover_interrupted(self) -> int:
        with self.engine.connect() as connection:
            ids = connection.execute(select(analyses.c.id).where(
                analyses.c.status == "processing",
            )).scalars().all()
        for analysis_id in ids:
            self.finish(analysis_id, "failed", error_code="WORKER_INTERRUPTED",
                        error_message="Processing was interrupted; upload the video again")
        return len(ids)

    def update_progress(self, analysis_id: str, stage: str, progress: int) -> None:
        with self.engine.begin() as connection:
            connection.execute(update(analyses).where(and_(
                analyses.c.id == analysis_id, analyses.c.status == "processing",
            )).values(stage=stage, progress=max(0, min(99, progress)), updated_at=utc_now()))

    def is_cancel_requested(self, analysis_id: str) -> bool:
        with self.engine.connect() as connection:
            value = connection.execute(select(analyses.c.cancel_requested).where(
                analyses.c.id == analysis_id
            )).scalar_one_or_none()
        return bool(value)

    def finish(self, analysis_id: str, status: str, duration_sec: Optional[float] = None,
               error_code: Optional[str] = None, error_message: Optional[str] = None) -> None:
        if status not in ("completed", "failed", "cancelled"):
            raise ValueError("Invalid terminal status")
        with self.engine.begin() as connection:
            row = self._row(connection.execute(select(analyses).where(
                analyses.c.id == analysis_id
            )).first())
            if row is None or row["status"] in ("completed", "failed", "cancelled"):
                return
            updated = connection.execute(update(analyses).where(and_(
                analyses.c.id == analysis_id, analyses.c.status == row["status"],
            )).values(
                status=status, stage=status, progress=100,
                updated_at=utc_now(), completed_at=utc_now(),
                duration_sec=duration_sec, reserved_minutes=0,
                error_code=error_code, error_message=error_message,
            ))
            if updated.rowcount != 1:
                return
            reserved = int(row["reserved_minutes"])
            charged = reserved if status == "completed" else 0
            connection.execute(update(usage_periods).where(and_(
                usage_periods.c.user_id == row["user_id"],
                usage_periods.c.month == row["usage_month"],
            )).values(
                used_minutes=usage_periods.c.used_minutes + charged,
                reserved_minutes=usage_periods.c.reserved_minutes - reserved,
            ))

    def cancel(self, user_id: str, analysis_id: str) -> bool:
        row = self.get_analysis(analysis_id)
        if row is None or row["user_id"] != user_id:
            return False
        if row["status"] == "queued":
            self.finish(analysis_id, "cancelled")
        if self.get_analysis(analysis_id)["status"] == "processing":
            with self.engine.begin() as connection:
                connection.execute(update(analyses).where(and_(
                    analyses.c.id == analysis_id, analyses.c.status == "processing",
                )).values(
                    cancel_requested=True, updated_at=utc_now(),
                ))
        return True

    def delete_analysis(self, user_id: str, analysis_id: str) -> bool:
        row = self.get_analysis(analysis_id)
        if row is None or row["user_id"] != user_id or row["status"] in ("processing", "uploading"):
            return False
        if row["status"] == "queued":
            self.finish(analysis_id, "cancelled")
        with self.engine.begin() as connection:
            deleted = connection.execute(delete(analyses).where(and_(
                analyses.c.id == analysis_id, analyses.c.user_id == user_id,
                analyses.c.status.in_(("completed", "failed", "cancelled")),
            )))
            if deleted.rowcount == 1:
                size = connection.execute(select(analysis_storage.c.input_bytes).where(
                    analysis_storage.c.analysis_id == analysis_id,
                )).scalar_one_or_none()
                if size is not None:
                    connection.execute(update(storage_usage).where(storage_usage.c.user_id == user_id).values(
                        used_bytes=storage_usage.c.used_bytes - size,
                    ))
                    connection.execute(delete(analysis_storage).where(
                        analysis_storage.c.analysis_id == analysis_id,
                    ))
            return deleted.rowcount == 1

    def usage(self, user_id: str) -> Dict[str, int | str]:
        month = current_month()
        with self.engine.connect() as connection:
            row = self._row(connection.execute(select(usage_periods).where(and_(
                usage_periods.c.user_id == user_id, usage_periods.c.month == month,
            ))).first())
        return {"month": month, "used_minutes": row["used_minutes"] if row else 0,
                "reserved_minutes": row["reserved_minutes"] if row else 0,
                "stored_input_bytes": self.stored_input_bytes(user_id)}

    def stored_input_bytes(self, user_id: str) -> int:
        with self.engine.connect() as connection:
            return int(connection.execute(select(storage_usage.c.used_bytes).where(
                storage_usage.c.user_id == user_id,
            )).scalar_one_or_none() or 0)

    def expired_analyses(self, retention_days: Dict[str, int]) -> list[Dict[str, str]]:
        now = datetime.now(timezone.utc)
        with self.engine.connect() as connection:
            rows = connection.execute(select(
                analyses.c.id, analyses.c.user_id, analyses.c.created_at, users.c.plan,
            ).join(users, analyses.c.user_id == users.c.id).where(
                analyses.c.status.in_(("completed", "failed", "cancelled")),
            )).mappings().all()
        expired = []
        for row in rows:
            created = datetime.fromisoformat(row["created_at"])
            if now - created > timedelta(days=retention_days.get(row["plan"], 7)):
                expired.append({"id": row["id"], "user_id": row["user_id"]})
        return expired

    def purge_expired_tokens_and_counters(self) -> None:
        old_window = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M")
        with self.engine.begin() as connection:
            connection.execute(delete(request_counters).where(request_counters.c.window < old_window))
            connection.execute(delete(sessions).where(sessions.c.expires_at < utc_now()))
            connection.execute(delete(password_resets).where(password_resets.c.expires_at < utc_now()))
            connection.execute(update(email_verifications).where(and_(
                email_verifications.c.verified_at.is_(None),
                email_verifications.c.expires_at < utc_now(),
            )).values(token_hash=None, expires_at=None))

    def allow_request(self, principal: str, per_minute: int) -> bool:
        window = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
        with self.engine.begin() as connection:
            try:
                with connection.begin_nested():
                    connection.execute(insert(request_counters).values(
                        principal=principal, window=window, count=0,
                    ))
            except IntegrityError:
                pass
            result = connection.execute(update(request_counters).where(and_(
                request_counters.c.principal == principal,
                request_counters.c.window == window,
                request_counters.c.count < per_minute,
            )).values(count=request_counters.c.count + 1))
            return result.rowcount == 1

    def record_checkout(self, transaction_id: str, user_id: str, plan: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(insert(billing_checkouts).values(
                transaction_id=transaction_id, user_id=user_id, plan=plan, created_at=utc_now(),
            ))

    def get_checkout(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as connection:
            return self._row(connection.execute(select(billing_checkouts).where(
                billing_checkouts.c.transaction_id == transaction_id,
            )).first())

    def get_user_by_subscription(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as connection:
            user_id = connection.execute(select(users.c.id).where(
                users.c.paddle_subscription_id == subscription_id,
            )).scalar_one_or_none()
        return self.get_user(user_id) if user_id else None

    def apply_billing_event(self, event_id: str, user_id: str, plan: str, status: str,
                            customer_id: Optional[str], subscription_id: Optional[str],
                            occurred_at: str) -> bool:
        with self.engine.begin() as connection:
            try:
                with connection.begin_nested():
                    connection.execute(insert(billing_events).values(id=event_id, processed_at=utc_now()))
            except IntegrityError:
                return False
            try:
                with connection.begin_nested():
                    connection.execute(insert(billing_state).values(user_id=user_id, event_at=""))
            except IntegrityError:
                pass
            changed = connection.execute(update(billing_state).where(and_(
                billing_state.c.user_id == user_id, billing_state.c.event_at < occurred_at,
            )).values(event_at=occurred_at))
            if changed.rowcount != 1:
                return True
            connection.execute(update(users).where(users.c.id == user_id).values(
                plan=plan, subscription_status=status,
                paddle_customer_id=customer_id,
                paddle_subscription_id=subscription_id,
            ))
        return True
