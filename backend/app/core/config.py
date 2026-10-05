from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    environment: str
    data_dir: Path
    runs_dir: Path
    model_path: str
    database_url: str
    max_upload_bytes: int
    max_duration_sec: int
    max_video_pixels: int
    max_video_fps: int
    max_video_frames: int
    max_processing_seconds: int
    max_derived_bytes: int
    min_free_disk_bytes: int
    embedded_worker: bool
    cookie_secure: bool
    public_base_url: str
    paddle_api_key: str
    paddle_webhook_secret: str
    paddle_environment: str
    paddle_developer_price_id: str
    paddle_team_price_id: str
    paddle_developer_display_price: str
    paddle_team_display_price: str
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    smtp_from: str
    legal_terms_url: str
    legal_privacy_url: str
    business_contact_email: str
    model_license_cleared: bool

    def billing_ready(self, plan: str) -> bool:
        price = self.paddle_developer_price_id if plan == "developer" else self.paddle_team_price_id if plan == "team" else ""
        display = self.paddle_developer_display_price if plan == "developer" else self.paddle_team_display_price if plan == "team" else ""
        return bool(
            price and display and self.paddle_api_key and self.paddle_webhook_secret and
            self.legal_terms_url.startswith("https://") and
            self.legal_privacy_url.startswith("https://") and
            "@" in self.business_contact_email and
            (self.environment != "production" or
             (self.paddle_environment == "production" and self.model_license_cleared))
        )

    @classmethod
    def from_env(cls) -> "Settings":
        environment = os.getenv("VISIONINSIGHT_ENV", "development").lower()
        data_dir = Path(os.getenv("VISIONINSIGHT_DATA_DIR", str(BACKEND_DIR / "data"))).resolve()
        runs_dir = Path(os.getenv("VISIONINSIGHT_RUNS_DIR", str(BACKEND_DIR / "runs"))).resolve()
        model = os.getenv("VISIONINSIGHT_MODEL_PATH", str(BACKEND_DIR / "yolov8n.pt"))
        database_url = os.getenv("VISIONINSIGHT_DATABASE_URL", "sqlite:///" + (data_dir / "visioninsight.sqlite3").as_posix())
        public_base_url = os.getenv("VISIONINSIGHT_PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
        if environment == "production" and (not public_base_url.startswith("https://") or not _boolean("VISIONINSIGHT_COOKIE_SECURE", True)):
            raise ValueError("Production requires an HTTPS public URL and secure cookies")
        if environment == "production" and not database_url.startswith("postgresql+psycopg://"):
            raise ValueError("Production requires PostgreSQL")
        legal_terms_url = os.getenv("VISIONINSIGHT_TERMS_URL", "")
        legal_privacy_url = os.getenv("VISIONINSIGHT_PRIVACY_URL", "")
        business_contact_email = os.getenv("VISIONINSIGHT_BUSINESS_EMAIL", "")
        if environment == "production" and (
            not legal_terms_url.startswith("https://") or
            not legal_privacy_url.startswith("https://") or
            "@" not in business_contact_email
        ):
            raise ValueError("Production requires HTTPS legal pages and a business contact email")
        return cls(
            environment=environment,
            data_dir=data_dir,
            runs_dir=runs_dir,
            model_path=model,
            database_url=database_url,
            max_upload_bytes=int(os.getenv("VISIONINSIGHT_MAX_UPLOAD_MB", "1000")) * 1024 * 1024,
            max_duration_sec=int(os.getenv("VISIONINSIGHT_MAX_DURATION_SEC", "7200")),
            max_video_pixels=int(os.getenv("VISIONINSIGHT_MAX_VIDEO_PIXELS", "8294400")),
            max_video_fps=int(os.getenv("VISIONINSIGHT_MAX_VIDEO_FPS", "60")),
            max_video_frames=int(os.getenv("VISIONINSIGHT_MAX_VIDEO_FRAMES", "432000")),
            max_processing_seconds=int(os.getenv("VISIONINSIGHT_MAX_PROCESSING_SECONDS", "7200")),
            max_derived_bytes=int(os.getenv("VISIONINSIGHT_MAX_DERIVED_MB", "2048")) * 1024 * 1024,
            min_free_disk_bytes=int(os.getenv("VISIONINSIGHT_MIN_FREE_DISK_MB", "1024")) * 1024 * 1024,
            embedded_worker=_boolean("VISIONINSIGHT_EMBEDDED_WORKER", environment == "development"),
            cookie_secure=_boolean("VISIONINSIGHT_COOKIE_SECURE", environment != "development"),
            public_base_url=public_base_url,
            paddle_api_key=os.getenv("PADDLE_API_KEY", ""),
            paddle_webhook_secret=os.getenv("PADDLE_WEBHOOK_SECRET", ""),
            paddle_environment=os.getenv("PADDLE_ENVIRONMENT", "sandbox"),
            paddle_developer_price_id=os.getenv("PADDLE_DEVELOPER_PRICE_ID", ""),
            paddle_team_price_id=os.getenv("PADDLE_TEAM_PRICE_ID", ""),
            paddle_developer_display_price=os.getenv("PADDLE_DEVELOPER_DISPLAY_PRICE", ""),
            paddle_team_display_price=os.getenv("PADDLE_TEAM_DISPLAY_PRICE", ""),
            smtp_host=os.getenv("VISIONINSIGHT_SMTP_HOST", ""),
            smtp_port=int(os.getenv("VISIONINSIGHT_SMTP_PORT", "465")),
            smtp_user=os.getenv("VISIONINSIGHT_SMTP_USER", ""),
            smtp_password=os.getenv("VISIONINSIGHT_SMTP_PASSWORD", ""),
            smtp_from=os.getenv("VISIONINSIGHT_SMTP_FROM", ""),
            legal_terms_url=legal_terms_url,
            legal_privacy_url=legal_privacy_url,
            business_contact_email=business_contact_email,
            model_license_cleared=_boolean("VISIONINSIGHT_MODEL_LICENSE_CLEARED", False),
        )
