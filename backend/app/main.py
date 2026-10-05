from __future__ import annotations

import threading
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.api.v1.analyses import router as analyses_router
from app.api.v1.auth import router as auth_router
from app.api.v1.billing import router as billing_router
from app.core.config import Settings
from app.core.body_limit import RequestBodyLimitMiddleware
from app.db.store import Store
from app.jobs.worker import AnalysisWorker
from app.storage.local import LocalStorage
from app.version import __version__


FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    storage = LocalStorage(settings.runs_dir)
    store = Store(settings.database_url)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        stop_event = threading.Event()
        worker_thread = None
        if settings.embedded_worker:
            store.recover_interrupted()
            worker = AnalysisWorker(store, settings)
            worker_thread = threading.Thread(
                target=worker.run_forever,
                kwargs={"stop_event": stop_event},
                name="visioninsight-worker",
                daemon=True,
            )
            worker_thread.start()
        try:
            yield
        finally:
            stop_event.set()
            if worker_thread is not None:
                worker_thread.join(timeout=10)
            store.engine.dispose()

    application = FastAPI(title="VisionInsight API", version=__version__, lifespan=lifespan)
    application.state.settings = settings
    application.state.store = store
    application.state.storage = storage
    application.add_middleware(RequestBodyLimitMiddleware)
    application.include_router(auth_router)
    application.include_router(analyses_router)
    application.include_router(billing_router)

    @application.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @application.get("/ready")
    def ready():
        with store.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "ready"}

    @application.get("/")
    def home():
        index = FRONTEND_DIR / "index.html"
        return FileResponse(index) if index.is_file() else {"service": "VisionInsight", "api": "/docs"}

    @application.get("/verify")
    def verify_page():
        return home()

    @application.get("/reset")
    def reset_page():
        return home()

    if FRONTEND_DIR.is_dir():
        application.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @application.middleware("http")
    async def security_headers(request, call_next):
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.url.path in ("/", "/verify", "/reset"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; "
                "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
            )
        if settings.environment == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        logger.info("request_id=%s method=%s path=%s status=%s elapsed_ms=%.1f",
                    request_id, request.method, request.url.path, response.status_code,
                    (time.perf_counter() - started) * 1000)
        return response

    return application


app = create_app()
