from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from app.core.config import Settings
from app.db.store import Store
from app.storage.local import LocalStorage
from app.video.processor import AnalysisCancelled, VideoProcessor


logger = logging.getLogger(__name__)


class AnalysisWorker:
    def __init__(self, store: Store, settings: Settings, processor: Optional[VideoProcessor] = None):
        if settings.environment == "production" and not settings.model_license_cleared:
            raise RuntimeError("Model license clearance is required before production processing")
        self.store = store
        self.settings = settings
        self.storage = LocalStorage(settings.runs_dir)
        self.processor = processor

    def run_once(self) -> bool:
        job = self.store.claim_next()
        if job is None:
            return False
        analysis_id = job["id"]
        if self.store.is_cancel_requested(analysis_id):
            self.store.finish(analysis_id, "cancelled")
            return True

        if self.processor is None:
            try:
                self.processor = VideoProcessor(
                    runs_dir=str(self.settings.runs_dir), model_path=self.settings.model_path,
                )
            except Exception:
                logger.exception("Could not initialize video processor")
                self.store.finish(analysis_id, "failed", error_code="MODEL_UNAVAILABLE",
                                  error_message="Video processing model is unavailable")
                return True

        try:
            result = self.processor.process(
                str(self.storage.artifact(analysis_id, "input.mp4")),
                analysis_id=analysis_id,
                progress_callback=lambda stage, progress: self.store.update_progress(analysis_id, stage, progress),
                cancel_check=lambda: self.store.is_cancel_requested(analysis_id),
                **job["options"],
            )
        except AnalysisCancelled:
            self.store.finish(analysis_id, "cancelled")
        except Exception:
            logger.exception("Analysis %s failed", analysis_id)
            self.store.finish(analysis_id, "failed", error_code="PROCESSING_FAILED",
                              error_message="Video analysis failed; see the job status for support")
        else:
            duration = float(result.get("source", {}).get("duration_sec_est") or 0)
            self.store.finish(analysis_id, "completed", duration_sec=duration)
        return True

    def run_forever(self, stop_event: Optional[threading.Event] = None, poll_interval: float = 1.0) -> None:
        stop_event = stop_event or threading.Event()
        next_cleanup = 0.0
        while not stop_event.is_set():
            try:
                if time.monotonic() >= next_cleanup:
                    self.purge_expired()
                    next_cleanup = time.monotonic() + 3600
                if not self.run_once():
                    stop_event.wait(poll_interval)
            except Exception:
                logger.exception("Worker loop error")
                stop_event.wait(poll_interval)

    def purge_expired(self) -> None:
        from app.api.v1.analyses import PLANS

        self.store.purge_expired_tokens_and_counters()
        policy = {name: plan["retention_days"] for name, plan in PLANS.items()}
        for row in self.store.expired_analyses(policy):
            self.storage.delete(row["id"])
            if self.store.delete_analysis(row["user_id"], row["id"]):
                logger.info("Expired analysis %s removed", row["id"])


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    if settings.environment == "production" and not settings.model_license_cleared:
        raise RuntimeError("Model license clearance is required before production processing")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(settings.database_url)
    interrupted = store.recover_interrupted()
    if interrupted:
        logger.warning("Recovered %s interrupted analyses", interrupted)
    worker = AnalysisWorker(store, settings)
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
