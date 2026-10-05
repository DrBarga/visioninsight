import tempfile
import unittest
import os
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.jobs.worker import AnalysisWorker
from app.main import create_app
from app.video.processor import VideoProcessor
from test_processor_e2e import FakeDetector, make_video


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        settings = replace(
            Settings.from_env(),
            data_dir=root / "data",
            runs_dir=root / "runs",
            database_url="sqlite:///" + (root / "data" / "test.sqlite3").as_posix(),
            embedded_worker=False,
        )
        self.app = create_app(settings)
        self.client_manager = TestClient(self.app)
        self.client = self.client_manager.__enter__()
        self.video = root / "video.mp4"
        make_video(self.video)

    def tearDown(self):
        self.client_manager.__exit__(None, None, None)
        self.temp.cleanup()

    def register(self, email="owner@example.com"):
        response = self.client.post("/v1/auth/register", json={
            "email": email, "password": "strong-password-123",
        })
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def upload(self, csrf):
        return self.client.post(
            "/v1/analyses?recipe=quick_scan",
            headers={"x-csrf-token": csrf, "content-type": "video/mp4"},
            content=self.video.read_bytes(),
        )

    def test_health_and_private_artifacts(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/v1/analyses").status_code, 401)
        self.assertEqual(self.client.get("/analysis/test/summary").status_code, 404)
        owner = self.register()
        self.assertEqual(self.client.post("/v1/auth/logout").status_code, 403)
        queued = self.upload(owner["csrf_token"])
        self.assertEqual(queued.status_code, 202, queued.text)
        analysis_id = queued.json()["analysis_id"]
        self.assertEqual(self.client.get(f"/v1/analyses/{analysis_id}/overview").status_code, 409)

        with patch("app.video.processor.YOLODetector", FakeDetector):
            processor = VideoProcessor(runs_dir=str(self.app.state.settings.runs_dir))
        worker = AnalysisWorker(self.app.state.store, self.app.state.settings, processor)
        self.assertTrue(worker.run_once())
        self.assertEqual(self.client.get(f"/v1/analyses/{analysis_id}").json()["status"], "completed")
        overview = self.client.get(f"/v1/analyses/{analysis_id}/overview")
        self.assertEqual(overview.status_code, 200, overview.text)
        self.assertEqual(overview.json()["schema_version"], "1.0")
        self.assertEqual(self.client.get(f"/v1/analyses/{analysis_id}/report").status_code, 200)
        video = self.client.get(f"/v1/analyses/{analysis_id}/video")
        self.assertEqual(video.status_code, 200)
        self.assertTrue(video.headers["content-disposition"].startswith("inline;"))
        self.assertEqual(video.headers["content-type"], "video/mp4")
        clip = self.client.get(f"/v1/analyses/{analysis_id}/clip?start_sec=0&end_sec=0.5")
        self.assertEqual(clip.status_code, 200, clip.text)
        self.assertGreater(len(clip.content), 0)
        original_settings = self.app.state.settings
        try:
            self.app.state.settings = replace(original_settings, max_derived_bytes=1)
            limited = self.client.get(f"/v1/analyses/{analysis_id}/clip?start_sec=0.2&end_sec=0.7")
            self.assertEqual(limited.status_code, 507, limited.text)
        finally:
            self.app.state.settings = original_settings
        first_moment = self.client.get(f"/v1/analyses/{analysis_id}/moments").json()["moments"][0]
        image = self.client.get(
            f"/v1/analyses/{analysis_id}/evidence/{first_moment['evidence_ids'][0]}/image"
        )
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.headers["content-type"], "image/jpeg")
        answer = self.client.post(f"/v1/analyses/{analysis_id}/query",
                                  headers={"x-csrf-token": owner["csrf_token"]},
                                  json={"question": "How many people?"})
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertTrue(answer.json()["evidence"])

        self.client.post("/v1/auth/logout", headers={"x-csrf-token": owner["csrf_token"]})
        second = self.register("other@example.com")
        self.assertEqual(self.client.get(f"/v1/analyses/{analysis_id}/overview").status_code, 404)
        self.assertEqual(self.client.get(f"/v1/analyses/{analysis_id}/video").status_code, 404)
        self.assertEqual(self.client.get(f"/v1/analyses/{analysis_id}/artifacts/summary.json").status_code, 404)
        self.assertEqual(self.client.delete(f"/v1/analyses/{analysis_id}",
                                            headers={"x-csrf-token": second["csrf_token"]}).status_code, 404)

    def test_cancel_releases_reservation(self):
        owner = self.register()
        queued = self.upload(owner["csrf_token"])
        self.assertEqual(queued.status_code, 202, queued.text)
        analysis_id = queued.json()["analysis_id"]
        recipe_options = self.app.state.store.get_analysis(analysis_id)["options"]
        self.assertEqual(recipe_options["detection_profile"], "balanced")
        self.assertTrue(recipe_options["include_objects"])
        usage = self.client.get("/v1/account/usage").json()
        self.assertEqual(usage["reserved_minutes"], 1)
        self.assertGreater(usage["stored_input_bytes"], 0)
        cancelled = self.client.post(f"/v1/analyses/{analysis_id}/cancel",
                                     headers={"x-csrf-token": owner["csrf_token"]})
        self.assertEqual(cancelled.status_code, 200)
        self.assertEqual(cancelled.json()["status"], "cancelled")
        self.assertEqual(self.client.get("/v1/account/usage").json()["reserved_minutes"], 0)
        with patch.object(self.app.state.storage, "delete", side_effect=OSError("Disk unavailable")):
            with self.assertRaises(OSError):
                self.client.delete(f"/v1/analyses/{analysis_id}",
                                   headers={"x-csrf-token": owner["csrf_token"]})
        self.assertEqual(self.client.get(f"/v1/analyses/{analysis_id}").status_code, 200)
        self.assertEqual(self.client.delete(f"/v1/analyses/{analysis_id}",
                                            headers={"x-csrf-token": owner["csrf_token"]}).status_code, 200)
        self.assertEqual(self.client.get("/v1/account/usage").json()["stored_input_bytes"], 0)

    def test_production_upload_requires_model_license_clearance(self):
        owner = self.register()
        original = self.app.state.settings
        self.app.state.settings = replace(original, environment="production", model_license_cleared=False)
        try:
            denied = self.upload(owner["csrf_token"])
            self.assertEqual(denied.status_code, 503)
            self.assertEqual(self.client.get("/v1/analyses").json()["analyses"], [])
        finally:
            self.app.state.settings = original

    def test_event_pulse_uses_video_minutes_without_hidden_multiplier(self):
        owner = self.register()
        with patch("app.api.v1.analyses._probe_video", return_value=(90, 5)):
            response = self.client.post(
                "/v1/analyses?recipe=event_pulse",
                headers={"x-csrf-token": owner["csrf_token"], "content-type": "video/mp4"},
                content=self.video.read_bytes(),
            )
        self.assertEqual(response.status_code, 202, response.text)
        usage = self.client.get("/v1/account/usage").json()
        self.assertEqual(usage["reserved_minutes"], 2)
        options = self.app.state.store.get_analysis(response.json()["analysis_id"])["options"]
        self.assertLess(options["max_source_frames"], 520)

    def test_production_config_requires_legal_pages(self):
        with patch.dict(os.environ, {
            "VISIONINSIGHT_ENV": "production",
            "VISIONINSIGHT_PUBLIC_BASE_URL": "https://example.com",
            "VISIONINSIGHT_COOKIE_SECURE": "true",
            "VISIONINSIGHT_DATABASE_URL": "postgresql+psycopg://user:pass@localhost/db",
            "VISIONINSIGHT_TERMS_URL": "",
            "VISIONINSIGHT_PRIVACY_URL": "",
            "VISIONINSIGHT_BUSINESS_EMAIL": "",
        }):
            with self.assertRaisesRegex(ValueError, "legal pages"):
                Settings.from_env()

    def test_request_and_video_limits(self):
        large_json = self.client.post("/v1/auth/login", content=b"{" + b" " * 70000 + b"}",
                                      headers={"content-type": "application/json"})
        self.assertEqual(large_json.status_code, 413)
        self.assertEqual(self.client.post("/v1/analyses", content=self.video.read_bytes(),
                                          headers={"content-type": "video/mp4"}).status_code, 401)
        owner = self.register()
        original = self.app.state.settings
        try:
            self.app.state.settings = replace(original, max_upload_bytes=1)
            self.assertEqual(self.upload(owner["csrf_token"]).status_code, 413)
            self.app.state.settings = replace(original, max_video_pixels=1)
            self.assertEqual(self.upload(owner["csrf_token"]).status_code, 413)
        finally:
            self.app.state.settings = original

    def test_webhook_rejects_oversized_body(self):
        response = self.client.post("/v1/billing/webhooks/paddle", content=b"x" * (1024 * 1024 + 1),
                                    headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 413)

    def test_regions_are_saved_with_the_video_job(self):
        owner = self.register()
        zones = [{"id": "entry", "points": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.8]]}]
        lines = [{"id": "gate", "start": [0.5, 0.1], "end": [0.5, 0.9]}]
        response = self.client.post(
            "/v1/analyses", params={"recipe": "event_pulse", "zones": json.dumps(zones),
                                    "lines": json.dumps(lines)},
            headers={"x-csrf-token": owner["csrf_token"], "content-type": "video/mp4"},
            content=self.video.read_bytes(),
        )
        self.assertEqual(response.status_code, 202, response.text)
        options = self.app.state.store.get_analysis(response.json()["analysis_id"])["options"]
        self.assertEqual(options["regions"]["zones"][0]["id"], "entry")
        self.assertEqual(options["regions"]["lines"][0]["id"], "gate")


if __name__ == "__main__":
    unittest.main()
