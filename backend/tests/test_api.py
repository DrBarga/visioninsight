import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class _DummyVideoProcessor:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def process(self, *args, **kwargs):
        analysis_id = kwargs.get("analysis_id", "stub-analysis")
        return {"analysis_id": analysis_id, "status": "ok"}


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_processor_module = types.ModuleType("app.video.processor")
        fake_processor_module.VideoProcessor = _DummyVideoProcessor
        sys.modules["app.video.processor"] = fake_processor_module
        sys.modules.pop("app.main", None)

        cls.main = importlib.import_module("app.main")
        cls.client = TestClient(cls.main.app)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_runs_dir = self.main.RUNS_DIR
        self.main.RUNS_DIR = Path(self.tmp.name)

    def tearDown(self):
        self.main.RUNS_DIR = self.original_runs_dir
        self.tmp.cleanup()

    def test_health_and_root_endpoints(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        root = self.client.get("/")
        self.assertEqual(root.status_code, 200)
        self.assertIn("VisionInsight API is running", root.text)

    def test_summary_endpoint_returns_404_and_existing_file(self):
        missing = self.client.get("/analysis/run-404/summary")
        self.assertEqual(missing.status_code, 404)

        run_dir = self.main.RUNS_DIR / "run-1"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"
        summary_path.write_text(
            json.dumps({"analysis_id": "run-1", "ok": True}, ensure_ascii=False),
            encoding="utf-8",
        )

        response = self.client.get("/analysis/run-1/summary")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["analysis_id"], "run-1")

    def test_ask_endpoint_uses_artifacts_from_run_directory(self):
        run_dir = self.main.RUNS_DIR / "run-ask"
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "summary.json").write_text(
            json.dumps({
                "analysis_id": "run-ask",
                "tracks_summary": {"unique_people": 4},
                "timeline_count": 9,
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "stats.json").write_text(
            json.dumps({
                "analysis_id": "run-ask",
                "fps": 25.0,
                "duration_sec_est": 11.5,
                "people_count": {
                    "max": 3,
                    "avg": 1.8,
                    "p95": 3,
                    "max_at": {"time_sec": 4.2, "frame": 105},
                },
                "crowd_threshold": 2,
                "crowd_windows": [],
                "crowd_dynamics": {},
            }, ensure_ascii=False),
            encoding="utf-8",
        )

        response = self.client.post(
            "/analysis/run-ask/ask",
            json={"question": "How many people?"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["intent"], "count_people")
        self.assertIn("4", payload["answer"])
        self.assertEqual(payload["evidence"]["unique_people"], 4)


if __name__ == "__main__":
    unittest.main()
