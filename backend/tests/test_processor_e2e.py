import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.video.processor import VideoProcessor


class FakeDetector:
    model_name = "test-detector"
    profile_name = "balanced"

    def __init__(self, *args, **kwargs):
        pass

    def set_profile(self, name):
        self.profile_name = name

    def detect_frame(self, frame):
        return frame, [{
            "bbox": [8, 8, 30, 30],
            "confidence": 0.95,
            "class_id": 0,
            "class_name": "person",
        }]


class FailingDetector(FakeDetector):
    def detect_frame(self, frame):
        raise RuntimeError("detector failed")


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (48, 48))
    if not writer.isOpened():
        raise RuntimeError("test video writer unavailable")
    try:
        for _ in range(5):
            writer.write(np.zeros((48, 48, 3), dtype=np.uint8))
    finally:
        writer.release()


class ProcessorE2ETests(unittest.TestCase):
    def test_sequential_analyses_isolate_tracking_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "test.mp4"
            make_video(video)
            with patch("app.video.processor.YOLODetector", FakeDetector):
                processor = VideoProcessor(runs_dir=str(root / "runs"))

            for run_id in ("first", "second"):
                summary = processor.process(str(video), analysis_id=run_id, analysis_mode="fast")
                self.assertEqual(summary["tracks_summary"]["track_ids"], [1])
                events = (root / "runs" / run_id / "events.jsonl").read_text(encoding="utf-8")
                self.assertIn("track_started", events)
                self.assertNotIn("person_entered", events)
                moments = json.loads((root / "runs" / run_id / "moments.json").read_text(encoding="utf-8"))
                evidence = json.loads((root / "runs" / run_id / "evidence.json").read_text(encoding="utf-8"))
                self.assertEqual(moments["schema_version"], "1.0")
                self.assertTrue(moments["moments"])
                self.assertEqual(moments["moments"][0]["evidence_ids"], [evidence["evidence"][0]["id"]])

    def test_failure_is_persisted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "test.mp4"
            make_video(video)
            with patch("app.video.processor.YOLODetector", FakeDetector):
                processor = VideoProcessor(runs_dir=str(root / "runs"))
            processor.detector = FailingDetector()
            with self.assertRaisesRegex(RuntimeError, "detector failed"):
                processor.process(str(video), analysis_id="failed", analysis_mode="fast")
            meta = json.loads((root / "runs" / "failed" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["status"], "failed")
            self.assertEqual(meta["error"]["code"], "PROCESSING_FAILED")

    def test_processing_budgets_stop_malformed_or_expensive_video(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "test.mp4"
            make_video(video)
            with patch("app.video.processor.YOLODetector", FakeDetector):
                processor = VideoProcessor(runs_dir=str(root / "runs"))
            with self.assertRaisesRegex(RuntimeError, "frame limit"):
                processor.process(str(video), analysis_id="too-many-frames", analysis_mode="fast",
                                  max_source_frames=1)
            with self.assertRaisesRegex(RuntimeError, "output size limit"):
                processor.process(str(video), analysis_id="too-much-output", analysis_mode="fast",
                                  max_derived_bytes=1)
