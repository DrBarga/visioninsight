import json
import tempfile
import unittest
from pathlib import Path

from app.analytics.quality_builder import TrackingQualityBuilder
from app.analytics.stats_builder import StatsBuilder
from app.tracking.iou_tracker import IOUTracker
from app.video.analysis_profiles import resolve_analysis_options


def _person_detection(x1=0, y1=0, x2=100, y2=100):
    return {
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.9,
        "class_id": 0,
        "class_name": "person",
    }


class PipelineStabilizationTests(unittest.TestCase):
    def test_tracker_marks_unmatched_track_as_not_visible(self):
        tracker = IOUTracker(min_hits=1, max_missed=2)
        first = tracker.update(0, [_person_detection()])
        self.assertTrue(first[0]["visible"])
        self.assertEqual(first[0]["track_state"], "confirmed")

        second = tracker.update(1, [])
        self.assertFalse(second[0]["visible"])
        self.assertEqual(second[0]["missed_frames"], 1)

    def test_tracker_reset_prevents_state_leak(self):
        tracker = IOUTracker(min_hits=1)
        first = tracker.update(0, [_person_detection()])
        self.assertEqual(first[0]["track_id"], 1)

        tracker.reset()
        second = tracker.update(0, [_person_detection()])
        self.assertEqual(second[0]["track_id"], 1)
        self.assertEqual(len(tracker.tracks), 1)

    def test_analysis_modes_and_overrides(self):
        fast = resolve_analysis_options(mode="fast")
        self.assertEqual(fast.detection_profile, "people_strict")
        self.assertFalse(fast.enable_transcript)
        self.assertFalse(fast.enable_object_refinement)
        self.assertEqual(fast.frame_stride, 2)

        overridden = resolve_analysis_options(
            mode="fast",
            enable_object_refinement=True,
            include_objects=False,
        )
        self.assertTrue(overridden.enable_object_refinement)
        self.assertTrue(overridden.include_objects)

    def test_stats_use_analysis_fps_for_stride(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            timeline = Path(temp_dir) / "timeline.jsonl"
            events = Path(temp_dir) / "events.jsonl"
            rows = [
                {"frame": 0, "time_sec": 0.0, "people": []},
                {"frame": 2, "time_sec": 0.2, "people": [{"track_id": 1}]},
                {"frame": 4, "time_sec": 0.4, "people": [{"track_id": 1}, {"track_id": 2}]},
            ]
            with timeline.open("w", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row) + "\n")
            events.write_text("", encoding="utf-8")

            result = StatsBuilder(crowd_smoothing_sec=0.2).build_from_timeline_jsonl(
                analysis_id="stride-test",
                timeline_path=timeline,
                fps=10.0,
                events_path=events,
                frame_stride=2,
            )
            payload = result.to_dict()
            self.assertEqual(payload["fps"], 10.0)
            self.assertEqual(payload["analysis_fps"], 5.0)
            self.assertEqual(payload["frame_stride"], 2)
            self.assertEqual(payload["people_count"]["max"], 2)
            self.assertEqual(payload["people_count"]["avg"], 1.0)

    def test_quality_continuity_accounts_for_stride(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            people_path = Path(temp_dir) / "people.jsonl"
            rows = [
                {"frame": 0, "people": [{"track_id": 1}]},
                {"frame": 2, "people": [{"track_id": 1}]},
                {"frame": 4, "people": [{"track_id": 1}]},
            ]
            with people_path.open("w", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row) + "\n")

            result = TrackingQualityBuilder().build_from_people_jsonl(
                analysis_id="quality-test",
                fps=10.0,
                people_jsonl_path=str(people_path),
                frame_stride=2,
            )
            track = result.tracks["1"]
            self.assertEqual(track["expected_observations"], 3)
            self.assertEqual(track["continuity"], 1.0)


if __name__ == "__main__":
    unittest.main()
