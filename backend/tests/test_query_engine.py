import json
import tempfile
import unittest
from pathlib import Path

from app.query.engine import answer_question
from app.query.intents import detect_intent


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class QueryEngineTests(unittest.TestCase):
    def test_detect_intent_covers_main_question_types(self):
        self.assertEqual(detect_intent("How many people?"), "count_people")
        self.assertEqual(detect_intent("When was it crowded?"), "crowd_windows")
        self.assertEqual(detect_intent("Give me highlights"), "highlights")
        self.assertEqual(detect_intent("Summarize the video"), "summarize_video")
        self.assertEqual(detect_intent("What objects were detected?"), "list_objects")

    def test_answer_question_uses_generated_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_json(run_dir / "summary.json", {
                "analysis_id": "run-1",
                "tracks_summary": {"unique_people": 7, "track_ids": [1, 2, 3, 4, 5, 6, 7]},
                "timeline_count": 12,
            })
            _write_json(run_dir / "stats.json", {
                "analysis_id": "run-1",
                "fps": 25.0,
                "duration_sec_est": 18.4,
                "people_count": {
                    "max": 5,
                    "avg": 2.4,
                    "p95": 4,
                    "max_at": {"time_sec": 8.0, "frame": 200},
                },
                "crowd_threshold": 3,
                "crowd_windows": [{"start_sec": 6.0, "end_sec": 10.0, "type": "stable_crowd"}],
                "crowd_dynamics": {
                    "fastest_growth": {
                        "delta": 2.0,
                        "start_sec": 5.0,
                        "end_sec": 6.0,
                        "from": 1.0,
                        "to": 3.0,
                        "window_sec": 1.0,
                    },
                    "fastest_drop": {
                        "delta": -2.0,
                        "start_sec": 10.0,
                        "end_sec": 11.0,
                        "from": 4.0,
                        "to": 2.0,
                        "window_sec": 1.0,
                    },
                    "most_dynamic_window": {
                        "count": 4,
                        "start_sec": 7.0,
                        "end_sec": 8.0,
                        "window_sec": 1.0,
                    },
                },
            })
            _write_json(run_dir / "highlights.json", {
                "analysis_id": "run-1",
                "highlights": [
                    {"type": "peak_crowd", "start_sec": 7.8, "end_sec": 8.2},
                    {"type": "crowd_window", "start_sec": 6.0, "end_sec": 10.0},
                ],
            })
            _write_json(run_dir / "quality.json", {
                "analysis_id": "run-1",
                "quality_summary": {
                    "tracks_total": 7,
                    "avg_track_duration_sec": 3.2,
                    "short_tracks_pct": 14.3,
                    "short_track_threshold_sec": 0.7,
                },
            })
            _write_json(run_dir / "objects_stats.json", {
                "analysis_id": "run-1",
                "unique_total": 3,
                "unique_by_class": {"car": 2, "bus": 1},
            })

            intent, answer, evidence, confidence = answer_question(str(run_dir), "How many people?")
            self.assertEqual(intent, "count_people")
            self.assertIn("7", answer)
            self.assertEqual(evidence["unique_people"], 7)
            self.assertGreaterEqual(confidence, 0.9)

            intent, answer, evidence, confidence = answer_question(str(run_dir), "Give me highlights")
            self.assertEqual(intent, "highlights")
            self.assertIn("peak_crowd", answer)
            self.assertEqual(evidence["count"], 2)
            self.assertGreaterEqual(confidence, 0.9)

            intent, answer, evidence, confidence = answer_question(str(run_dir), "How many objects?")
            self.assertEqual(intent, "count_objects")
            self.assertIn("3", answer)
            self.assertEqual(evidence["unique_total"], 3)
            self.assertGreaterEqual(confidence, 0.9)

            intent, answer, evidence, confidence = answer_question(str(run_dir), "What objects were detected?")
            self.assertEqual(intent, "list_objects")
            self.assertIn("car=2", answer)
            self.assertIn("bus=1", answer)
            self.assertGreaterEqual(confidence, 0.9)

            intent, answer, evidence, confidence = answer_question(str(run_dir), "Summarize the video")
            self.assertEqual(intent, "summarize_video")
            self.assertIn("Transcript not found", answer)
            self.assertFalse(evidence["found"])
            self.assertGreaterEqual(confidence, 0.45)


if __name__ == "__main__":
    unittest.main()
