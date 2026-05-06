import json
import tempfile
import unittest
from pathlib import Path

from app.analytics.highlights_builder import HighlightsBuilder
from app.analytics.objects_stats_builder import ObjectsStatsBuilder
from app.analytics.stats_builder import StatsBuilder


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


class AnalyticsBuilderTests(unittest.TestCase):
    def test_stats_builder_finds_windows_and_dynamics(self):
        with tempfile.TemporaryDirectory() as tmp:
            timeline_path = Path(tmp) / "timeline.jsonl"
            events_path = Path(tmp) / "events.jsonl"

            timeline_rows = [
                {"frame": 0, "time_sec": 0.0, "people": [{"track_id": 1}]},
                {"frame": 1, "time_sec": 0.5, "people": [{"track_id": i} for i in range(1, 5)]},
                {"frame": 2, "time_sec": 1.0, "people": [{"track_id": i} for i in range(1, 6)]},
                {"frame": 3, "time_sec": 1.5, "people": [{"track_id": 1}, {"track_id": 2}]},
            ]
            event_rows = [
                {"frame": 1, "time_sec": 0.5, "events": [{"type": "person_entered"}]},
                {
                    "frame": 2,
                    "time_sec": 1.0,
                    "events": [{"type": "person_entered"}, {"type": "person_entered"}],
                },
                {"frame": 3, "time_sec": 1.5, "events": [{"type": "person_exited"}]},
            ]
            _write_jsonl(timeline_path, timeline_rows)
            _write_jsonl(events_path, event_rows)

            builder = StatsBuilder(
                crowd_threshold=3,
                crowd_smoothing_sec=0.1,
                min_window_sec=0.4,
                dynamics_window_sec=0.5,
            )
            result = builder.build_from_timeline_jsonl(
                analysis_id="analysis-1",
                timeline_path=timeline_path,
                fps=2.0,
                events_path=events_path,
            )

            self.assertEqual(result.people_count["max"], 5)
            self.assertEqual(result.people_count["max_at"]["frame"], 2)
            self.assertEqual(result.timeline_frames, 4)
            self.assertAlmostEqual(result.duration_sec_est, 1.5)
            self.assertEqual(result.crowd_windows[0]["start_sec"], 0.5)
            self.assertEqual(result.crowd_windows[0]["end_sec"], 1.0)
            self.assertGreater(result.crowd_dynamics["fastest_growth"]["delta"], 0)
            self.assertLess(result.crowd_dynamics["fastest_drop"]["delta"], 0)
            self.assertIsNotNone(result.crowd_dynamics["most_dynamic_window"])

            highlights = HighlightsBuilder().build_from_stats_dict(
                analysis_id="analysis-1",
                stats=result.to_dict(),
            )
            highlight_types = [item["type"] for item in highlights.highlights]
            self.assertIn("fastest_growth", highlight_types)
            self.assertIn("peak_crowd", highlight_types)
            self.assertIn("crowd_window", highlight_types)
            self.assertIn("most_dynamic", highlight_types)
            self.assertEqual(highlight_types[0], "fastest_growth")

    def test_objects_stats_builder_uses_majority_vote_per_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            objects_path = Path(tmp) / "objects.jsonl"
            rows = [
                {
                    "frame": 0,
                    "time_sec": 0.0,
                    "objects": [
                        {"track_id": 10, "class_name": "car"},
                        {"track_id": 11, "class_name": "bus"},
                    ],
                },
                {
                    "frame": 1,
                    "time_sec": 0.5,
                    "objects": [
                        {"track_id": 10, "class_name": "car"},
                        {"track_id": 11, "class_name": "bus"},
                    ],
                },
                {
                    "frame": 2,
                    "time_sec": 1.0,
                    "objects": [
                        {"track_id": 10, "class_name": "truck"},
                        {"track_id": 11, "class_name": "bus"},
                    ],
                },
            ]
            _write_jsonl(objects_path, rows)

            result = ObjectsStatsBuilder().build_from_objects_jsonl(
                analysis_id="analysis-2",
                objects_jsonl_path=str(objects_path),
            )

            self.assertEqual(result["objects_total"], 6)
            self.assertEqual(result["objects_frames"], 3)
            self.assertEqual(result["unique_total"], 2)
            self.assertEqual(result["unique_by_class"]["car"], 1)
            self.assertEqual(result["unique_by_class"]["bus"], 1)
            self.assertTrue(result["has_object_track_ids"])
            self.assertEqual(result["top_classes"][0]["class_name"], "bus")


if __name__ == "__main__":
    unittest.main()
