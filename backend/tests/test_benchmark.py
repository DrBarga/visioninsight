import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_benchmark import evaluate


class BenchmarkTests(unittest.TestCase):
    def test_labeled_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name, value in {
                "stats.json": {"people_count": {"max": 4}},
                "zones.json": {"line_crossings": [{"line_id": "door", "direction": "in", "time_sec": 2.1}]},
                "moments.json": {"moments": [{"start_sec": 1.5, "end_sec": 2.5}]},
                "summary.json": {"source": {"duration_sec_est": 10}, "timings": {"total_sec": 5}},
            }.items():
                (directory / name).write_text(json.dumps(value), encoding="utf-8")
            report = evaluate({"cases": [{"name": "sample", "run_dir": str(directory), "truth": {
                "peak_visible_people": 3,
                "line_crossings": [{"line_id": "door", "direction": "in", "time_sec": 2.0}],
                "notable_intervals": [[2.0, 3.0]],
            }}]})
            self.assertEqual(report["peak_people_mae"], 1)
            self.assertEqual(report["line_crossing_f1"], 1)
            self.assertEqual(report["notable_interval_recall"], 1)


if __name__ == "__main__":
    unittest.main()
