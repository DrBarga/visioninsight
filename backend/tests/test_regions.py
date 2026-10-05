import unittest

from app.analytics.regions import RegionAnalyzer, validate_regions


class RegionTests(unittest.TestCase):
    def test_crossing_requires_segment_intersection_and_track_continuity(self):
        analyzer = RegionAnalyzer({
            "zones": [{"id": "door", "points": [[0.3, 0], [0.7, 0], [0.7, 1], [0.3, 1]]}],
            "lines": [{"id": "threshold", "start": [0.5, 0], "end": [0.5, 1]}],
        }, width=100, height=100, fps=10, frame_stride=1)
        left = {"track_id": 1, "bbox": [10, 10, 30, 50]}
        right = {"track_id": 1, "bbox": [70, 10, 90, 50]}
        occupancy, events = analyzer.update(0, 0.0, [left])
        self.assertEqual(occupancy["door"], 0)
        self.assertEqual(events, [])
        occupancy, events = analyzer.update(1, 0.1, [right])
        self.assertEqual(occupancy["door"], 0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["line_id"], "threshold")
        analyzer.update(2, 0.2, [])
        _, events = analyzer.update(3, 0.3, [left])
        self.assertEqual(events, [])

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            validate_regions({"lines": [{"id": "bad", "start": [0.5, 0.5], "end": [0.5, 0.5]}]})
        with self.assertRaises(ValueError):
            validate_regions({"zones": [{"id": "bad", "points": [[0, 0], [2, 0], [1, 1]]}]})


if __name__ == "__main__":
    unittest.main()
