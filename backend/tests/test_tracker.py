import unittest

from app.tracking.iou_tracker import IOUTracker, _iou_xyxy


class IOUTrackerTests(unittest.TestCase):
    def test_iou_xyxy_handles_overlap_and_zero_union(self):
        self.assertAlmostEqual(
            _iou_xyxy((0, 0, 10, 10), (5, 5, 15, 15)),
            25.0 / 175.0,
        )
        self.assertEqual(_iou_xyxy((0, 0, 0, 0), (0, 0, 0, 0)), 0.0)
        self.assertEqual(_iou_xyxy((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)

    def test_tracker_confirms_track_and_smooths_bbox(self):
        tracker = IOUTracker(
            iou_threshold=0.1,
            max_missed=2,
            min_hits=2,
            smooth_alpha=0.5,
            match_by_class=True,
        )

        first = tracker.update(0, [{
            "bbox": [0, 0, 10, 10],
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.9,
        }])
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["track_state"], "tentative")

        second = tracker.update(1, [{
            "bbox": [2, 2, 12, 12],
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.95,
        }])
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["track_id"], first[0]["track_id"])
        self.assertEqual(second[0]["track_state"], "confirmed")
        self.assertEqual(second[0]["bbox"], [1.0, 1.0, 11.0, 11.0])

    def test_match_by_class_blocks_cross_class_match(self):
        tracker = IOUTracker(
            iou_threshold=0.1,
            max_missed=2,
            min_hits=2,
            smooth_alpha=0.8,
            match_by_class=True,
        )

        tracker.update(0, [{
            "bbox": [0, 0, 10, 10],
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.9,
        }])

        results = tracker.update(1, [{
            "bbox": [0, 0, 10, 10],
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.85,
        }])

        self.assertEqual({item["track_id"] for item in results}, {1, 2})
        self.assertEqual(
            {item["class_name"] for item in results},
            {"person", "car"},
        )

    def test_short_tentative_tracks_are_counted_when_deleted(self):
        tracker = IOUTracker(
            iou_threshold=0.3,
            max_missed=0,
            min_hits=2,
            smooth_alpha=0.8,
            match_by_class=True,
        )

        tracker.update(0, [{
            "bbox": [0, 0, 10, 10],
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.9,
        }])
        results = tracker.update(1, [])

        self.assertEqual(results, [])
        self.assertEqual(tracker.short_tracks_filtered, 1)


if __name__ == "__main__":
    unittest.main()
