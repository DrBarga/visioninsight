from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    last_frame: int
    missed: int = 0


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class IOUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 30):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, frame_id: int, detections: List[Dict]) -> List[Dict]:
        """
        detections: [{"bbox":[x1,y1,x2,y2], ...}, ...]
        returns detections enriched with "track_id"
        """
        det_bboxes = [tuple(d["bbox"]) for d in detections]

        # mark all tracks as missed by default
        for t in self.tracks:
            t.missed += 1

        assigned = set()

        for idx, db in enumerate(det_bboxes):
            best_score = 0.0
            best_track = None

            for t in self.tracks:
                if t.track_id in assigned:
                    continue
                score = iou(t.bbox, db)
                if score > best_score:
                    best_score = score
                    best_track = t

            if best_track is not None and best_score >= self.iou_threshold:
                best_track.bbox = db
                best_track.last_frame = frame_id
                best_track.missed = 0
                detections[idx]["track_id"] = best_track.track_id
                assigned.add(best_track.track_id)
            else:
                tid = self._next_id
                self._next_id += 1
                self.tracks.append(Track(track_id=tid, bbox=db, last_frame=frame_id))
                detections[idx]["track_id"] = tid
                assigned.add(tid)

        # drop dead tracks
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        return detections
