from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _smooth(prev: float, cur: float, alpha: float) -> float:
    return alpha * prev + (1.0 - alpha) * cur


@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]  # xyxy
    class_id: int
    class_name: str
    confidence: float

    hits: int = 1
    age: int = 1
    missed: int = 0

    first_frame: int = 0
    last_frame: int = 0

    # diagnostics
    iou_sum: float = 0.0
    iou_count: int = 0


class IOUTracker:
    """
    Simple IOU tracker with:
      - tentative -> confirmed tracks (min_hits)
      - bbox smoothing
      - optional class-aware matching (match_by_class=True)
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 30,
        min_hits: int = 3,
        smooth_alpha: float = 0.8,
        match_by_class: bool = True,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.min_hits = int(min_hits)
        self.smooth_alpha = float(smooth_alpha)
        self.match_by_class = bool(match_by_class)

        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

        # quality counters
        self.short_tracks_filtered = 0

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, frame_id: int, detections: List[dict]) -> List[dict]:
        """
        detections: list of dict with at least:
          - bbox: [x1,y1,x2,y2]
          - class_id
          - class_name
          - confidence

        returns list of active tracks as dicts with:
          - track_id
          - track_state: "tentative" | "confirmed"
          - bbox
          - class_id
          - class_name
          - confidence
        """
        # 1) mark all tracks missed initially; will reset when matched
        for t in self.tracks.values():
            t.age += 1
            t.missed += 1

        unmatched_det = set(range(len(detections)))
        matches: List[Tuple[int, int, float]] = []  # (track_id, det_idx, iou)

        # Build candidate pairs
        candidates: List[Tuple[float, int, int]] = []
        for tid, tr in self.tracks.items():
            tb = tr.bbox
            for di, d in enumerate(detections):
                if di not in unmatched_det:
                    continue
                if self.match_by_class:
                    if int(d.get("class_id", -1)) != tr.class_id:
                        continue
                db = tuple(d.get("bbox", [0, 0, 0, 0]))
                i = _iou_xyxy(tb, db)
                if i >= self.iou_threshold:
                    candidates.append((i, tid, di))

        # sort by best IOU first (greedy)
        candidates.sort(reverse=True, key=lambda x: x[0])

        used_t = set()
        used_d = set()
        for i, tid, di in candidates:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            matches.append((tid, di, i))
            unmatched_det.discard(di)

        # 3) apply matches
        for tid, di, iou in matches:
            tr = self.tracks[tid]
            d = detections[di]
            db = tuple(d["bbox"])

            px1, py1, px2, py2 = tr.bbox
            x1, y1, x2, y2 = db
            sb = (
                _smooth(px1, x1, self.smooth_alpha),
                _smooth(py1, y1, self.smooth_alpha),
                _smooth(px2, x2, self.smooth_alpha),
                _smooth(py2, y2, self.smooth_alpha),
            )

            tr.bbox = sb
            tr.class_id = int(d.get("class_id", tr.class_id))
            tr.class_name = str(d.get("class_name", tr.class_name))
            tr.confidence = float(d.get("confidence", tr.confidence))
            tr.hits += 1
            tr.missed = 0
            tr.last_frame = frame_id
            tr.iou_sum += float(iou)
            tr.iou_count += 1

        # 4) create new tracks for unmatched detections
        for di in sorted(list(unmatched_det)):
            d = detections[di]
            tid = self._new_id()
            db = tuple(d["bbox"])
            tr = Track(
                track_id=tid,
                bbox=db,
                class_id=int(d.get("class_id", 0)),
                class_name=str(d.get("class_name", "")),
                confidence=float(d.get("confidence", 0.0)),
                hits=1,
                age=1,
                missed=0,
                first_frame=frame_id,
                last_frame=frame_id,
            )
            self.tracks[tid] = tr

        # 5) delete old tracks; count filtered shorts
        to_delete = []
        for tid, tr in self.tracks.items():
            if tr.missed > self.max_missed:
                if tr.hits < self.min_hits:
                    self.short_tracks_filtered += 1
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        # 6) return active tracks as dicts
        results: List[dict] = []
        for tid, tr in self.tracks.items():
            state = "confirmed" if tr.hits >= self.min_hits else "tentative"
            results.append({
                "track_id": tid,
                "track_state": state,
                "bbox": [float(x) for x in tr.bbox],
                "class_id": tr.class_id,
                "class_name": tr.class_name,
                "confidence": tr.confidence,
            })

        return results
