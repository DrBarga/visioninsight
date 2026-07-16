from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


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
    bbox: Tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float

    hits: int = 1
    age: int = 1
    missed: int = 0

    first_frame: int = 0
    last_frame: int = 0

    iou_sum: float = 0.0
    iou_count: int = 0


class IOUTracker:
    """
    Lightweight IOU tracker.

    Important output semantics:
      - visible=True: matched to a detector result on the current analyzed frame.
      - visible=False: retained temporarily by the tracker, but not observed now.
      - track_state=confirmed: track has reached min_hits.

    Downstream on-screen statistics should use confirmed + visible tracks only.
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
        self.reset()

    def reset(self) -> None:
        """Clear all state so a tracker can never leak tracks into another video."""
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.short_tracks_filtered = 0

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, frame_id: int, detections: List[dict]) -> List[dict]:
        for track in self.tracks.values():
            track.age += 1
            track.missed += 1

        unmatched_det = set(range(len(detections)))
        matches: List[Tuple[int, int, float]] = []

        candidates: List[Tuple[float, int, int]] = []
        for tid, track in self.tracks.items():
            for det_index, detection in enumerate(detections):
                if det_index not in unmatched_det:
                    continue
                if self.match_by_class and int(detection.get("class_id", -1)) != track.class_id:
                    continue

                detection_bbox = tuple(detection.get("bbox", [0, 0, 0, 0]))
                overlap = _iou_xyxy(track.bbox, detection_bbox)
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, tid, det_index))

        candidates.sort(reverse=True, key=lambda item: item[0])
        used_tracks = set()
        used_detections = set()

        for overlap, tid, det_index in candidates:
            if tid in used_tracks or det_index in used_detections:
                continue
            used_tracks.add(tid)
            used_detections.add(det_index)
            unmatched_det.discard(det_index)
            matches.append((tid, det_index, overlap))

        for tid, det_index, overlap in matches:
            track = self.tracks[tid]
            detection = detections[det_index]
            x1, y1, x2, y2 = tuple(detection["bbox"])
            px1, py1, px2, py2 = track.bbox

            track.bbox = (
                _smooth(px1, x1, self.smooth_alpha),
                _smooth(py1, y1, self.smooth_alpha),
                _smooth(px2, x2, self.smooth_alpha),
                _smooth(py2, y2, self.smooth_alpha),
            )
            track.class_id = int(detection.get("class_id", track.class_id))
            track.class_name = str(detection.get("class_name", track.class_name))
            track.confidence = float(detection.get("confidence", track.confidence))
            track.hits += 1
            track.missed = 0
            track.last_frame = frame_id
            track.iou_sum += float(overlap)
            track.iou_count += 1

        for det_index in sorted(unmatched_det):
            detection = detections[det_index]
            tid = self._new_id()
            self.tracks[tid] = Track(
                track_id=tid,
                bbox=tuple(detection["bbox"]),
                class_id=int(detection.get("class_id", 0)),
                class_name=str(detection.get("class_name", "")),
                confidence=float(detection.get("confidence", 0.0)),
                first_frame=frame_id,
                last_frame=frame_id,
            )

        to_delete: List[int] = []
        for tid, track in self.tracks.items():
            if track.missed > self.max_missed:
                if track.hits < self.min_hits:
                    self.short_tracks_filtered += 1
                to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        results: List[dict] = []
        for tid, track in self.tracks.items():
            visible = track.missed == 0
            state = "confirmed" if track.hits >= self.min_hits else "tentative"
            mean_iou = track.iou_sum / track.iou_count if track.iou_count else None
            results.append({
                "track_id": tid,
                "track_state": state,
                "visible": visible,
                "missed_frames": track.missed,
                "hits": track.hits,
                "age": track.age,
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "bbox": [float(value) for value in track.bbox],
                "class_id": track.class_id,
                "class_name": track.class_name,
                "confidence": track.confidence,
                "mean_match_iou": round(mean_iou, 4) if mean_iou is not None else None,
            })

        return results
