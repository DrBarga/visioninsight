from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
from ultralytics import YOLO


# COCO names in Ultralytics are typically:
# model.names: Dict[int, str]
# We'll rely on that instead of hardcoding.
@dataclass(frozen=True)
class DetectionProfile:
    name: str
    # global thresholds
    conf: float = 0.25
    iou: float = 0.45
    # optional allowlist of class names (lowercase)
    allowlist: Optional[Set[str]] = None
    # optional per-class minimum confidence override (lowercase)
    per_class_conf: Optional[Dict[str, float]] = None


def _lc(s: Optional[str]) -> Optional[str]:
    return s.lower().strip() if isinstance(s, str) else None


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[float, float, float, float]:
    x1 = max(0.0, min(float(x1), float(w)))
    x2 = max(0.0, min(float(x2), float(w)))
    y1 = max(0.0, min(float(y1), float(h)))
    y2 = max(0.0, min(float(y2), float(h)))
    # ensure proper ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


class YOLODetector:
    """
    Ultralytics YOLOv8 wrapper that returns:
      processed_frame, detections

    detection dict shape (stable contract for the pipeline):
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.87,
      "class_id": 0,
      "class_name": "person"
    }
    """

    # Safe defaults (you can tweak later)
    PROFILES: Dict[str, DetectionProfile] = {
        # General-purpose
        "balanced": DetectionProfile(
            name="balanced",
            conf=0.25,
            iou=0.45,
            allowlist=None,
            per_class_conf=None,
        ),
        # Crowd/festival: detect people more aggressively, but only people
        "crowd_people": DetectionProfile(
            name="crowd_people",
            conf=0.18,
            iou=0.50,
            allowlist={"person"},
            per_class_conf={"person": 0.18},
        ),
        # Strict people-only (less false positives)
        "people_strict": DetectionProfile(
            name="people_strict",
            conf=0.30,
            iou=0.45,
            allowlist={"person"},
            per_class_conf={"person": 0.30},
        ),
        # Road/vehicles: keep only vehicle classes to kill umbrellas/traffic lights etc.
        "vehicles": DetectionProfile(
            name="vehicles",
            conf=0.25,
            iou=0.45,
            allowlist={"car", "truck", "bus", "train", "motorcycle", "bicycle"},
            per_class_conf={
                "car": 0.25,
                "truck": 0.25,
                "bus": 0.25,
                "train": 0.25,
                "motorcycle": 0.25,
                "bicycle": 0.25,
            },
        ),
    }

    def __init__(self, model_path: str = "yolov8n.pt", profile: str = "balanced"):
        self.model_name = model_path
        self.model = YOLO(model_path)
        self.profile_name = "balanced"
        self.profile = self.PROFILES["balanced"]
        self.set_profile(profile)

    def set_profile(self, profile: str) -> None:
        p = (profile or "balanced").strip()
        if p not in self.PROFILES:
            p = "balanced"
        self.profile_name = p
        self.profile = self.PROFILES[p]

    def _accept(self, class_name: str, conf: float) -> bool:
        cn = _lc(class_name) or ""
        if self.profile.allowlist is not None and cn not in self.profile.allowlist:
            return False
        if self.profile.per_class_conf and cn in self.profile.per_class_conf:
            return float(conf) >= float(self.profile.per_class_conf[cn])
        return float(conf) >= float(self.profile.conf)

    def detect_frame(self, frame) -> tuple:
        """
        Returns:
          processed_frame: np.ndarray
          detections: List[dict]
        """
        h, w = frame.shape[:2]

        # Run model
        # Note: we pass conf/iou to reduce raw noise at model output level
        results = self.model.predict(
            source=frame,
            conf=float(self.profile.conf),
            iou=float(self.profile.iou),
            verbose=False,
        )

        detections: List[dict] = []
        processed = frame.copy()

        if not results:
            return processed, detections

        r0 = results[0]
        names = getattr(r0, "names", None) or getattr(self.model, "names", {}) or {}

        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return processed, detections

        # boxes.xyxy, boxes.conf, boxes.cls
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls

        for (x1, y1, x2, y2), c, ci in zip(xyxy, confs, clss):
            class_id = int(ci)
            class_name = names.get(class_id, f"class_{class_id}")
            conf = float(c)

            if not self._accept(class_name, conf):
                continue

            x1, y1, x2, y2 = _clip_xyxy(x1, y1, x2, y2, w=w, h=h)

            det = {
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "confidence": round(conf, 4),
                "class_id": class_id,
                "class_name": str(class_name),
            }
            detections.append(det)

            # draw
            cv2.rectangle(processed, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(processed, label, (int(x1), max(0, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return processed, detections
