from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
from ultralytics import YOLO


@dataclass(frozen=True)
class DetectionProfile:
    name: str
    conf: float = 0.25
    iou: float = 0.45
    allowlist: Optional[Set[str]] = None
    per_class_conf: Optional[Dict[str, float]] = None
    yolo_classes: Optional[List[int]] = None


def _lc(value: Optional[str]) -> Optional[str]:
    return value.lower().strip() if isinstance(value, str) else None


def _clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x1 = max(0.0, min(float(x1), float(width)))
    x2 = max(0.0, min(float(x2), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    y2 = max(0.0, min(float(y2), float(height)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


class YOLODetector:
    """Ultralytics YOLOv8 wrapper with stable VisionInsight detections."""

    PROFILES: Dict[str, DetectionProfile] = {
        "balanced": DetectionProfile(
            name="balanced",
            conf=0.25,
            iou=0.45,
        ),
        "crowd_people": DetectionProfile(
            name="crowd_people",
            conf=0.18,
            iou=0.50,
            allowlist={"person"},
            per_class_conf={"person": 0.18},
            yolo_classes=[0],
        ),
        "people_strict": DetectionProfile(
            name="people_strict",
            conf=0.30,
            iou=0.45,
            allowlist={"person"},
            per_class_conf={"person": 0.30},
            yolo_classes=[0],
        ),
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
            # COCO: bicycle=1, car=2, motorcycle=3, bus=5, train=6, truck=7
            yolo_classes=[1, 2, 3, 5, 6, 7],
        ),
    }

    def __init__(self, model_path: str = "yolov8n.pt", profile: str = "balanced"):
        self.model_name = model_path
        self.model = YOLO(model_path)
        self.profile_name = "balanced"
        self.profile = self.PROFILES["balanced"]
        self.set_profile(profile)

    @classmethod
    def profile_names(cls) -> List[str]:
        return sorted(cls.PROFILES)

    def set_profile(self, profile: str) -> None:
        profile_name = (profile or "balanced").strip()
        if profile_name not in self.PROFILES:
            valid = ", ".join(self.profile_names())
            raise ValueError(f"Unknown detection profile '{profile}'. Expected one of: {valid}")
        self.profile_name = profile_name
        self.profile = self.PROFILES[profile_name]

    def _accept(self, class_name: str, confidence: float) -> bool:
        normalized_name = _lc(class_name) or ""
        if self.profile.allowlist is not None and normalized_name not in self.profile.allowlist:
            return False
        if self.profile.per_class_conf and normalized_name in self.profile.per_class_conf:
            return float(confidence) >= float(self.profile.per_class_conf[normalized_name])
        return float(confidence) >= float(self.profile.conf)

    def detect_frame(self, frame) -> tuple:
        height, width = frame.shape[:2]

        predict_kwargs = {
            "source": frame,
            "conf": float(self.profile.conf),
            "iou": float(self.profile.iou),
            "verbose": False,
        }
        if self.profile.yolo_classes is not None:
            # Filtering inside YOLO is faster than discarding classes afterwards.
            predict_kwargs["classes"] = self.profile.yolo_classes

        results = self.model.predict(**predict_kwargs)
        detections: List[dict] = []
        processed = frame.copy()

        if not results:
            return processed, detections

        result = results[0]
        names = getattr(result, "names", None) or getattr(self.model, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return processed, detections

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        confidences = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls

        for (x1, y1, x2, y2), confidence_raw, class_raw in zip(xyxy, confidences, classes):
            class_id = int(class_raw)
            class_name = str(names.get(class_id, f"class_{class_id}"))
            confidence = float(confidence_raw)

            if not self._accept(class_name, confidence):
                continue

            x1, y1, x2, y2 = _clip_xyxy(x1, y1, x2, y2, width=width, height=height)
            detection = {
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "confidence": round(confidence, 4),
                "class_id": class_id,
                "class_name": class_name,
            }
            detections.append(detection)

            cv2.rectangle(processed, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                processed,
                label,
                (int(x1), max(0, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return processed, detections
