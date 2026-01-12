from ultralytics import YOLO
import cv2


class YOLODetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect_frame(self, frame):
        results = self.model(frame, verbose=False)[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "class_id": cls_id,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2]
                })

                # Важно: rectangle, а не rectange
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame, detections
