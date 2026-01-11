import cv2
from app.detection.yolo import YOLODetector


class VideoProcessor:
    def __init__(self):
        self.detector = YOLODetector()

    def process(self, input_path: str, output_path: str):
        print("PROCESS START")
        print("INPUT:", input_path)
        print("OUTPUT:", output_path)

        cap = cv2.VideoCapture(input_path)
        print("CAP OPENED:", cap.isOpened())


        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        frame_id = 0
        events = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, detections = self.detector.detect_frame(frame)

            if detections:
                events.append({
                    "frame": frame_id,
                    "objects": detections
                })

            out.write(processed_frame)
            frame_id += 1

        cap.release()
        out.release()

        return events
