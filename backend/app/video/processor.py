import os
import json
import uuid
import shutil
from pathlib import Path
import cv2

from app.detection.yolo import YOLODetector
from app.tracking.iou_tracker import IOUTracker


class VideoProcessor:
    def __init__(self, runs_dir: str = "runs"):
        self.detector = YOLODetector()
        self.tracker = IOUTracker(iou_threshold=0.3, max_missed=30)
        self.runs_dir = Path(runs_dir)

    def _ensure_dir(self, p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    def process(self, input_path: str, output_path: str | None = None, analysis_id: str | None = None):
        """
        Process video, write artifacts to runs/<analysis_id>/ and return compact response.
        """
        analysis_id = analysis_id or str(uuid.uuid4())
        run_dir = self.runs_dir / analysis_id
        self._ensure_dir(run_dir)

        # Copy input into run dir for reproducibility
        input_src = Path(input_path)
        input_dst = run_dir / "input.mp4"
        if input_src.resolve() != input_dst.resolve():
            shutil.copyfile(input_path, str(input_dst))

        # Output path inside run dir by default
        output_dst = run_dir / "output.mp4"
        if output_path:
            # If caller provided output_path, we still write to run_dir/output.mp4
            # and caller can use artifacts path from response.
            pass

        cap = cv2.VideoCapture(str(input_dst))
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        out = cv2.VideoWriter(
            str(output_dst),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        # Artifact files
        timeline_path = run_dir / "timeline.jsonl"
        events_path = run_dir / "events.jsonl"
        people_path = run_dir / "people.jsonl"
        meta_path = run_dir / "meta.json"
        summary_path = run_dir / "summary.json"

        # Tracking bookkeeping
        frame_id = 0
        timeline_count = 0
        seen_tracks = set()
        last_seen = {}  # track_id -> last frame index
        exit_threshold = int(fps * 2)  # exit after 2 seconds not seen

        # Write meta upfront
        meta = {
            "analysis_id": analysis_id,
            "input_file": "input.mp4",
            "output_file": "output.mp4",
            "width": width,
            "height": height,
            "fps": fps,
            "tracker": {"type": "IOUTracker", "iou_threshold": 0.3, "max_missed": 30},
            "detector": {"type": "YOLOv8", "model": "yolov8n.pt"},
            "version": "0.2"
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        with open(timeline_path, "w", encoding="utf-8") as f_tl, \
             open(events_path, "w", encoding="utf-8") as f_ev, \
             open(people_path, "w", encoding="utf-8") as f_people:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, detections = self.detector.detect_frame(frame)

                # Only persons (COCO: person=0)
                people = [d for d in detections if d.get("class_id") == 0]

                # Tracking
                people = self.tracker.update(frame_id, people)

                # Events for this frame
                events = []
                for d in people:
                    tid = d["track_id"]
                    last_seen[tid] = frame_id

                    if tid not in seen_tracks:
                        seen_tracks.add(tid)
                        events.append({"type": "person_entered", "track_id": tid})

                # Exit events
                for tid, last_fr in list(last_seen.items()):
                    if frame_id - last_fr > exit_threshold:
                        events.append({"type": "person_exited", "track_id": tid})
                        del last_seen[tid]

                # Write artifacts (only if there is data)
                if people or events:
                    time_sec = round(frame_id / fps, 2)

                    if people:
                        # store people frame line
                        f_people.write(json.dumps({
                            "frame": frame_id,
                            "time_sec": time_sec,
                            "people": people
                        }, ensure_ascii=False) + "\n")

                    if events:
                        # store events frame line
                        f_ev.write(json.dumps({
                            "frame": frame_id,
                            "time_sec": time_sec,
                            "events": events
                        }, ensure_ascii=False) + "\n")

                    # timeline line (combined)
                    f_tl.write(json.dumps({
                        "frame": frame_id,
                        "time_sec": time_sec,
                        "people": people,
                        "events": events
                    }, ensure_ascii=False) + "\n")

                    timeline_count += 1

                out.write(processed_frame)
                frame_id += 1

        cap.release()
        out.release()

        summary = {
            "analysis_id": analysis_id,
            "tracks_summary": {
                "unique_people": len(seen_tracks),
                "track_ids": sorted(list(seen_tracks))
            },
            "timeline_count": timeline_count,
            "artifacts": {
                "run_dir": str(run_dir),
                "meta": str(meta_path),
                "summary": str(summary_path),
                "timeline_jsonl": str(timeline_path),
                "events_jsonl": str(events_path),
                "people_jsonl": str(people_path),
                "output_video": str(output_dst)
            }
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
