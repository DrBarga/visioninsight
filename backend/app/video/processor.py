import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

import cv2

from app.detection.yolo import YOLODetector
from app.tracking.iou_tracker import IOUTracker
from app.analytics.stats_builder import StatsBuilder


class VideoProcessor:
    def __init__(self, runs_dir: str = "runs"):
        self.detector = YOLODetector()
        self.tracker = IOUTracker(iou_threshold=0.3, max_missed=30)
        self.runs_dir = Path(runs_dir)

    def _ensure_dir(self, p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    def process(self, input_path: str, output_path: Optional[str] = None, analysis_id: Optional[str] = None):
        """
        Process video, write artifacts to runs/<analysis_id>/ and return compact summary.
        """
        analysis_id = analysis_id or str(uuid.uuid4())
        run_dir = self.runs_dir / analysis_id
        self._ensure_dir(run_dir)

        # Copy input into run dir
        input_src = Path(input_path)
        input_dst = run_dir / "input.mp4"
        if input_src.resolve() != input_dst.resolve():
            shutil.copyfile(str(input_src), str(input_dst))

        # Output in run dir
        output_dst = run_dir / "output.mp4"

        cap = cv2.VideoCapture(str(input_dst))
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

        out = cv2.VideoWriter(
            str(output_dst),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        # Artifact files
        timeline_path = run_dir / "timeline.jsonl"
        events_path = run_dir / "events.jsonl"
        people_path = run_dir / "people.jsonl"
        meta_path = run_dir / "meta.json"
        summary_path = run_dir / "summary.json"
        stats_path = run_dir / "stats.json"

        # --- Tracking & event intelligence (anti-flicker) ---
        min_presence_sec = 0.3   # must be present for >=0.3s to count as entered
        min_absence_sec = 0.7    # must be absent for >=0.7s to count as exited
        min_presence_frames = max(1, int(fps * min_presence_sec))
        min_absence_frames = max(1, int(fps * min_absence_sec))

        # track_state[tid] = {"first_seen": int, "last_seen": int, "confirmed": bool}
        track_state = {}

        # Bookkeeping
        frame_id = 0
        timeline_count = 0
        seen_tracks = set()

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
            "version": "0.3.7",
            "events_logic": {
                "min_presence_sec": min_presence_sec,
                "min_absence_sec": min_absence_sec
            }
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

                # Keep only persons (COCO: person=0)
                people = [d for d in detections if d.get("class_id") == 0]

                # Tracking (adds track_id)
                people = self.tracker.update(frame_id, people)

                time_sec = round(frame_id / fps, 2)

                # --- Smarter enter/exit events ---
                events = []
                current_ids = set()

                for d in people:
                    tid = d["track_id"]
                    current_ids.add(tid)

                    st = track_state.get(tid)
                    if st is None:
                        track_state[tid] = {"first_seen": frame_id, "last_seen": frame_id, "confirmed": False}
                        st = track_state[tid]
                    else:
                        st["last_seen"] = frame_id

                    # confirm enter only after stable presence
                    if not st["confirmed"] and (frame_id - st["first_seen"]) >= min_presence_frames:
                        st["confirmed"] = True
                        seen_tracks.add(tid)
                        events.append({"type": "person_entered", "track_id": tid})

                # confirm exits only after stable absence (for confirmed tracks)
                for tid, st in list(track_state.items()):
                    if st["confirmed"]:
                        if tid not in current_ids and (frame_id - st["last_seen"]) >= min_absence_frames:
                            events.append({"type": "person_exited", "track_id": tid})
                            del track_state[tid]

                # Write people/events/timeline JSONL
                if people:
                    f_people.write(json.dumps({
                        "frame": frame_id,
                        "time_sec": time_sec,
                        "people": people
                    }, ensure_ascii=False) + "\n")

                if events:
                    f_ev.write(json.dumps({
                        "frame": frame_id,
                        "time_sec": time_sec,
                        "events": events
                    }, ensure_ascii=False) + "\n")

                # Timeline stores both
                if people or events:
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

        # --- Build stats.json (v0.3.5) ---
        stats_builder = StatsBuilder(
            min_window_sec=0.7,
            smoothing_sec=0.5,
            dynamics_window_sec=1.0,
            top_percentile=90
        )

        stats = stats_builder.build(
            analysis_id=analysis_id,
            timeline_path=timeline_path,
            fps=fps,
            events_path=events_path
        )

        stats_path.write_text(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        # Summary
        summary = {
            "analysis_id": analysis_id,
            "tracks_summary": {
                "unique_people": len(seen_tracks),
                "track_ids": sorted(list(seen_tracks)),
            },
            "timeline_count": timeline_count,
            "artifacts": {
                "run_dir": str(run_dir),
                "meta": str(meta_path),
                "summary": str(summary_path),
                "stats": str(stats_path),
                "timeline_jsonl": str(timeline_path),
                "events_jsonl": str(events_path),
                "people_jsonl": str(people_path),
                "output_video": str(output_dst),
            },
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
