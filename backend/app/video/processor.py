from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

import cv2

from app.detection.yolo import YOLODetector
from app.tracking.iou_tracker import IOUTracker

from app.analytics.stats_builder import StatsBuilder
from app.analytics.highlights_builder import HighlightsBuilder
from app.analytics.quality_builder import TrackingQualityBuilder


class VideoProcessor:
    def __init__(self, runs_dir: str = "runs"):
        self.detector = YOLODetector()
        self.tracker = IOUTracker(iou_threshold=0.3, max_missed=30)
        self.runs_dir = Path(runs_dir)

    def _ensure_dir(self, p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    def process(self, input_path: str, analysis_id: Optional[str] = None) -> dict:
        analysis_id = analysis_id or str(uuid.uuid4())
        run_dir = self.runs_dir / analysis_id
        self._ensure_dir(run_dir)

        # Copy input to run dir
        input_src = Path(input_path)
        input_dst = run_dir / "input.mp4"
        if input_src.resolve() != input_dst.resolve():
            shutil.copyfile(str(input_src), str(input_dst))

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
            (width, height)
        )

        # Paths
        timeline_path = run_dir / "timeline.jsonl"
        events_path = run_dir / "events.jsonl"
        people_path = run_dir / "people.jsonl"

        meta_path = run_dir / "meta.json"
        stats_path = run_dir / "stats.json"
        highlights_path = run_dir / "highlights.json"
        quality_path = run_dir / "quality.json"
        summary_path = run_dir / "summary.json"

        # Meta
        meta = {
            "analysis_id": analysis_id,
            "input_file": "input.mp4",
            "output_file": "output.mp4",
            "width": width,
            "height": height,
            "fps": fps,
            "tracker": {"type": "IOUTracker", "iou_threshold": 0.3, "max_missed": 30},
            "detector": {"type": "YOLOv8", "model": "yolov8n.pt"},
            "version": "0.3.8",
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # Tracking bookkeeping
        frame_id = 0
        timeline_count = 0
        seen_tracks = set()
        last_seen = {}
        exit_threshold = int(fps * 2)

        # 1) Write JSONL artifacts FIRST
        with timeline_path.open("w", encoding="utf-8") as f_tl, \
             events_path.open("w", encoding="utf-8") as f_ev, \
             people_path.open("w", encoding="utf-8") as f_people:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, detections = self.detector.detect_frame(frame)

                people = [d for d in detections if d.get("class_id") == 0]
                people = self.tracker.update(frame_id, people)

                events = []
                for d in people:
                    tid = d.get("track_id")
                    if tid is None:
                        continue

                    last_seen[tid] = frame_id
                    if tid not in seen_tracks:
                        seen_tracks.add(tid)
                        events.append({"type": "person_entered", "track_id": tid})

                for tid, last_fr in list(last_seen.items()):
                    if frame_id - last_fr > exit_threshold:
                        events.append({"type": "person_exited", "track_id": tid})
                        del last_seen[tid]

                if people or events:
                    time_sec = round(frame_id / fps, 2)

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

        # 2) Build stats (IMPORTANT: pass Path, not str!)
        stats_builder = StatsBuilder(
            high_density_threshold=10,
            min_window_sec=0.7,
            crowd_smoothing_sec=0.5
        )

        stats = stats_builder.build_from_timeline_jsonl(
            analysis_id=analysis_id,
            timeline_path=timeline_path,
            fps=fps,
            events_path=events_path
        )

        stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else stats
        if not isinstance(stats_dict, dict):
            stats_dict = {}

        stats_path.write_text(json.dumps(stats_dict, ensure_ascii=False, indent=2), encoding="utf-8")

        # 3) Build highlights
        highlights_builder = HighlightsBuilder()
        highlights = highlights_builder.build_from_stats_dict(analysis_id=analysis_id, stats=stats_dict)
        highlights_path.write_text(json.dumps(highlights.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        # 4) Build tracking quality
        quality_builder = TrackingQualityBuilder(short_track_threshold_sec=0.7)
        quality = quality_builder.build_from_people_jsonl(
            analysis_id=analysis_id,
            fps=fps,
            people_jsonl_path=str(people_path)
        )
        quality_path.write_text(json.dumps(quality.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        # 5) Summary
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
                "highlights": str(highlights_path),
                "timeline_jsonl": str(timeline_path),
                "events_jsonl": str(events_path),
                "people_jsonl": str(people_path),
                "output_video": str(output_dst),
                "quality": str(quality_path),
            }
        }

        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
