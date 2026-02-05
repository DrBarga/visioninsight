from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from app.detection.yolo import YOLODetector
from app.tracking.iou_tracker import IOUTracker

from app.analytics.stats_builder import StatsBuilder
from app.analytics.highlights_builder import HighlightsBuilder
from app.analytics.quality_builder import TrackingQualityBuilder
from app.analytics.objects_stats_builder import ObjectsStatsBuilder
from app.analytics.transcript_builder import TranscriptBuilder


def _jsonl_write(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _is_person(det: Dict[str, Any]) -> bool:
    cname = det.get("class_name")
    return isinstance(cname, str) and cname.lower() == "person"


def _is_confirmed(track: Dict[str, Any]) -> bool:
    return track.get("track_state") == "confirmed"


class VideoProcessor:
    def __init__(self, runs_dir: str = "runs", model_path: str = "yolov8n.pt"):
        self.runs_dir = Path(runs_dir)
        self.detector = YOLODetector(model_path=model_path, profile="balanced")

        # People tracking: class-aware OK
        self.people_tracker = IOUTracker(
            iou_threshold=0.3,
            max_missed=30,
            min_hits=3,
            smooth_alpha=0.8,
            match_by_class=True,
        )

        # Objects tracking: match_by_class=False to prevent class-flip splitting
        self.objects_tracker = IOUTracker(
            iou_threshold=0.25,
            max_missed=15,
            min_hits=2,
            smooth_alpha=0.8,
            match_by_class=False,
        )

        self.transcript_builder = TranscriptBuilder()

    def _ensure_dir(self, p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    def process(
        self,
        input_path: str,
        analysis_id: Optional[str] = None,
        detection_profile: str = "balanced",
        # transcript options (safe defaults)
        enable_transcript: bool = True,
        transcript_backend: str = "auto",   # auto | faster-whisper | none
        transcript_model: str = "base",
        transcript_language: Optional[str] = None,
    ) -> dict:
        analysis_id = analysis_id or str(uuid.uuid4())
        run_dir = self.runs_dir / analysis_id
        self._ensure_dir(run_dir)

        input_path = Path(input_path)
        if not input_path.exists():
            raise RuntimeError(f"Input file not found: {input_path}")

        output_path = run_dir / "output.mp4"

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

        out = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        # Artifacts
        timeline_path = run_dir / "timeline.jsonl"
        events_path = run_dir / "events.jsonl"
        people_path = run_dir / "people.jsonl"
        objects_path = run_dir / "objects.jsonl"

        meta_path = run_dir / "meta.json"
        stats_path = run_dir / "stats.json"
        highlights_path = run_dir / "highlights.json"
        quality_path = run_dir / "quality.json"
        objects_stats_path = run_dir / "objects_stats.json"
        summary_path = run_dir / "summary.json"

        # new artifacts
        transcript_path = run_dir / "transcript.jsonl"
        audio_wav_path = run_dir / "audio.wav"

        # Apply detection profile
        self.detector.set_profile(detection_profile)

        meta = {
            "analysis_id": analysis_id,
            "input_file": str(input_path.name),
            "output_file": "output.mp4",
            "width": width,
            "height": height,
            "fps": fps,
            "detector": {
                "type": "YOLOv8",
                "model": getattr(self.detector, "model_name", "unknown"),
                "profile": getattr(self.detector, "profile_name", str(detection_profile)),
            },
            "tracker": {
                "people": {"type": "IOUTracker", "iou_threshold": 0.3, "max_missed": 30, "min_hits": 3, "match_by_class": True},
                "objects": {"type": "IOUTracker", "iou_threshold": 0.25, "max_missed": 15, "min_hits": 2, "match_by_class": False},
            },
            "transcript": {
                "enabled": enable_transcript,
                "backend": transcript_backend,
                "model": transcript_model,
                "language": transcript_language,
            },
            "version": "0.4.1+transcript-layer",
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        frame_id = 0
        timeline_count = 0

        seen_people_confirmed_tracks = set()
        last_seen_confirmed: Dict[int, int] = {}
        exit_threshold = int(fps * 2)  # 2 seconds

        with timeline_path.open("w", encoding="utf-8") as f_tl, \
             events_path.open("w", encoding="utf-8") as f_ev, \
             people_path.open("w", encoding="utf-8") as f_people, \
             objects_path.open("w", encoding="utf-8") as f_obj:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, detections = self.detector.detect_frame(frame)

                det_people_raw = [d for d in detections if _is_person(d)]
                det_objects_raw = [d for d in detections if not _is_person(d)]

                det_people_tracks = self.people_tracker.update(frame_id, det_people_raw)
                det_objects_tracks = self.objects_tracker.update(frame_id, det_objects_raw)

                events: List[Dict[str, Any]] = []
                for t in det_people_tracks:
                    if not _is_confirmed(t):
                        continue
                    tid = t.get("track_id")
                    if tid is None:
                        continue
                    tid = int(tid)

                    last_seen_confirmed[tid] = frame_id
                    if tid not in seen_people_confirmed_tracks:
                        seen_people_confirmed_tracks.add(tid)
                        events.append({"type": "person_entered", "track_id": tid})

                for tid, last_fr in list(last_seen_confirmed.items()):
                    if frame_id - last_fr > exit_threshold:
                        events.append({"type": "person_exited", "track_id": int(tid)})
                        del last_seen_confirmed[tid]

                time_sec = round(frame_id / fps, 2)

                if det_people_tracks:
                    _jsonl_write(f_people, {"frame": frame_id, "time_sec": time_sec, "people": det_people_tracks})
                if det_objects_tracks:
                    _jsonl_write(f_obj, {"frame": frame_id, "time_sec": time_sec, "objects": det_objects_tracks})
                if events:
                    _jsonl_write(f_ev, {"frame": frame_id, "time_sec": time_sec, "events": events})

                if det_people_tracks or events:
                    _jsonl_write(
                        f_tl,
                        {"frame": frame_id, "time_sec": time_sec, "people": det_people_tracks, "events": events},
                    )
                    timeline_count += 1

                out.write(processed_frame)
                frame_id += 1

        cap.release()
        out.release()

        # -------- Builders (existing) --------
        stats_result = StatsBuilder().build_from_timeline_jsonl(
            analysis_id=analysis_id,
            timeline_path=timeline_path,
            fps=fps,
            events_path=events_path,
        )
        stats_dict = stats_result.to_dict()
        stats_path.write_text(json.dumps(stats_dict, ensure_ascii=False, indent=2), encoding="utf-8")

        highlights_result = HighlightsBuilder().build_from_stats_dict(analysis_id=analysis_id, stats=stats_dict)
        highlights_path.write_text(json.dumps(highlights_result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        quality_result = TrackingQualityBuilder().build_from_people_jsonl(
            analysis_id=analysis_id,
            fps=fps,
            people_jsonl_path=str(people_path),
        )
        quality_path.write_text(json.dumps(quality_result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        obj_stats = ObjectsStatsBuilder().build_from_objects_jsonl(
            analysis_id=analysis_id,
            objects_jsonl_path=str(objects_path),
        )
        objects_stats_path.write_text(json.dumps(obj_stats, ensure_ascii=False, indent=2), encoding="utf-8")

        # -------- Transcript (new, safe) --------
        transcript_info = None
        if enable_transcript:
            tr = self.transcript_builder.build_from_video(
                analysis_id=analysis_id,
                input_video_path=str(input_path),
                transcript_jsonl_path=str(transcript_path),
                extracted_wav_path=str(audio_wav_path),
                backend=transcript_backend,
                model_size=transcript_model,
                language=transcript_language,
            )
            transcript_info = tr.to_dict()

        summary = {
            "analysis_id": analysis_id,
            "tracks_summary": {
                "unique_people": len(seen_people_confirmed_tracks),
                "track_ids": sorted(list(seen_people_confirmed_tracks)),
            },
            "timeline_count": timeline_count,
            "transcript": transcript_info,
            "artifacts": {
                "run_dir": str(run_dir),
                "meta": str(meta_path),
                "summary": str(summary_path),
                "stats": str(stats_path),
                "highlights": str(highlights_path),
                "quality": str(quality_path),
                "timeline_jsonl": str(timeline_path),
                "events_jsonl": str(events_path),
                "people_jsonl": str(people_path),
                "objects_jsonl": str(objects_path),
                "objects_stats": str(objects_stats_path),
                "output_video": str(output_path),
                "transcript_jsonl": str(transcript_path) if enable_transcript else None,
                "audio_wav": str(audio_wav_path) if enable_transcript else None,
            },
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
