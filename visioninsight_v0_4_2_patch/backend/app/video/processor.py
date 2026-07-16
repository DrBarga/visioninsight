from __future__ import annotations

import json
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from app.analytics.highlights_builder import HighlightsBuilder
from app.analytics.object_refinement_builder import ObjectRefinementBuilder
from app.analytics.objects_refined_stats_builder import ObjectsRefinedStatsBuilder
from app.analytics.objects_stats_builder import ObjectsStatsBuilder
from app.analytics.quality_builder import TrackingQualityBuilder
from app.analytics.stats_builder import StatsBuilder
from app.analytics.transcript_builder import TranscriptBuilder
from app.detection.yolo import YOLODetector
from app.tracking.iou_tracker import IOUTracker
from app.version import __version__
from app.video.analysis_profiles import ResolvedAnalysisOptions, resolve_analysis_options


def _jsonl_write(file, payload: Dict[str, Any]) -> None:
    file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_person(detection: Dict[str, Any]) -> bool:
    class_name = detection.get("class_name")
    return isinstance(class_name, str) and class_name.lower() == "person"


def _is_visible_confirmed(track: Dict[str, Any]) -> bool:
    return track.get("track_state") == "confirmed" and track.get("visible") is True


class VideoProcessor:
    def __init__(self, runs_dir: str = "runs", model_path: str = "yolov8n.pt"):
        self.runs_dir = Path(runs_dir)
        self.detector = YOLODetector(model_path=model_path, profile="balanced")
        self.transcript_builder = TranscriptBuilder()
        # Keep CLIP loaded after its first use instead of reloading it for every full analysis.
        self.object_refinement_builder = ObjectRefinementBuilder()
        self.objects_refined_stats_builder = ObjectsRefinedStatsBuilder()
        # Shared model objects are protected until processing is moved to dedicated workers.
        self._process_lock = threading.Lock()

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _create_trackers(frame_stride: int) -> Tuple[IOUTracker, IOUTracker]:
        # Preserve approximately the same real-time retention when frames are sampled.
        people_max_missed = max(1, int(math.ceil(30 / frame_stride)))
        objects_max_missed = max(1, int(math.ceil(15 / frame_stride)))
        return (
            IOUTracker(
                iou_threshold=0.3,
                max_missed=people_max_missed,
                min_hits=3,
                smooth_alpha=0.8,
                match_by_class=True,
            ),
            IOUTracker(
                iou_threshold=0.25,
                max_missed=objects_max_missed,
                min_hits=2,
                smooth_alpha=0.8,
                match_by_class=False,
            ),
        )

    def process(
        self,
        input_path: str,
        analysis_id: Optional[str] = None,
        analysis_mode: str = "balanced",
        detection_profile: Optional[str] = None,
        include_objects: Optional[bool] = None,
        enable_transcript: Optional[bool] = None,
        enable_object_refinement: Optional[bool] = None,
        save_output_video: Optional[bool] = None,
        frame_stride: Optional[int] = None,
        transcript_backend: str = "auto",
        transcript_model: str = "base",
        transcript_language: Optional[str] = None,
    ) -> dict:
        with self._process_lock:
            return self._process_locked(
                input_path=input_path,
                analysis_id=analysis_id,
                analysis_mode=analysis_mode,
                detection_profile=detection_profile,
                include_objects=include_objects,
                enable_transcript=enable_transcript,
                enable_object_refinement=enable_object_refinement,
                save_output_video=save_output_video,
                frame_stride=frame_stride,
                transcript_backend=transcript_backend,
                transcript_model=transcript_model,
                transcript_language=transcript_language,
            )

    def _process_locked(
        self,
        input_path: str,
        analysis_id: Optional[str],
        analysis_mode: str,
        detection_profile: Optional[str],
        include_objects: Optional[bool],
        enable_transcript: Optional[bool],
        enable_object_refinement: Optional[bool],
        save_output_video: Optional[bool],
        frame_stride: Optional[int],
        transcript_backend: str,
        transcript_model: str,
        transcript_language: Optional[str],
    ) -> dict:
        total_started = time.perf_counter()
        options: ResolvedAnalysisOptions = resolve_analysis_options(
            mode=analysis_mode,
            detection_profile=detection_profile,
            include_objects=include_objects,
            enable_transcript=enable_transcript,
            enable_object_refinement=enable_object_refinement,
            save_output_video=save_output_video,
            frame_stride=frame_stride,
        )

        analysis_id = analysis_id or str(uuid.uuid4())
        run_dir = self.runs_dir / analysis_id
        self._ensure_dir(run_dir)

        source_path = Path(input_path)
        if not source_path.exists():
            raise RuntimeError(f"Input file not found: {source_path}")

        output_path = run_dir / "output.mp4"
        timeline_path = run_dir / "timeline.jsonl"
        events_path = run_dir / "events.jsonl"
        people_path = run_dir / "people.jsonl"
        objects_path = run_dir / "objects.jsonl"
        meta_path = run_dir / "meta.json"
        stats_path = run_dir / "stats.json"
        highlights_path = run_dir / "highlights.json"
        quality_path = run_dir / "quality.json"
        objects_stats_path = run_dir / "objects_stats.json"
        object_refinements_path = run_dir / "object_refinements.json"
        objects_refined_stats_path = run_dir / "objects_refined_stats.json"
        summary_path = run_dir / "summary.json"
        transcript_path = run_dir / "transcript.jsonl"
        audio_wav_path = run_dir / "audio.wav"

        self.detector.set_profile(options.detection_profile)
        people_tracker, objects_tracker = self._create_trackers(options.frame_stride)

        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise RuntimeError("Cannot open video file")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("Video has invalid dimensions")

        writer = None
        if options.save_output_video:
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                capture.release()
                raise RuntimeError("Cannot create output video")

        meta: Dict[str, Any] = {
            "analysis_id": analysis_id,
            "status": "processing",
            "version": __version__,
            "input_file": source_path.name,
            "width": width,
            "height": height,
            "fps": fps,
            "analysis_fps": round(fps / options.frame_stride, 4),
            "options": options.to_dict(),
            "detector": {
                "type": "YOLOv8",
                "model": getattr(self.detector, "model_name", "unknown"),
                "profile": self.detector.profile_name,
            },
            "tracker": {
                "people": {
                    "type": "IOUTracker",
                    "iou_threshold": people_tracker.iou_threshold,
                    "max_missed": people_tracker.max_missed,
                    "min_hits": people_tracker.min_hits,
                    "match_by_class": True,
                },
                "objects": {
                    "type": "IOUTracker",
                    "iou_threshold": objects_tracker.iou_threshold,
                    "max_missed": objects_tracker.max_missed,
                    "min_hits": objects_tracker.min_hits,
                    "match_by_class": False,
                },
            },
            "transcript": {
                "enabled": options.enable_transcript,
                "backend": transcript_backend,
                "model": transcript_model,
                "language": transcript_language,
            },
        }
        _write_json(meta_path, meta)

        source_frame_id = 0
        analyzed_frames = 0
        seen_people_tracks = set()
        last_seen_confirmed: Dict[int, int] = {}
        exit_threshold_source_frames = max(1, int(round(fps * 2.0)))

        video_loop_started = time.perf_counter()
        try:
            with timeline_path.open("w", encoding="utf-8") as timeline_file, \
                 events_path.open("w", encoding="utf-8") as events_file, \
                 people_path.open("w", encoding="utf-8") as people_file, \
                 objects_path.open("w", encoding="utf-8") as objects_file:

                while True:
                    success, frame = capture.read()
                    if not success:
                        break

                    should_analyze = source_frame_id % options.frame_stride == 0
                    if not should_analyze:
                        if writer is not None:
                            writer.write(frame)
                        source_frame_id += 1
                        continue

                    processed_frame, detections = self.detector.detect_frame(frame)
                    raw_people = [item for item in detections if _is_person(item)]
                    raw_objects = [item for item in detections if not _is_person(item)] if options.include_objects else []

                    people_active = people_tracker.update(source_frame_id, raw_people)
                    objects_active = objects_tracker.update(source_frame_id, raw_objects) if options.include_objects else []

                    people_visible = [track for track in people_active if _is_visible_confirmed(track)]
                    objects_visible = [track for track in objects_active if _is_visible_confirmed(track)]

                    events: List[Dict[str, Any]] = []
                    for track in people_visible:
                        track_id = int(track["track_id"])
                        last_seen_confirmed[track_id] = source_frame_id
                        if track_id not in seen_people_tracks:
                            seen_people_tracks.add(track_id)
                            events.append({"type": "person_entered", "track_id": track_id})

                    for track_id, last_frame in list(last_seen_confirmed.items()):
                        if source_frame_id - last_frame > exit_threshold_source_frames:
                            events.append({"type": "person_exited", "track_id": int(track_id)})
                            del last_seen_confirmed[track_id]

                    time_sec = round(source_frame_id / fps, 2)
                    people_row = {
                        "frame": source_frame_id,
                        "time_sec": time_sec,
                        "people": people_visible,
                    }
                    objects_row = {
                        "frame": source_frame_id,
                        "time_sec": time_sec,
                        "objects": objects_visible,
                    }
                    timeline_row = {
                        "frame": source_frame_id,
                        "time_sec": time_sec,
                        "people": people_visible,
                        "events": events,
                    }

                    # Full sampled time axis: empty frames are intentionally retained.
                    _jsonl_write(people_file, people_row)
                    _jsonl_write(objects_file, objects_row)
                    _jsonl_write(timeline_file, timeline_row)
                    if events:
                        _jsonl_write(events_file, {
                            "frame": source_frame_id,
                            "time_sec": time_sec,
                            "events": events,
                        })

                    analyzed_frames += 1
                    if writer is not None:
                        writer.write(processed_frame)
                    source_frame_id += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        timings: Dict[str, float] = {
            "video_loop_sec": round(time.perf_counter() - video_loop_started, 3),
        }

        analytics_started = time.perf_counter()
        stats_result = StatsBuilder().build_from_timeline_jsonl(
            analysis_id=analysis_id,
            timeline_path=timeline_path,
            fps=fps,
            events_path=events_path,
            frame_stride=options.frame_stride,
        )
        stats_dict = stats_result.to_dict()
        _write_json(stats_path, stats_dict)

        highlights_result = HighlightsBuilder().build_from_stats_dict(
            analysis_id=analysis_id,
            stats=stats_dict,
        )
        _write_json(highlights_path, highlights_result.to_dict())

        quality_result = TrackingQualityBuilder().build_from_people_jsonl(
            analysis_id=analysis_id,
            fps=fps,
            people_jsonl_path=str(people_path),
            frame_stride=options.frame_stride,
        )
        _write_json(quality_path, quality_result.to_dict())

        object_stats = ObjectsStatsBuilder().build_from_objects_jsonl(
            analysis_id=analysis_id,
            objects_jsonl_path=str(objects_path),
        )
        _write_json(objects_stats_path, object_stats)
        timings["analytics_sec"] = round(time.perf_counter() - analytics_started, 3)

        refinement_summary: Dict[str, Any]
        refinement_started = time.perf_counter()
        if options.enable_object_refinement:
            try:
                refinement_summary = self.object_refinement_builder.build(
                    input_video_path=source_path,
                    objects_jsonl_path=objects_path,
                    output_refinements_json_path=object_refinements_path,
                    samples_per_track=3,
                    topk=5,
                )
            except Exception as error:
                refinement_summary = {
                    "available": False,
                    "reason": f"CLIP refinement failed: {type(error).__name__}: {error}",
                }
                _write_json(object_refinements_path, refinement_summary)
        else:
            refinement_summary = {
                "available": False,
                "reason": "disabled by analysis options",
            }
            _write_json(object_refinements_path, refinement_summary)

        refined_stats_result = self.objects_refined_stats_builder.build(
            object_refinements_json_path=object_refinements_path,
            output_stats_path=objects_refined_stats_path,
            top_n=20,
            min_confidence=0.0,
        )
        timings["object_refinement_sec"] = round(time.perf_counter() - refinement_started, 3)

        transcript_info = None
        transcript_started = time.perf_counter()
        if options.enable_transcript:
            transcript_result = self.transcript_builder.build_from_video(
                analysis_id=analysis_id,
                input_video_path=str(source_path),
                transcript_jsonl_path=str(transcript_path),
                extracted_wav_path=str(audio_wav_path),
                backend=transcript_backend,
                model_size=transcript_model,
                language=transcript_language,
            )
            transcript_info = transcript_result.to_dict()
        timings["transcript_sec"] = round(time.perf_counter() - transcript_started, 3)
        timings["total_sec"] = round(time.perf_counter() - total_started, 3)

        summary = {
            "analysis_id": analysis_id,
            "version": __version__,
            "status": "completed",
            "options": options.to_dict(),
            "source": {
                "frames": source_frame_id,
                "analyzed_frames": analyzed_frames,
                "fps": fps,
                "analysis_fps": round(fps / options.frame_stride, 4),
                "duration_sec_est": round(source_frame_id / fps, 2) if fps else 0.0,
            },
            "tracks_summary": {
                "unique_people": len(seen_people_tracks),
                "track_ids": sorted(seen_people_tracks),
            },
            "objects_summary": {
                "unique_total": object_stats.get("unique_total", 0),
                "refinement": refinement_summary,
                "refined_stats": refined_stats_result.to_dict(),
            },
            "timeline_count": analyzed_frames,
            "transcript": transcript_info,
            "timings": timings,
            "artifacts": {
                "run_dir": str(run_dir),
                "input_video": str(source_path),
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
                "object_refinements": str(object_refinements_path),
                "objects_refined_stats": str(objects_refined_stats_path),
                "output_video": str(output_path) if options.save_output_video else None,
                "transcript_jsonl": str(transcript_path) if options.enable_transcript else None,
                "audio_wav": str(audio_wav_path) if options.enable_transcript else None,
            },
        }
        _write_json(summary_path, summary)

        meta["status"] = "completed"
        meta["source_frames"] = source_frame_id
        meta["analyzed_frames"] = analyzed_frames
        meta["timings"] = timings
        _write_json(meta_path, meta)
        return summary
