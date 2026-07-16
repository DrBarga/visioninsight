from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TrackingQualityResult:
    analysis_id: str
    fps: float
    tracks: Dict[str, Dict[str, Any]]
    quality_summary: Dict[str, Any]
    frame_stride: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "fps": self.fps,
            "frame_stride": self.frame_stride,
            "tracks": self.tracks,
            "quality_summary": self.quality_summary,
        }


class TrackingQualityBuilder:
    """Compute tracking-quality metrics from visible confirmed people tracks."""

    def __init__(self, short_track_threshold_sec: float = 0.7):
        self.short_track_threshold_sec = float(short_track_threshold_sec)

    def build_from_people_jsonl(
        self,
        analysis_id: str,
        fps: float,
        people_jsonl_path: Optional[str] = None,
        people_path: Optional[str] = None,
        path: Optional[str] = None,
        frame_stride: int = 1,
        **_kwargs,
    ) -> TrackingQualityResult:
        final_path = people_jsonl_path or people_path or path
        if not final_path:
            raise ValueError("people_jsonl_path (or people_path/path) is required")

        source_fps = float(fps) if fps else 30.0
        stride = max(1, int(frame_stride))
        tracks: Dict[str, Dict[str, Any]] = {}

        def touch_track(track_id: str, frame: int) -> None:
            current = tracks.get(track_id)
            if current is None:
                tracks[track_id] = {
                    "first_frame": frame,
                    "last_frame": frame,
                    "frames_seen": 1,
                }
            else:
                current["last_frame"] = frame
                current["frames_seen"] += 1

        with open(final_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                frame = int(row.get("frame", 0))
                for person in row.get("people") or []:
                    track_id = str(person.get("track_id"))
                    if track_id and track_id != "None":
                        touch_track(track_id, frame)

        total_duration = 0.0
        total_continuity = 0.0
        short_tracks = 0

        for track in tracks.values():
            first_frame = int(track["first_frame"])
            last_frame = int(track["last_frame"])
            frames_seen = int(track["frames_seen"])
            source_span_frames = max(1, last_frame - first_frame + 1)
            expected_observations = max(1, ((last_frame - first_frame) // stride) + 1)

            duration_sec = source_span_frames / source_fps
            continuity = min(1.0, frames_seen / expected_observations)

            track["source_span_frames"] = source_span_frames
            track["expected_observations"] = expected_observations
            # Backward-compatible field name.
            track["span_frames"] = source_span_frames
            track["duration_sec"] = round(duration_sec, 2)
            track["continuity"] = round(continuity, 3)

            total_duration += duration_sec
            total_continuity += continuity
            if duration_sec < self.short_track_threshold_sec:
                short_tracks += 1

        tracks_total = len(tracks)
        average_duration = total_duration / tracks_total if tracks_total else 0.0
        average_continuity = total_continuity / tracks_total if tracks_total else 0.0
        short_percentage = short_tracks / tracks_total * 100.0 if tracks_total else 0.0

        return TrackingQualityResult(
            analysis_id=analysis_id,
            fps=source_fps,
            frame_stride=stride,
            tracks=tracks,
            quality_summary={
                "tracks_total": tracks_total,
                "avg_track_duration_sec": round(average_duration, 2),
                "avg_continuity": round(average_continuity, 3),
                "short_tracks_count": short_tracks,
                "short_tracks_pct": round(short_percentage, 1),
                "short_track_threshold_sec": self.short_track_threshold_sec,
            },
        )
