from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class TrackingQualityResult:
    analysis_id: str
    fps: float
    tracks: Dict[str, Dict[str, Any]]
    quality_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "fps": self.fps,
            "tracks": self.tracks,
            "quality_summary": self.quality_summary,
        }


class TrackingQualityBuilder:
    """
    Computes basic tracking quality metrics from people.jsonl:
      - per track: first_frame, last_frame, frames_seen, span_frames, duration_sec, continuity
      - summary: total tracks, avg duration, avg continuity, short tracks count/pct
    """

    def __init__(self, short_track_threshold_sec: float = 0.7):
        self.short_track_threshold_sec = float(short_track_threshold_sec)

    def build_from_people_jsonl(
        self,
        analysis_id: str,
        fps: float,
        people_jsonl_path: Optional[str] = None,
        people_path: Optional[str] = None,
        path: Optional[str] = None,
        **kwargs,
    ) -> TrackingQualityResult:
        """
        BRUTE-FORCE COMPATIBILITY:
        Accepts people_jsonl_path OR people_path OR path (any one).
        Also ignores unexpected kwargs safely.
        """
        final_path = people_jsonl_path or people_path or path
        if not final_path:
            raise ValueError("people_jsonl_path (or people_path/path) is required")

        fps = float(fps) if fps else 30.0

        tracks: Dict[str, Dict[str, Any]] = {}

        def _touch_track(tid: str, frame: int):
            t = tracks.get(tid)
            if t is None:
                tracks[tid] = {
                    "first_frame": frame,
                    "last_frame": frame,
                    "frames_seen": 1,
                }
            else:
                t["last_frame"] = frame
                t["frames_seen"] += 1

        with open(final_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                frame = int(row.get("frame", 0))
                people = row.get("people") or []
                for p in people:
                    tid = str(p.get("track_id"))
                    if tid and tid != "None":
                        _touch_track(tid, frame)

        # finalize per-track derived fields
        total_duration = 0.0
        total_cont = 0.0
        short_tracks = 0

        for tid, t in tracks.items():
            first_fr = int(t["first_frame"])
            last_fr = int(t["last_frame"])
            frames_seen = int(t["frames_seen"])
            span_frames = (last_fr - first_fr + 1) if last_fr >= first_fr else frames_seen

            duration_sec = span_frames / fps
            continuity = (frames_seen / span_frames) if span_frames > 0 else 0.0

            t["span_frames"] = span_frames
            t["duration_sec"] = round(duration_sec, 2)
            t["continuity"] = round(continuity, 3)

            total_duration += duration_sec
            total_cont += continuity

            if duration_sec < self.short_track_threshold_sec:
                short_tracks += 1

        n = len(tracks)
        avg_dur = (total_duration / n) if n else 0.0
        avg_cont = (total_cont / n) if n else 0.0
        short_pct = (short_tracks / n * 100.0) if n else 0.0

        summary = {
            "tracks_total": n,
            "avg_track_duration_sec": round(avg_dur, 2),
            "avg_continuity": round(avg_cont, 3),
            "short_tracks_count": short_tracks,
            "short_tracks_pct": round(short_pct, 1),
            "short_track_threshold_sec": self.short_track_threshold_sec,
        }

        return TrackingQualityResult(
            analysis_id=analysis_id,
            fps=fps,
            tracks=tracks,
            quality_summary=summary,
        )
