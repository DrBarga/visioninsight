import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StatsResult:
    analysis_id: str
    fps: float
    duration_sec_est: float
    timeline_frames: int
    people_count: Dict[str, Any]
    crowd_windows: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "fps": self.fps,
            "duration_sec_est": round(self.duration_sec_est, 2),
            "timeline_frames": self.timeline_frames,
            "people_count": self.people_count,
            "crowd_windows": self.crowd_windows,
        }


class StatsBuilder:
    """
    Builds high-level statistics from timeline.jsonl.
    timeline.jsonl: one JSON object per line:
      { "frame": ..., "time_sec": ..., "people": [...], "events": [...] }
    """

    def __init__(self, high_density_threshold: int = 10, min_window_sec: float = 0.7):
        self.high_density_threshold = high_density_threshold
        self.min_window_sec = min_window_sec

    def build_from_timeline_jsonl(
        self,
        analysis_id: str,
        timeline_path: Path,
        fps: float,
    ) -> StatsResult:
        if not timeline_path.exists():
            raise FileNotFoundError(f"timeline not found: {timeline_path}")

        counts: List[int] = []
        time_points: List[float] = []
        frame_points: List[int] = []

        with timeline_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                people = obj.get("people") or []
                frame = int(obj.get("frame", 0))
                time_sec = float(obj.get("time_sec", frame / (fps or 25.0)))

                counts.append(len(people))
                time_points.append(time_sec)
                frame_points.append(frame)

        if not counts:
            # timeline empty => stats empty
            return StatsResult(
                analysis_id=analysis_id,
                fps=float(fps),
                duration_sec_est=0.0,
                timeline_frames=0,
                people_count={
                    "max": 0,
                    "avg": 0.0,
                    "p95": 0,
                    "max_at": {"time_sec": 0.0, "frame": 0},
                },
                crowd_windows=[],
            )

        max_count = max(counts)
        max_idx = counts.index(max_count)
        avg_count = sum(counts) / len(counts)

        p95 = self._percentile(counts, 95)

        duration_est = max(time_points) if time_points else 0.0

        people_count = {
            "max": int(max_count),
            "avg": round(float(avg_count), 2),
            "p95": int(p95),
            "max_at": {
                "time_sec": round(float(time_points[max_idx]), 2),
                "frame": int(frame_points[max_idx]),
            },
        }

        windows = self._build_crowd_windows(time_points, counts)

        return StatsResult(
            analysis_id=analysis_id,
            fps=float(fps),
            duration_sec_est=float(duration_est),
            timeline_frames=len(counts),
            people_count=people_count,
            crowd_windows=windows,
        )

    def _build_crowd_windows(self, times: List[float], counts: List[int]) -> List[Dict[str, Any]]:
        """
        Finds contiguous windows where people_count >= high_density_threshold.
        """
        windows: List[Dict[str, Any]] = []
        in_window = False
        start_sec: Optional[float] = None

        for t, c in zip(times, counts):
            if c >= self.high_density_threshold and not in_window:
                in_window = True
                start_sec = t
            elif c < self.high_density_threshold and in_window:
                # close window
                end_sec = t
                if start_sec is not None and (end_sec - start_sec) >= self.min_window_sec:
                    windows.append({
                        "start_sec": round(start_sec, 2),
                        "end_sec": round(end_sec, 2),
                        "reason": "high_density"
                    })
                in_window = False
                start_sec = None

        # if ended inside window
        if in_window and start_sec is not None:
            end_sec = times[-1]
            if (end_sec - start_sec) >= self.min_window_sec:
                windows.append({
                    "start_sec": round(start_sec, 2),
                    "end_sec": round(end_sec, 2),
                    "reason": "high_density"
                })

        return windows

    @staticmethod
    def _percentile(values: List[int], p: int) -> int:
        if not values:
            return 0
        v = sorted(values)
        # nearest-rank method
        k = int(round((p / 100.0) * (len(v) - 1)))
        k = max(0, min(k, len(v) - 1))
        return v[k]
