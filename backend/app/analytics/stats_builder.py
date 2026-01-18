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
    crowd_threshold: int
    crowd_smoothing_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "fps": self.fps,
            "duration_sec_est": round(self.duration_sec_est, 2),
            "timeline_frames": self.timeline_frames,
            "people_count": self.people_count,
            "crowd_windows": self.crowd_windows,
            "crowd_threshold": self.crowd_threshold,
            "crowd_smoothing_sec": self.crowd_smoothing_sec,
        }


class StatsBuilder:
    def __init__(self, min_window_sec: float = 0.7, smoothing_sec: float = 0.5):
        self.min_window_sec = min_window_sec
        self.smoothing_sec = smoothing_sec

    def build_from_timeline_jsonl(
        self,
        analysis_id: str,
        timeline_path: Path,
        fps: float,
    ) -> StatsResult:

        counts: List[int] = []
        times: List[float] = []

        with timeline_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                counts.append(len(obj.get("people", [])))
                times.append(float(obj.get("time_sec", 0)))

        if not counts:
            return StatsResult(
                analysis_id,
                fps,
                0.0,
                0,
                {},
                [],
                3,
                self.smoothing_sec,
            )

        duration = max(times)
        max_count = max(counts)
        avg_count = sum(counts) / len(counts)
        p95 = self._percentile(counts, 95)

        threshold = max(3, min(int(p95), int(max_count) - 1))

        smoothed = self._moving_average(counts, int(self.smoothing_sec * fps))

        windows = self._stable_windows(times, smoothed, threshold)
        if not windows:
            windows = self._top_percentile_windows(times, smoothed, percentile=90)

        return StatsResult(
            analysis_id=analysis_id,
            fps=fps,
            duration_sec_est=duration,
            timeline_frames=len(counts),
            people_count={
                "max": max_count,
                "avg": round(avg_count, 2),
                "p95": p95,
            },
            crowd_windows=windows,
            crowd_threshold=threshold,
            crowd_smoothing_sec=self.smoothing_sec,
        )

    # -------- helpers --------

    def _stable_windows(self, times, counts, threshold):
        windows = []
        start = None

        for t, c in zip(times, counts):
            if c >= threshold and start is None:
                start = t
            elif c < threshold and start is not None:
                if t - start >= self.min_window_sec:
                    windows.append({
                        "start_sec": round(start, 2),
                        "end_sec": round(t, 2),
                        "type": "stable_crowd"
                    })
                start = None

        return windows

    def _top_percentile_windows(self, times, counts, percentile=90):
        cutoff = self._percentile(counts, percentile)
        windows = []
        start = None

        for t, c in zip(times, counts):
            if c >= cutoff and start is None:
                start = t
            elif c < cutoff and start is not None:
                windows.append({
                    "start_sec": round(start, 2),
                    "end_sec": round(t, 2),
                    "type": "top_percentile_crowd"
                })
                start = None

        return windows

    def _moving_average(self, values, window):
        if window <= 1:
            return values
        out = []
        buf = []
        s = 0.0
        for v in values:
            buf.append(v)
            s += v
            if len(buf) > window:
                s -= buf.pop(0)
            out.append(s / len(buf))
        return out

    def _percentile(self, values, p):
        v = sorted(values)
        k = int((p / 100) * (len(v) - 1))
        return v[k]
