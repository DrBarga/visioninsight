import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    crowd_dynamics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "fps": float(self.fps),
            "duration_sec_est": round(float(self.duration_sec_est), 2),
            "timeline_frames": int(self.timeline_frames),
            "people_count": self.people_count,
            "crowd_windows": self.crowd_windows,
            "crowd_threshold": int(self.crowd_threshold),
            "crowd_smoothing_sec": float(self.crowd_smoothing_sec),
            "crowd_dynamics": self.crowd_dynamics,
        }


class StatsBuilder:
    """
    Builds high-level statistics from timeline.jsonl (+ optional events.jsonl).
    v0.3.7:
      - adaptive threshold
      - smoothing
      - robust crowd windows (stable then top-percentile fallback)
      - crowd dynamics (fastest growth/drop)
      - most dynamic window by events (enter/exit count)
    """

    def __init__(
        self,
        min_window_sec: float = 0.7,
        smoothing_sec: float = 0.5,
        dynamics_window_sec: float = 1.0,
        top_percentile: int = 90,
    ):
        self.min_window_sec = float(min_window_sec)
        self.smoothing_sec = float(smoothing_sec)
        self.dynamics_window_sec = float(dynamics_window_sec)
        self.top_percentile = int(top_percentile)

    def build(
        self,
        analysis_id: str,
        timeline_path: Path,
        fps: float,
        events_path: Optional[Path] = None,
    ) -> StatsResult:
        if not timeline_path.exists():
            raise FileNotFoundError(f"timeline not found: {timeline_path}")

        counts: List[int] = []
        times: List[float] = []
        frames: List[int] = []

        fps_val = float(fps or 25.0)

        with timeline_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                people = obj.get("people") or []
                frame = int(obj.get("frame", 0))
                t = float(obj.get("time_sec", frame / fps_val))

                counts.append(len(people))
                times.append(t)
                frames.append(frame)

        if not counts:
            return StatsResult(
                analysis_id=analysis_id,
                fps=fps_val,
                duration_sec_est=0.0,
                timeline_frames=0,
                people_count={
                    "max": 0,
                    "avg": 0.0,
                    "p95": 0,
                    "max_at": {"time_sec": 0.0, "frame": 0},
                },
                crowd_windows=[],
                crowd_threshold=3,
                crowd_smoothing_sec=self.smoothing_sec,
                crowd_dynamics={},
            )

        duration = max(times) if times else 0.0

        max_count = max(counts)
        max_idx = counts.index(max_count)
        avg_count = sum(counts) / len(counts)
        p95 = self._percentile(counts, 95)

        # Adaptive threshold: meaningful but reachable
        if max_count > 1:
            threshold = max(3, min(int(p95), int(max_count) - 1))
        else:
            threshold = 3

        # Smoothing (moving average)
        smooth_n = max(1, int(round(self.smoothing_sec * fps_val)))
        smoothed = self._moving_average(counts, smooth_n)

        # Crowd windows: stable first, fallback to top-percentile
        windows = self._stable_windows(times, smoothed, threshold=float(threshold))
        if not windows:
            windows = self._top_percentile_windows(times, smoothed, percentile=self.top_percentile)

        # Dynamics: fastest growth / drop using sliding window on smoothed counts
        dyn = self._crowd_dynamics(times, smoothed, window_sec=self.dynamics_window_sec)

        # Most dynamic window: by enter/exit events frequency
        if events_path and events_path.exists():
            dyn_window = self._most_dynamic_window_by_events(events_path, window_sec=self.dynamics_window_sec)
            if dyn_window:
                dyn["most_dynamic_window"] = dyn_window

        people_count = {
            "max": int(max_count),
            "avg": round(float(avg_count), 2),
            "p95": int(p95),
            "max_at": {
                "time_sec": round(float(times[max_idx]), 2),
                "frame": int(frames[max_idx]),
            },
        }

        return StatsResult(
            analysis_id=analysis_id,
            fps=fps_val,
            duration_sec_est=float(duration),
            timeline_frames=len(counts),
            people_count=people_count,
            crowd_windows=windows,
            crowd_threshold=int(threshold),
            crowd_smoothing_sec=float(self.smoothing_sec),
            crowd_dynamics=dyn,
        )

    # ---------------- crowd windows ----------------

    def _stable_windows(self, times: List[float], smoothed: List[float], threshold: float) -> List[Dict[str, Any]]:
        windows: List[Dict[str, Any]] = []
        start: Optional[float] = None

        for t, c in zip(times, smoothed):
            if c >= threshold and start is None:
                start = t
            elif c < threshold and start is not None:
                end = t
                if (end - start) >= self.min_window_sec:
                    windows.append({"start_sec": round(start, 2), "end_sec": round(end, 2), "type": "stable_crowd"})
                start = None

        if start is not None:
            end = times[-1]
            if (end - start) >= self.min_window_sec:
                windows.append({"start_sec": round(start, 2), "end_sec": round(end, 2), "type": "stable_crowd"})

        return windows

    def _top_percentile_windows(self, times: List[float], smoothed: List[float], percentile: int) -> List[Dict[str, Any]]:
        cutoff = float(self._percentile(smoothed, percentile))
        windows: List[Dict[str, Any]] = []
        start: Optional[float] = None

        for t, c in zip(times, smoothed):
            if c >= cutoff and start is None:
                start = t
            elif c < cutoff and start is not None:
                end = t
                windows.append({"start_sec": round(start, 2), "end_sec": round(end, 2), "type": "top_percentile_crowd"})
                start = None

        if start is not None:
            end = times[-1]
            windows.append({"start_sec": round(start, 2), "end_sec": round(end, 2), "type": "top_percentile_crowd"})

        # merge ultra-short gaps (optional simple merge)
        return self._merge_close_windows(windows, gap_sec=0.15)

    def _merge_close_windows(self, windows: List[Dict[str, Any]], gap_sec: float) -> List[Dict[str, Any]]:
        if not windows:
            return []
        windows = sorted(windows, key=lambda w: w["start_sec"])
        merged = [windows[0].copy()]
        for w in windows[1:]:
            last = merged[-1]
            if w["start_sec"] - last["end_sec"] <= gap_sec and w.get("type") == last.get("type"):
                last["end_sec"] = max(last["end_sec"], w["end_sec"])
            else:
                merged.append(w.copy())
        return merged

    # ---------------- dynamics ----------------

    def _crowd_dynamics(self, times: List[float], smoothed: List[float], window_sec: float) -> Dict[str, Any]:
        if len(times) < 3:
            return {}

        # Two-pointer window: find delta in approximately window_sec
        best_growth = {"delta": 0.0}
        best_drop = {"delta": 0.0}

        j = 0
        for i in range(len(times)):
            while j < len(times) and times[j] - times[i] < window_sec:
                j += 1
            if j >= len(times):
                break

            delta = smoothed[j] - smoothed[i]
            if delta > best_growth.get("delta", 0.0):
                best_growth = {
                    "delta": round(float(delta), 2),
                    "start_sec": round(float(times[i]), 2),
                    "end_sec": round(float(times[j]), 2),
                    "from": round(float(smoothed[i]), 2),
                    "to": round(float(smoothed[j]), 2),
                    "window_sec": round(float(window_sec), 2),
                }
            if delta < best_drop.get("delta", 0.0):
                best_drop = {
                    "delta": round(float(delta), 2),  # negative
                    "start_sec": round(float(times[i]), 2),
                    "end_sec": round(float(times[j]), 2),
                    "from": round(float(smoothed[i]), 2),
                    "to": round(float(smoothed[j]), 2),
                    "window_sec": round(float(window_sec), 2),
                }

        # best_drop.delta negative => use magnitude
        out: Dict[str, Any] = {}
        if best_growth.get("delta", 0.0) > 0:
            out["fastest_growth"] = best_growth
        if best_drop.get("delta", 0.0) < 0:
            out["fastest_drop"] = best_drop
        return out

    def _most_dynamic_window_by_events(self, events_path: Path, window_sec: float) -> Optional[Dict[str, Any]]:
        # Load event timestamps (time_sec) for enter/exit
        times: List[float] = []
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                t = float(obj.get("time_sec", 0.0))
                evs = obj.get("events") or []
                for ev in evs:
                    et = ev.get("type")
                    if et in ("person_entered", "person_exited"):
                        times.append(t)

        if not times:
            return None
        times.sort()

        # Sliding window count
        best = {"count": 0}
        l = 0
        for r in range(len(times)):
            while times[r] - times[l] > window_sec:
                l += 1
            count = r - l + 1
            if count > best["count"]:
                best = {
                    "count": int(count),
                    "start_sec": round(float(times[l]), 2),
                    "end_sec": round(float(times[r]), 2),
                    "window_sec": round(float(window_sec), 2),
                }
        return best

    # ---------------- utils ----------------

    @staticmethod
    def _moving_average(values: List[int], window: int) -> List[float]:
        if window <= 1:
            return [float(v) for v in values]
        out: List[float] = []
        buf: List[float] = []
        s = 0.0
        for v in values:
            fv = float(v)
            buf.append(fv)
            s += fv
            if len(buf) > window:
                s -= buf.pop(0)
            out.append(s / len(buf))
        return out

    @staticmethod
    def _percentile(values: List[Any], p: int) -> Any:
        if not values:
            return 0
        v = sorted(values)
        k = int((p / 100) * (len(v) - 1))
        k = max(0, min(k, len(v) - 1))
        return v[k]
