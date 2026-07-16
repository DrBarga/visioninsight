from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    items = sorted(values)
    if len(items) == 1:
        return float(items[0])
    position = q * (len(items) - 1)
    lower = int(position)
    upper = min(lower + 1, len(items) - 1)
    fraction = position - lower
    return float(items[lower] * (1 - fraction) + items[upper] * fraction)


def _moving_average(values: List[float], window_size: int) -> List[float]:
    if window_size <= 1:
        return [float(value) for value in values]

    output: List[float] = []
    queue: List[float] = []
    running_sum = 0.0
    for value in values:
        numeric = float(value)
        queue.append(numeric)
        running_sum += numeric
        if len(queue) > window_size:
            running_sum -= queue.pop(0)
        output.append(running_sum / len(queue))
    return output


def _find_windows(
    times: List[float],
    signal: List[float],
    threshold: float,
    minimum_length_sec: float,
    window_type: str,
) -> List[Dict[str, Any]]:
    if not times or not signal or len(times) != len(signal):
        return []

    windows: List[Dict[str, Any]] = []
    in_window = False
    start_time = 0.0
    last_time = 0.0

    for time_value, signal_value in zip(times, signal):
        if signal_value >= threshold and not in_window:
            in_window = True
            start_time = float(time_value)
            last_time = float(time_value)
        elif signal_value >= threshold:
            last_time = float(time_value)
        elif in_window:
            end_time = float(last_time)
            if end_time - start_time >= float(minimum_length_sec):
                windows.append({
                    "start_sec": round(start_time, 2),
                    "end_sec": round(end_time, 2),
                    "type": window_type,
                })
            in_window = False

    if in_window:
        end_time = float(last_time)
        if end_time - start_time >= float(minimum_length_sec):
            windows.append({
                "start_sec": round(start_time, 2),
                "end_sec": round(end_time, 2),
                "type": window_type,
            })

    return windows


def _event_counts_by_time(events_path: Path) -> Dict[float, int]:
    output: Dict[float, int] = {}
    if not events_path.exists():
        return output

    for row in _iter_jsonl(events_path):
        time_value = float(row.get("time_sec", 0.0))
        events = row.get("events") or []
        output[time_value] = output.get(time_value, 0) + len(events)
    return output


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
    frame_stride: int = 1
    analysis_fps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "fps": self.fps,
            "analysis_fps": self.analysis_fps,
            "frame_stride": self.frame_stride,
            "duration_sec_est": self.duration_sec_est,
            "timeline_frames": self.timeline_frames,
            "people_count": self.people_count,
            "crowd_windows": self.crowd_windows,
            "crowd_threshold": self.crowd_threshold,
            "crowd_smoothing_sec": self.crowd_smoothing_sec,
            "crowd_dynamics": self.crowd_dynamics,
        }


class StatsBuilder:
    def __init__(
        self,
        crowd_threshold: Optional[int] = None,
        crowd_smoothing_sec: float = 0.5,
        min_window_sec: float = 0.7,
        top_percentile_q: float = 0.95,
        min_percentile_window_sec: float = 0.1,
        dynamics_window_sec: float = 1.0,
        high_density_threshold: Optional[int] = None,
        **_ignored: Any,
    ):
        if crowd_threshold is None and high_density_threshold is not None:
            crowd_threshold = int(high_density_threshold)

        self.crowd_threshold = crowd_threshold
        self.crowd_smoothing_sec = float(crowd_smoothing_sec)
        self.min_window_sec = float(min_window_sec)
        self.top_percentile_q = float(top_percentile_q)
        self.min_percentile_window_sec = float(min_percentile_window_sec)
        self.dynamics_window_sec = float(dynamics_window_sec)

    def build_from_timeline_jsonl(
        self,
        analysis_id: str,
        timeline_path: Path,
        fps: float,
        events_path: Optional[Path] = None,
        frame_stride: int = 1,
    ) -> StatsResult:
        source_fps = float(fps) if fps else 25.0
        stride = max(1, int(frame_stride))
        analysis_fps = source_fps / stride

        times: List[float] = []
        counts: List[float] = []
        frames: List[int] = []

        for row in _iter_jsonl(timeline_path):
            times.append(float(row.get("time_sec", 0.0)))
            counts.append(float(len(row.get("people") or [])))
            frames.append(int(row.get("frame", 0)))

        timeline_frames = len(times)
        duration_sec_est = float(times[-1]) if times else 0.0
        average_on_screen = sum(counts) / len(counts) if counts else 0.0
        peak_on_screen = int(max(counts)) if counts else 0
        p95 = _percentile(counts, 0.95) if counts else 0.0

        max_at = {"time_sec": 0.0, "frame": 0}
        if counts:
            max_index = max(range(len(counts)), key=lambda index: counts[index])
            max_at = {
                "time_sec": round(float(times[max_index]), 2),
                "frame": int(frames[max_index]),
            }

        smoothing_window = max(1, int(round(self.crowd_smoothing_sec * analysis_fps)))
        smoothed = _moving_average(counts, smoothing_window)

        threshold = int(self.crowd_threshold) if self.crowd_threshold is not None else int(round(p95))
        threshold = max(1, threshold)

        stable_windows = _find_windows(
            times,
            smoothed,
            float(threshold),
            self.min_window_sec,
            "stable_crowd",
        )
        percentile_threshold = _percentile(smoothed, self.top_percentile_q) if smoothed else 0.0
        percentile_windows = _find_windows(
            times,
            smoothed,
            float(percentile_threshold),
            self.min_percentile_window_sec,
            "top_percentile_crowd",
        )
        crowd_windows = stable_windows if stable_windows else percentile_windows

        dynamics_window = max(1, int(round(self.dynamics_window_sec * analysis_fps)))
        window_average = _moving_average(smoothed, dynamics_window)

        fastest_growth = None
        fastest_drop = None
        for index in range(1, len(window_average)):
            delta = float(window_average[index] - window_average[index - 1])
            item = {
                "delta": round(delta, 2),
                "start_sec": round(float(times[index - 1]), 2),
                "end_sec": round(float(times[index]), 2),
                "from": round(float(window_average[index - 1]), 2),
                "to": round(float(window_average[index]), 2),
                "window_sec": round(self.dynamics_window_sec, 2),
            }
            if fastest_growth is None or delta > float(fastest_growth["delta"]):
                fastest_growth = item
            if fastest_drop is None or delta < float(fastest_drop["delta"]):
                fastest_drop = item

        most_dynamic = None
        if events_path and events_path.exists() and times:
            event_map = _event_counts_by_time(events_path)
            event_series = [float(event_map.get(float(time_value), 0)) for time_value in times]
            rolling_sums: List[float] = []
            queue: List[float] = []
            running_sum = 0.0
            for value in event_series:
                queue.append(value)
                running_sum += value
                if len(queue) > dynamics_window:
                    running_sum -= queue.pop(0)
                rolling_sums.append(running_sum)

            if rolling_sums:
                max_index = max(range(len(rolling_sums)), key=lambda index: rolling_sums[index])
                start_index = max(0, max_index - (dynamics_window - 1))
                most_dynamic = {
                    "count": int(round(rolling_sums[max_index])),
                    "start_sec": round(float(times[start_index]), 2),
                    "end_sec": round(float(times[max_index]), 2),
                    "window_sec": round(self.dynamics_window_sec, 2),
                }

        return StatsResult(
            analysis_id=analysis_id,
            fps=source_fps,
            analysis_fps=round(analysis_fps, 4),
            frame_stride=stride,
            duration_sec_est=round(duration_sec_est, 2),
            timeline_frames=timeline_frames,
            people_count={
                "max": peak_on_screen,
                "avg": round(float(average_on_screen), 2),
                "p95": int(round(float(p95))),
                "max_at": max_at,
            },
            crowd_windows=crowd_windows,
            crowd_threshold=threshold,
            crowd_smoothing_sec=round(self.crowd_smoothing_sec, 2),
            crowd_dynamics={
                "fastest_growth": fastest_growth,
                "fastest_drop": fastest_drop,
                "most_dynamic_window": most_dynamic,
            },
        )
