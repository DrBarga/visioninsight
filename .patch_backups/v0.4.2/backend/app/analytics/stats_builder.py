from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def _percentile(values: List[float], q: float) -> float:
    """
    Simple percentile implementation (q in [0..1]).
    """
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


def _moving_average(values: List[float], window_n: int) -> List[float]:
    if window_n <= 1:
        return [float(v) for v in values]
    out: List[float] = []
    s = 0.0
    q: List[float] = []
    for v in values:
        q.append(float(v))
        s += float(v)
        if len(q) > window_n:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def _find_windows(times: List[float], signal: List[float], thr: float, min_len_sec: float, wtype: str) -> List[Dict[str, Any]]:
    """
    Finds contiguous windows where signal >= thr.
    Window boundaries are based on times[].
    """
    if not times or not signal or len(times) != len(signal):
        return []

    windows: List[Dict[str, Any]] = []
    in_win = False
    start_t = 0.0
    last_t = 0.0

    for t, v in zip(times, signal):
        if v >= thr and not in_win:
            in_win = True
            start_t = float(t)
            last_t = float(t)
        elif v >= thr and in_win:
            last_t = float(t)
        elif v < thr and in_win:
            end_t = float(last_t)
            if end_t - start_t >= float(min_len_sec):
                windows.append({"start_sec": round(start_t, 2), "end_sec": round(end_t, 2), "type": wtype})
            in_win = False

    # close if ended in window
    if in_win:
        end_t = float(last_t)
        if end_t - start_t >= float(min_len_sec):
            windows.append({"start_sec": round(start_t, 2), "end_sec": round(end_t, 2), "type": wtype})

    return windows


def _event_counts_by_time(events_jsonl: Path) -> Dict[float, int]:
    """
    Builds a map time_sec -> number_of_events at that time.
    Expected events.jsonl format:
      {"frame":..., "time_sec":..., "events":[{"type":"person_entered"...}, ...]}
    """
    out: Dict[float, int] = {}
    if not events_jsonl.exists():
        return out

    for row in _iter_jsonl(events_jsonl):
        t = float(row.get("time_sec", 0.0))
        evs = row.get("events") or []
        out[t] = out.get(t, 0) + int(len(evs))
    return out


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
            "fps": self.fps,
            "duration_sec_est": self.duration_sec_est,
            "timeline_frames": self.timeline_frames,
            "people_count": self.people_count,
            "crowd_windows": self.crowd_windows,
            "crowd_threshold": self.crowd_threshold,
            "crowd_smoothing_sec": self.crowd_smoothing_sec,
            "crowd_dynamics": self.crowd_dynamics,
        }


class StatsBuilder:
    """
    Backward compatible builder.

    Supported init params:
      - crowd_threshold (preferred)
      - high_density_threshold (legacy alias -> crowd_threshold)
      - crowd_smoothing_sec
      - min_window_sec (for 'stable_crowd' windows)
      - top_percentile_q (for 'top_percentile_crowd' windows)
      - min_percentile_window_sec (allows short percentile peaks)
      - dynamics_window_sec (for growth/drop + event burst)
    """

    def __init__(
        self,
        crowd_threshold: Optional[int] = None,
        crowd_smoothing_sec: float = 0.5,
        min_window_sec: float = 0.7,
        top_percentile_q: float = 0.95,
        min_percentile_window_sec: float = 0.1,
        dynamics_window_sec: float = 1.0,
        # legacy:
        high_density_threshold: Optional[int] = None,
        **_ignored: Any,
    ):
        # legacy alias support
        if crowd_threshold is None and high_density_threshold is not None:
            crowd_threshold = int(high_density_threshold)

        self.crowd_threshold = crowd_threshold  # may be None -> computed from p95
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
    ) -> StatsResult:
        fps = float(fps) if fps else 25.0

        times: List[float] = []
        counts: List[float] = []
        frames: List[int] = []

        for row in _iter_jsonl(timeline_path):
            t = float(row.get("time_sec", 0.0))
            ppl = row.get("people") or []
            k = float(len(ppl))

            times.append(t)
            counts.append(k)
            frames.append(int(row.get("frame", 0)))

        timeline_frames = len(times)

        duration_sec_est = 0.0
        if times:
            duration_sec_est = float(times[-1])
        elif timeline_frames > 0:
            duration_sec_est = float(timeline_frames) / fps

        avg_on_screen = sum(counts) / len(counts) if counts else 0.0
        peak_on_screen = int(max(counts)) if counts else 0
        p95 = _percentile(counts, 0.95) if counts else 0.0

        # max_at
        max_at = {"time_sec": 0.0, "frame": 0}
        if counts:
            idx = max(range(len(counts)), key=lambda i: counts[i])
            max_at = {"time_sec": round(float(times[idx]), 2), "frame": int(frames[idx])}

        # smoothing
        window_n = max(1, int(round(self.crowd_smoothing_sec * fps)))
        smooth = _moving_average(counts, window_n)

        # threshold
        thr = int(self.crowd_threshold) if self.crowd_threshold is not None else int(round(p95))
        thr = max(1, thr)

        # windows:
        # 1) stable windows (>= thr for >= min_window_sec)
        stable = _find_windows(times, smooth, float(thr), self.min_window_sec, "stable_crowd")

        # 2) top-percentile windows (can be shorter)
        top_thr = _percentile(smooth, self.top_percentile_q) if smooth else 0.0
        percentile_windows = _find_windows(times, smooth, float(top_thr), self.min_percentile_window_sec, "top_percentile_crowd")

        # choose which to expose:
        # - if stable exists -> keep stable
        # - else keep percentile windows (matches твоим кейсам)
        crowd_windows = stable if stable else percentile_windows

        # dynamics (growth/drop) using window averages of smoothed signal
        win_n = max(1, int(round(self.dynamics_window_sec * fps)))
        win_avg = _moving_average(smooth, win_n)

        fastest_growth = None
        fastest_drop = None
        for i in range(1, len(win_avg)):
            delta = float(win_avg[i] - win_avg[i - 1])
            start_t = float(times[max(0, i - 1)]) if times else 0.0
            end_t = float(times[i]) if times else 0.0
            item = {
                "delta": round(delta, 2),
                "start_sec": round(start_t, 2),
                "end_sec": round(end_t, 2),
                "from": round(float(win_avg[i - 1]), 2),
                "to": round(float(win_avg[i]), 2),
                "window_sec": round(self.dynamics_window_sec, 2),
            }
            if fastest_growth is None or delta > float(fastest_growth["delta"]):
                fastest_growth = item
            if fastest_drop is None or delta < float(fastest_drop["delta"]):
                fastest_drop = item

        # most_dynamic_window: use events.jsonl if available
        most_dynamic = None
        if events_path and events_path.exists() and times:
            event_map = _event_counts_by_time(events_path)
            # align events to times
            ev_series = [float(event_map.get(float(t), 0)) for t in times]
            ev_win = _moving_average(ev_series, win_n)  # average; we want sum, so multiply by len(window) roughly
            # better: compute rolling sum
            ev_sum: List[float] = []
            s = 0.0
            q: List[float] = []
            for v in ev_series:
                q.append(float(v))
                s += float(v)
                if len(q) > win_n:
                    s -= q.pop(0)
                ev_sum.append(s)

            if ev_sum:
                idx = max(range(len(ev_sum)), key=lambda i: ev_sum[i])
                start_i = max(0, idx - (win_n - 1))
                most_dynamic = {
                    "count": int(round(ev_sum[idx])),
                    "start_sec": round(float(times[start_i]), 2),
                    "end_sec": round(float(times[idx]), 2),
                    "window_sec": round(self.dynamics_window_sec, 2),
                }

        crowd_dynamics = {
            "fastest_growth": fastest_growth,
            "fastest_drop": fastest_drop,
            "most_dynamic_window": most_dynamic,
        }

        people_count = {
            "max": int(peak_on_screen),
            "avg": round(float(avg_on_screen), 2),
            "p95": int(round(float(p95))),
            "max_at": max_at,
        }

        return StatsResult(
            analysis_id=analysis_id,
            fps=fps,
            duration_sec_est=round(float(duration_sec_est), 2),
            timeline_frames=int(timeline_frames),
            people_count=people_count,
            crowd_windows=crowd_windows,
            crowd_threshold=int(thr),
            crowd_smoothing_sec=round(float(self.crowd_smoothing_sec), 2),
            crowd_dynamics=crowd_dynamics,
        )
