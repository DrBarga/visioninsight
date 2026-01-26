from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class HighlightsResult:
    analysis_id: str
    highlights: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {"analysis_id": self.analysis_id, "highlights": self.highlights}


class HighlightsBuilder:
    """
    Builds highlights.json from stats.json content.
    """

    def build_from_stats_dict(self, analysis_id: str, stats: Dict[str, Any]) -> HighlightsResult:
        fps = float(stats.get("fps", 30.0) or 30.0)

        highlights: List[Dict[str, Any]] = []

        # fastest growth / drop / most dynamic
        dyn = stats.get("crowd_dynamics") or {}

        g = dyn.get("fastest_growth")
        if g:
            highlights.append({
                "type": "fastest_growth",
                "start_sec": g.get("start_sec"),
                "end_sec": g.get("end_sec"),
                "reason": "Fastest crowd increase",
                "evidence": {
                    "delta": g.get("delta"),
                    "from": g.get("from"),
                    "to": g.get("to"),
                    "window_sec": g.get("window_sec"),
                }
            })

        d = dyn.get("fastest_drop")
        if d:
            highlights.append({
                "type": "fastest_drop",
                "start_sec": d.get("start_sec"),
                "end_sec": d.get("end_sec"),
                "reason": "Fastest crowd decrease",
                "evidence": {
                    "delta": d.get("delta"),
                    "from": d.get("from"),
                    "to": d.get("to"),
                    "window_sec": d.get("window_sec"),
                }
            })

        m = dyn.get("most_dynamic_window")
        if m:
            highlights.append({
                "type": "most_dynamic",
                "start_sec": m.get("start_sec"),
                "end_sec": m.get("end_sec"),
                "reason": "Highest enter/exit activity (event burst)",
                "evidence": {
                    "events_count": m.get("count"),
                    "window_sec": m.get("window_sec"),
                }
            })

        # peak crowd
        pc = stats.get("people_count") or {}
        peak = pc.get("max")
        max_at = pc.get("max_at") or {}
        if peak is not None and max_at:
            t = max_at.get("time_sec", 0.0)
            highlights.append({
                "type": "peak_crowd",
                "start_sec": round(max(0.0, float(t) - 0.2), 2),
                "end_sec": round(float(t) + 0.2, 2),
                "reason": "Maximum on-screen crowd",
                "evidence": {"people_peak": peak, "time_sec": t, "fps": fps}
            })

        # crowd windows
        windows = stats.get("crowd_windows") or []
        thr = stats.get("crowd_threshold")
        for w in windows[:10]:
            highlights.append({
                "type": "crowd_window",
                "start_sec": w.get("start_sec"),
                "end_sec": w.get("end_sec"),
                "reason": "High crowd density interval",
                "evidence": {
                    "window_type": w.get("type"),
                    "threshold": thr,
                }
            })

        # keep a stable, readable order (optional)
        order = {
            "fastest_growth": 1,
            "peak_crowd": 2,
            "crowd_window": 3,
            "most_dynamic": 4,
            "fastest_drop": 5,
        }
        highlights.sort(key=lambda x: order.get(x.get("type", ""), 999))

        return HighlightsResult(analysis_id=analysis_id, highlights=highlights)
