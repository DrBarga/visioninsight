import json
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class ObjectsStatsResult:
    analysis_id: str
    fps: float
    duration_sec_est: float
    timeline_frames: int

    unique_total: int
    unique_by_class: Dict[str, int]

    on_screen_avg: float
    on_screen_peak: int
    on_screen_p95: int
    peak_at: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "fps": self.fps,
            "duration_sec_est": self.duration_sec_est,
            "timeline_frames": self.timeline_frames,
            "unique_total": self.unique_total,
            "unique_by_class": self.unique_by_class,
            "on_screen": {
                "avg": self.on_screen_avg,
                "max": self.on_screen_peak,
                "p95": self.on_screen_p95,
                "max_at": self.peak_at,
            },
        }


class ObjectsStatsBuilder:
    def build_from_objects_jsonl(
        self,
        analysis_id: str,
        objects_path: str,
        fps: float,
    ) -> ObjectsStatsResult:
        counts_on_screen: List[int] = []

        # store unique per class by track_id
        unique_ids_by_class: Dict[str, set] = {}

        peak = 0
        peak_at = {"time_sec": 0.0, "frame": 0}

        frames = 0
        last_time = 0.0

        try:
            with open(objects_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    frames += 1
                    last_time = float(row.get("time_sec", last_time))

                    objs = row.get("objects") or []
                    counts_on_screen.append(len(objs))

                    if len(objs) > peak:
                        peak = len(objs)
                        peak_at = {
                            "time_sec": float(row.get("time_sec", 0.0)),
                            "frame": int(row.get("frame", 0)),
                        }

                    for o in objs:
                        tid = o.get("track_id")
                        if tid is None:
                            continue
                        cname = (o.get("class_name") or f"class_{o.get('class_id')}").lower()
                        unique_ids_by_class.setdefault(cname, set()).add(int(tid))

        except FileNotFoundError:
            # no objects file => empty stats
            duration = 0.0
            return ObjectsStatsResult(
                analysis_id=analysis_id,
                fps=float(fps),
                duration_sec_est=duration,
                timeline_frames=0,
                unique_total=0,
                unique_by_class={},
                on_screen_avg=0.0,
                on_screen_peak=0,
                on_screen_p95=0,
                peak_at={"time_sec": 0.0, "frame": 0},
            )

        duration = float(last_time) if last_time else (frames / fps if fps else 0.0)

        if not counts_on_screen:
            return ObjectsStatsResult(
                analysis_id=analysis_id,
                fps=float(fps),
                duration_sec_est=round(duration, 2),
                timeline_frames=0,
                unique_total=0,
                unique_by_class={},
                on_screen_avg=0.0,
                on_screen_peak=0,
                on_screen_p95=0,
                peak_at={"time_sec": 0.0, "frame": 0},
            )

        counts_sorted = sorted(counts_on_screen)
        p95_index = int(0.95 * (len(counts_sorted) - 1))
        p95 = int(counts_sorted[p95_index])

        unique_by_class = {k: len(v) for k, v in unique_ids_by_class.items()}
        unique_total = sum(unique_by_class.values())

        unique_by_class = dict(sorted(unique_by_class.items(), key=lambda x: x[1], reverse=True))

        return ObjectsStatsResult(
            analysis_id=analysis_id,
            fps=float(fps),
            duration_sec_est=round(duration, 2),
            timeline_frames=len(counts_on_screen),
            unique_total=int(unique_total),
            unique_by_class=unique_by_class,
            on_screen_avg=round(sum(counts_on_screen) / len(counts_on_screen), 2),
            on_screen_peak=int(peak),
            on_screen_p95=p95,
            peak_at=peak_at,
        )
