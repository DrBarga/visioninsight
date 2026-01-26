from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass
class TrackingQuality:
    analysis_id: str
    confirmed_tracks: int
    tentative_tracks: int
    avg_track_length_frames: float
    median_track_length_frames: float
    short_tracks_filtered: int
    coverage_est: float
    avg_iou_match: float
    track_fragmentation_est: float
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "confirmed_tracks": self.confirmed_tracks,
            "tentative_tracks": self.tentative_tracks,
            "avg_track_length_frames": self.avg_track_length_frames,
            "median_track_length_frames": self.median_track_length_frames,
            "short_tracks_filtered": self.short_tracks_filtered,
            "coverage_est": self.coverage_est,
            "avg_iou_match": self.avg_iou_match,
            "track_fragmentation_est": self.track_fragmentation_est,
            "params": self.params
        }


class TrackingQualityBuilder:
    def __init__(self, min_hits: int = 3):
        self.min_hits = int(min_hits)

    def build_from_people_jsonl(
        self,
        analysis_id: str,
        people_path: Path,
        total_frames_est: Optional[int],
        tracker_snapshot: Dict[str, Any]
    ) -> TrackingQuality:
        """
        Builds quality metrics from people.jsonl, which contains:
          {"frame":..., "time_sec":..., "people":[{track_id, track_state, ...}, ...]}
        """

        # track_id -> count_frames_seen
        counts: Dict[int, int] = {}
        states: Dict[int, str] = {}

        frames_with_people = 0
        rows = 0

        for row in _iter_jsonl(people_path):
            rows += 1
            ppl = row.get("people") or []
            if ppl:
                frames_with_people += 1
            for p in ppl:
                tid = int(p.get("track_id", -1))
                if tid < 0:
                    continue
                counts[tid] = counts.get(tid, 0) + 1
                states[tid] = str(p.get("track_state", "tentative"))

        lengths = sorted(counts.values())
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            mid = len(lengths) // 2
            if len(lengths) % 2 == 1:
                med = float(lengths[mid])
            else:
                med = (lengths[mid - 1] + lengths[mid]) / 2.0
        else:
            avg_len = 0.0
            med = 0.0

        confirmed = sum(1 for tid, st in states.items() if st == "confirmed")
        tentative = sum(1 for tid, st in states.items() if st != "confirmed")

        # coverage: fraction of frames where people were present
        # total_frames_est: if unknown, use rows (people.jsonl rows) as denominator fallback
        denom = float(total_frames_est or rows or 1)
        coverage = frames_with_people / denom

        # avg_iou_match from tracker snapshot (active confirmed avg iou)
        avg_iou = float(tracker_snapshot.get("avg_iou_match_active", 0.0))

        # fragmentation estimate: many short tracks implies fragmentation/noise
        # normalized by confirmed count
        shortish = sum(1 for L in lengths if L < (self.min_hits * 2))
        frag = (shortish / (confirmed or 1))

        short_filtered = int(tracker_snapshot.get("short_tracks_filtered", 0))

        params = tracker_snapshot.get("params", {})
        params["min_hits"] = self.min_hits

        return TrackingQuality(
            analysis_id=analysis_id,
            confirmed_tracks=confirmed,
            tentative_tracks=tentative,
            avg_track_length_frames=round(avg_len, 2),
            median_track_length_frames=round(med, 2),
            short_tracks_filtered=short_filtered,
            coverage_est=round(coverage, 4),
            avg_iou_match=round(avg_iou, 4),
            track_fragmentation_est=round(frag, 4),
            params=params
        )
