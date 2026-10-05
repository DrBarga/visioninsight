from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ObjectsRefinedStatsResult:
    available: bool
    unique_total: int
    unique_by_label: Dict[str, int]
    top_labels: List[Dict[str, Any]]
    refined_count: int = 0
    fallback_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "available": self.available,
            "unique_total": self.unique_total,
            "unique_by_label": self.unique_by_label,
            "top_labels": self.top_labels,
            "refined_count": self.refined_count,
            "fallback_count": self.fallback_count,
        }


class ObjectsRefinedStatsBuilder:
    """Count each object track once, retaining the detector label unless CLIP is clear."""

    def build(
        self,
        object_refinements_json_path: Path,
        output_stats_path: Path,
        objects_jsonl_path: Optional[Path] = None,
        top_n: int = 20,
        min_confidence: float = 0.50,
        min_margin: float = 0.15,
    ) -> ObjectsRefinedStatsResult:
        refinements_path = Path(object_refinements_json_path)
        output_path = Path(output_stats_path)
        votes: Dict[int, Counter[str]] = defaultdict(Counter)

        if objects_jsonl_path is not None and Path(objects_jsonl_path).exists():
            with Path(objects_jsonl_path).open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    for item in row.get("objects") or []:
                        track_id = item.get("track_id")
                        if track_id is None:
                            continue
                        label = str(item.get("class_name") or "unknown").strip().lower()
                        votes[int(track_id)][label] += 1

        data: Dict[str, Any] = {}
        if refinements_path.exists():
            data = json.loads(refinements_path.read_text(encoding="utf-8"))

        by_track = {
            track_id: sorted(counts.items(), key=lambda entry: (-entry[1], entry[0]))[0][0]
            for track_id, counts in votes.items()
        }
        refined_count = 0
        for item in data.get("refinements") or []:
            track_id = int(item["track_id"])
            detector_label = str(item.get("yolo_class") or "unknown").strip().lower()
            by_track.setdefault(track_id, detector_label)
            candidates = item.get("candidates") or []
            first_score = float(candidates[0].get("score", 0)) if candidates else 0.0
            second_score = float(candidates[1].get("score", 0)) if len(candidates) > 1 else 0.0
            label = str(item.get("refined_label") or "").strip().lower()
            if label and first_score >= min_confidence and first_score - second_score >= min_margin:
                by_track[track_id] = label
                refined_count += 1

        labels = Counter(by_track.values())
        ordered = sorted(labels.items(), key=lambda entry: (-entry[1], entry[0]))
        result = ObjectsRefinedStatsResult(
            available=bool(by_track),
            unique_total=len(by_track),
            unique_by_label=dict(ordered),
            top_labels=[{"label": label, "unique": count} for label, count in ordered[:top_n]],
            refined_count=refined_count,
            fallback_count=len(by_track) - refined_count,
        )
        output_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return result
