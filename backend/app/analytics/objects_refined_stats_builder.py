from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@dataclass
class ObjectsRefinedStatsResult:
    available: bool
    unique_total: int
    unique_by_label: Dict[str, int]
    top_labels: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "unique_total": self.unique_total,
            "unique_by_label": self.unique_by_label,
            "top_labels": self.top_labels,
        }


class ObjectsRefinedStatsBuilder:
    """
    Builds refined stats from object_refinements.json (one record per track_id).
    Output:
      objects_refined_stats.json
    """

    def build(
        self,
        object_refinements_json_path: Path,
        output_stats_path: Path,
        top_n: int = 20,
        min_confidence: float = 0.0,
    ) -> ObjectsRefinedStatsResult:
        object_refinements_json_path = Path(object_refinements_json_path)
        output_stats_path = Path(output_stats_path)

        if not object_refinements_json_path.exists():
            res = ObjectsRefinedStatsResult(False, 0, {}, [])
            _write_json(output_stats_path, res.to_dict())
            return res

        data = _read_json(object_refinements_json_path)
        if not data.get("available", False):
            res = ObjectsRefinedStatsResult(False, 0, {}, [])
            _write_json(output_stats_path, res.to_dict())
            return res

        refinements = data.get("refinements") or []
        by_label: Dict[str, int] = {}

        for r in refinements:
            label = (r.get("refined_label") or "").strip()
            conf = float(r.get("confidence") or 0.0)
            if not label:
                continue
            if conf < float(min_confidence):
                continue
            by_label[label] = by_label.get(label, 0) + 1

        # sort
        items = sorted(by_label.items(), key=lambda kv: kv[1], reverse=True)
        top = [{"label": k, "unique": v} for k, v in items[:top_n]]

        res = ObjectsRefinedStatsResult(True, sum(by_label.values()), by_label, top)
        _write_json(output_stats_path, res.to_dict())
        return res
