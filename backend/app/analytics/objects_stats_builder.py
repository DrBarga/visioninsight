import json
from collections import Counter, defaultdict
from typing import Dict, Any

def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

class ObjectsStatsBuilder:
    def build_from_objects_jsonl(self, analysis_id: str, objects_jsonl_path: str) -> Dict[str, Any]:
        total_frames = 0
        total_objects = 0

        per_class_total = Counter()
        per_class_max_in_frame = defaultdict(int)
        max_objects_in_frame = 0
        max_at = {"time_sec": 0.0, "frame": 0}

        for row in _iter_jsonl(objects_jsonl_path):
            total_frames += 1
            objs = row.get("objects") or []
            k = len(objs)
            total_objects += k

            if k > max_objects_in_frame:
                max_objects_in_frame = k
                max_at = {"time_sec": row.get("time_sec", 0.0), "frame": row.get("frame", 0)}

            per_frame_class = Counter(o.get("class_name", str(o.get("class_id", "unknown"))) for o in objs)
            for cls, cnt in per_frame_class.items():
                per_class_total[cls] += cnt
                if cnt > per_class_max_in_frame[cls]:
                    per_class_max_in_frame[cls] = cnt

        avg_objects_per_frame = round(total_objects / total_frames, 2) if total_frames else 0.0

        top_classes = [
            {"class_name": cls, "total": int(cnt), "max_in_frame": int(per_class_max_in_frame[cls])}
            for cls, cnt in per_class_total.most_common(15)
        ]

        return {
            "analysis_id": analysis_id,
            "objects_frames": total_frames,
            "objects_total": total_objects,
            "avg_objects_per_frame": avg_objects_per_frame,
            "objects_peak_in_frame": {
                "max": max_objects_in_frame,
                "max_at": max_at,
            },
            "top_classes": top_classes,
        }
