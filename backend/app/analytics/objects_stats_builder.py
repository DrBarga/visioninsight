from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class ObjectsStatsBuilder:
    """
    Builds objects_stats.json from objects.jsonl (NDJSON).

    Backward-compatible outputs:
      - objects_total (frame-aggregate)
      - objects_frames
      - avg_objects_per_frame
      - objects_peak_in_frame
      - top_classes (frame-aggregate totals)

    New outputs:
      - unique_total (track-based)
      - unique_by_class (track-based, majority-vote class per track)
      - top_classes_unique (ranked by unique counts)
    """

    def build_from_objects_jsonl(self, analysis_id: str, objects_jsonl_path: str) -> Dict[str, Any]:
        p = Path(objects_jsonl_path)
        if not p.exists():
            return {
                "analysis_id": analysis_id,
                "objects_total": 0,
                "objects_frames": 0,
                "avg_objects_per_frame": 0.0,
                "objects_peak_in_frame": {"max": 0, "max_at": None},
                "top_classes": [],
                "unique_total": 0,
                "unique_by_class": {},
                "top_classes_unique": [],
                "has_object_track_ids": False,
                "note": "objects.jsonl not found",
            }

        objects_total = 0
        frames = 0

        # frame-aggregate
        per_class_total = Counter()
        per_class_peak = Counter()

        peak_in_frame = 0
        peak_at = None

        # track-based votes: track_id -> Counter(class_name)
        track_votes: Dict[int, Counter] = defaultdict(Counter)

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                objs = row.get("objects") or []
                if not isinstance(objs, list):
                    continue

                frames += 1
                k = len(objs)
                objects_total += k

                if k > peak_in_frame:
                    peak_in_frame = k
                    peak_at = {"frame": row.get("frame"), "time_sec": row.get("time_sec")}

                # per-frame class counts
                frame_counter = Counter()

                for o in objs:
                    cn = str(o.get("class_name", "unknown")).lower()
                    frame_counter[cn] += 1

                    tid = o.get("track_id")
                    if tid is not None:
                        try:
                            tid_int = int(tid)
                            track_votes[tid_int][cn] += 1
                        except Exception:
                            pass

                for cn, cnt in frame_counter.items():
                    per_class_total[cn] += cnt
                    if cnt > per_class_peak[cn]:
                        per_class_peak[cn] = cnt

        avg = round(objects_total / frames, 4) if frames else 0.0

        # Backward-compatible top_classes (frame aggregates)
        top_classes_frame: List[Dict[str, Any]] = []
        for cn, total in per_class_total.most_common():
            top_classes_frame.append({
                "class_name": cn,
                "total": int(total),
                "max_in_frame": int(per_class_peak.get(cn, 0)),
            })

        # Track-based majority vote classification
        # final_track_class[tid] = majority class name
        final_track_class: Dict[int, str] = {}
        for tid, votes in track_votes.items():
            if not votes:
                continue
            # majority vote (ties resolved deterministically by name)
            best_count = max(votes.values())
            best_classes = sorted([c for c, v in votes.items() if v == best_count])
            final_track_class[tid] = best_classes[0]

        unique_by_class = Counter(final_track_class.values())
        unique_total = int(sum(unique_by_class.values()))

        top_unique = [{"class_name": cn, "unique": int(cnt)} for cn, cnt in unique_by_class.most_common()]

        # small debug sample: top 10 track votes to diagnose class flips
        debug = []
        for tid, votes in list(track_votes.items())[:200]:
            if not votes:
                continue
            best = final_track_class.get(tid)
            debug.append({
                "track_id": int(tid),
                "final_class": best,
                "votes": dict(votes.most_common(5)),
            })
        debug = debug[:10]

        return {
            "analysis_id": analysis_id,
            "objects_total": int(objects_total),
            "objects_frames": int(frames),
            "avg_objects_per_frame": avg,
            "objects_peak_in_frame": {"max": int(peak_in_frame), "max_at": peak_at},
            "top_classes": top_classes_frame[:20],          # frame-aggregate (compatible with Ask)
            "unique_total": unique_total,                   # track-based
            "unique_by_class": dict(unique_by_class),       # track-based
            "top_classes_unique": top_unique[:20],          # track-based ranking
            "has_object_track_ids": len(final_track_class) > 0,
            "track_class_vote_debug_top": debug,
            "note": "top_classes totals are frame-aggregate; unique_* uses track_id with majority-vote class per track.",
        }
