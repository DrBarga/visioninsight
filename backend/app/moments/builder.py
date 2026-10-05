from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2


SCHEMA_VERSION = "1.0"
MAX_MOMENTS = 20
MERGE_GAP_SEC = 0.75

TITLES = {
    "peak_crowd": "Peak crowd",
    "crowd_window": "Crowded interval",
    "fastest_growth": "Crowd increased",
    "fastest_drop": "Crowd decreased",
    "most_dynamic": "Tracking activity changed",
    "line_crossing": "Line crossing",
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _transcript_excerpt(path: Path, start: float, end: float) -> str:
    if not path.exists():
        return ""
    excerpts: List[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            segment = json.loads(line)
            if segment.get("available") is False:
                continue
            if float(segment.get("t_end", 0)) >= start and float(segment.get("t_start", 0)) <= end:
                text = str(segment.get("text") or "").strip()
                if text:
                    excerpts.append(text)
            if len(excerpts) == 3:
                break
    return " ".join(excerpts)[:400]


def _merge_intervals(highlights: Iterable[Dict[str, Any]], duration: float) -> List[Dict[str, Any]]:
    intervals: List[Dict[str, Any]] = []
    for highlight in highlights:
        try:
            start = max(0.0, float(highlight["start_sec"]))
            end = min(duration, max(start, float(highlight["end_sec"])))
        except (KeyError, TypeError, ValueError):
            continue
        intervals.append({"start_sec": start, "end_sec": end, "signals": [highlight]})
    intervals.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    merged: List[Dict[str, Any]] = []
    for interval in intervals:
        if merged and interval["start_sec"] <= merged[-1]["end_sec"] + MERGE_GAP_SEC:
            merged[-1]["end_sec"] = max(merged[-1]["end_sec"], interval["end_sec"])
            merged[-1]["signals"].extend(interval["signals"])
        else:
            merged.append(interval)
    return merged[:MAX_MOMENTS]


def _save_frame(capture: cv2.VideoCapture, frame: int, path: Path) -> bool:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    success, image = capture.read()
    if not success or image is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 82]))


class MomentBuilder:
    def build(self, run_dir: Path, source_path: Path) -> Dict[str, Any]:
        run_dir = Path(run_dir)
        stats = _load_json(run_dir / "stats.json")
        highlights = list(_load_json(run_dir / "highlights.json").get("highlights") or [])
        regions = _load_json(run_dir / "zones.json")
        quality = _load_json(run_dir / "quality.json").get("quality_summary") or {}
        duration = max(0.0, float(stats.get("duration_sec_est") or 0))
        for crossing in regions.get("line_crossings") or []:
            time_sec = float(crossing["time_sec"])
            highlights.append({"type": "line_crossing", "start_sec": max(0.0, time_sec - 0.5),
                               "end_sec": min(duration, time_sec + 0.5), "evidence": crossing})
        fps = max(1.0, float(stats.get("fps") or 25))
        intervals = _merge_intervals(highlights, duration)
        continuity = max(0.0, min(1.0, float(quality.get("avg_continuity") or 0)))
        capture = cv2.VideoCapture(str(source_path))
        moments: List[Dict[str, Any]] = []
        evidence: List[Dict[str, Any]] = []
        try:
            for index, interval in enumerate(intervals, 1):
                start = round(interval["start_sec"], 2)
                end = round(interval["end_sec"], 2)
                midpoint = (start + end) / 2
                frame = max(0, int(round(midpoint * fps)))
                moment_id = f"moment_{index:04d}"
                evidence_id = f"ev_{index:04d}"
                image_relative = f"evidence/{evidence_id}.jpg"
                image_path = run_dir / image_relative
                image_available = capture.isOpened() and _save_frame(capture, frame, image_path)
                signal_types = list(dict.fromkeys(item.get("type", "unknown") for item in interval["signals"]))
                primary = next((kind for kind in ("line_crossing", "peak_crowd", "crowd_window", "fastest_growth", "fastest_drop", "most_dynamic") if kind in signal_types), signal_types[0])
                excerpt = _transcript_excerpt(run_dir / "transcript.jsonl", start, end)
                evidence.append({
                    "id": evidence_id,
                    "moment_id": moment_id,
                    "start_sec": start,
                    "end_sec": end,
                    "time_sec": round(midpoint, 2),
                    "frame": frame,
                    "sources": ["highlights.json", "stats.json"] + (["zones.json"] if "line_crossing" in signal_types else []),
                    "image": image_relative if image_available else None,
                    "signals": signal_types,
                })
                moments.append({
                    "id": moment_id,
                    "start_sec": start,
                    "end_sec": end,
                    "title": TITLES.get(primary, "Video moment"),
                    "signals": signal_types,
                    "transcript": excerpt or None,
                    "evidence_ids": [evidence_id],
                    "quality_score": round(continuity, 3),
                    "quality_basis": "mean track continuity; heuristic, not a probability",
                })
        finally:
            capture.release()

        moments_payload = {"schema_version": SCHEMA_VERSION, "moments": moments}
        evidence_payload = {"schema_version": SCHEMA_VERSION, "evidence": evidence}
        (run_dir / "moments.json").write_text(json.dumps(moments_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "evidence.json").write_text(json.dumps(evidence_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"moments_total": len(moments), "evidence_total": len(evidence)}
