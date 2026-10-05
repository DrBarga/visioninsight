from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.query.engine import answer_question
from app.retrieval.moments import retrieve


def _read(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp(source: Any) -> Optional[float]:
    if isinstance(source, dict):
        for key in ("time_sec", "start_sec", "t_start", "peak_time_sec"):
            value = source.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for value in source.values():
            found = _timestamp(value)
            if found is not None:
                return found
    if isinstance(source, list):
        for value in source:
            found = _timestamp(value)
            if found is not None:
                return found
    return None


def _source(intent: str) -> str:
    return {
        "count_people": "summary.json/tracks_summary",
        "timeline_info": "summary.json/source",
        "count_objects": "objects_stats.json/unique_total",
        "list_objects": "objects_stats.json/unique_by_class",
        "quality": "quality.json/quality_summary",
        "events": "events.jsonl",
        "transcript_search": "transcript.jsonl",
        "summarize_video": "transcript.jsonl",
    }.get(intent, "stats.json")


def answer_with_evidence(run_dir: Path, question: str) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    moments = _read(run_dir / "moments.json").get("moments") or []
    cards = _read(run_dir / "evidence.json").get("evidence") or []
    summary = _read(run_dir / "summary.json")
    duration = float(summary.get("source", {}).get("duration_sec_est") or 0)
    intent, answer, facts, confidence = answer_question(str(run_dir), question)
    if intent == "unknown":
        method, hits = retrieve(question, moments)
        if not hits:
            return {"schema_version": "1.0", "answer": "I could not verify an answer from this analysis.",
                    "intent": "unsupported", "confidence": 0.0, "evidence_quality": "none",
                    "evidence": [], "search_mode": method}
        selected = [card for hit in hits for card in cards if card["id"] in hit["moment"]["evidence_ids"]]
        descriptions = [f"{hit['moment']['title']} at {hit['moment']['start_sec']:.1f}s" for hit in hits[:3]]
        return {"schema_version": "1.0", "answer": "Relevant moments: " + "; ".join(descriptions) + ".",
                "intent": "moment_search", "confidence": None, "evidence_quality": "direct",
                "evidence": selected, "search_mode": method}

    if intent == "error" or confidence < 0.6 or not facts:
        return {"schema_version": "1.0", "answer": answer, "intent": intent,
                "confidence": 0.0, "evidence_quality": "none", "evidence": [],
                "search_mode": "deterministic"}

    time_sec = _timestamp(facts)
    selected: List[Dict[str, Any]] = []
    if time_sec is not None and cards:
        nearby = sorted(cards, key=lambda card: abs(float(card["time_sec"]) - time_sec))
        if nearby and nearby[0]["start_sec"] - 1 <= time_sec <= nearby[0]["end_sec"] + 1:
            selected = nearby[:1]
    direct = bool(selected)
    if not selected:
        selected = [{
            "id": "aggregate_" + intent,
            "start_sec": 0.0,
            "end_sec": duration,
            "time_sec": time_sec,
            "frame": None,
            "sources": [_source(intent)],
            "image": None,
            "signals": [intent],
        }]
    return {"schema_version": "1.0", "answer": answer, "intent": intent,
            "confidence": confidence, "evidence_quality": "direct" if direct else "aggregate",
            "evidence": selected, "search_mode": "deterministic"}
