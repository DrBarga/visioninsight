import json
import os
from typing import Dict, Any, Tuple, List, Optional

from app.query.intents import detect_intent


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_get_stats(run_dir: str) -> Optional[Dict[str, Any]]:
    stats_path = os.path.join(run_dir, "stats.json")
    if os.path.exists(stats_path):
        try:
            return _read_json(stats_path)
        except Exception:
            return None
    return None


def answer_question(run_dir: str, question: str) -> Tuple[str, str, Dict[str, Any], float]:
    """
    Returns: (intent, answer, evidence, confidence)
    """
    intent = detect_intent(question)

    summary_path = os.path.join(run_dir, "summary.json")
    timeline_path = os.path.join(run_dir, "timeline.jsonl")
    events_path = os.path.join(run_dir, "events.jsonl")

    if not os.path.exists(summary_path):
        return ("error", "Summary not found for this analysis.", {}, 0.2)

    summary = _read_json(summary_path)
    stats = _safe_get_stats(run_dir)

    # --- intent handlers ---

    # 1) Count people: use summary (stable), optionally enrich with stats
    if intent == "count_people":
        unique_people = summary.get("tracks_summary", {}).get("unique_people", 0)
        evidence = {"unique_people": unique_people}

        if stats and "people_count" in stats:
            evidence["avg_on_screen"] = stats["people_count"].get("avg", 0.0)
            evidence["peak_on_screen"] = stats["people_count"].get("max", 0)

        answer = f"Total unique people detected: {unique_people}."
        if "avg_on_screen" in evidence or "peak_on_screen" in evidence:
            answer += f" On-screen crowd: avg={evidence.get('avg_on_screen', 0)}, peak={evidence.get('peak_on_screen', 0)}."

        return (intent, answer, evidence, 0.9)

    # 2) Timeline info: use summary, optionally enrich with duration from stats
    if intent == "timeline_info":
        timeline_count = summary.get("timeline_count", 0)
        evidence = {"timeline_count": timeline_count}

        if stats:
            evidence["duration_sec_est"] = stats.get("duration_sec_est", 0.0)
            evidence["fps"] = stats.get("fps", 0.0)

        answer = f"Timeline entries saved: {timeline_count}."
        if "duration_sec_est" in evidence:
            answer += f" Estimated duration: {evidence['duration_sec_est']}s."

        return (intent, answer, evidence, 0.85)

    # 3) Peak people: prefer stats.json; fallback to scanning timeline
    if intent == "peak_people":
        if stats and "people_count" in stats:
            pc = stats["people_count"]
            peak = pc.get("max", 0)
            max_at = pc.get("max_at", {}) or {}
            t = max_at.get("time_sec", 0.0)
            fr = max_at.get("frame", 0)

            answer = f"Peak crowd on screen: {peak} people (at {t}s, frame {fr})."
            evidence = {"max_people": peak, "time_sec": t, "frame": fr, "source": "stats.json"}
            return (intent, answer, evidence, 0.9)

        # fallback
        if not os.path.exists(timeline_path):
            return (intent, "Timeline file not found.", {}, 0.3)

        max_people = 0
        max_at = None
        scanned = 0

        for row in _iter_jsonl(timeline_path):
            scanned += 1
            ppl = row.get("people", [])
            k = len(ppl)
            if k > max_people:
                max_people = k
                max_at = (row.get("time_sec"), row.get("frame"))

        if max_at is None:
            return (intent, "No people entries found in the timeline.", {}, 0.5)

        t, fr = max_at
        answer = f"Peak crowd on screen: {max_people} people (at {t}s, frame {fr})."
        return (intent, answer, {"max_people": max_people, "time_sec": t, "frame": fr, "scanned_rows": scanned, "source": "timeline.jsonl"}, 0.8)

    # 4) Events: your current logic is OK, but fix counting bug (events.jsonl stores batches)
    if intent == "events":
        if os.path.exists(events_path):
            entered = 0
            exited = 0
            sample: List[Dict[str, Any]] = []

            for row in _iter_jsonl(events_path):
                # row structure: {"frame":..., "time_sec":..., "events":[...]}
                evs = row.get("events") or []
                for ev in evs:
                    et = ev.get("type")
                    if et == "person_entered":
                        entered += 1
                    elif et == "person_exited":
                        exited += 1
                    if len(sample) < 10:
                        sample.append({"frame": row.get("frame"), "time_sec": row.get("time_sec"), **ev})

            answer = f"Events summary: entered={entered}, exited={exited}. Sample (up to 10) included in evidence."
            return (intent, answer, {"entered": entered, "exited": exited, "sample": sample, "source": "events.jsonl"}, 0.85)

        return (intent, "Events file not found for this analysis.", {}, 0.4)

    # 5) Summary: use stats for richer overview
    if intent == "summary":
        unique_people = summary.get("tracks_summary", {}).get("unique_people", 0)
        timeline_count = summary.get("timeline_count", 0)

        evidence: Dict[str, Any] = {
            "unique_people": unique_people,
            "timeline_count": timeline_count
        }

        parts: List[str] = []
        parts.append(f"unique_people={unique_people}")
        parts.append(f"timeline_entries={timeline_count}")

        if stats:
            evidence["duration_sec_est"] = stats.get("duration_sec_est", 0.0)
            evidence["fps"] = stats.get("fps", 0.0)

            pc = stats.get("people_count", {}) or {}
            evidence["avg_on_screen"] = pc.get("avg", 0.0)
            evidence["peak_on_screen"] = pc.get("max", 0)
            evidence["p95_on_screen"] = pc.get("p95", 0)

            max_at = pc.get("max_at", {}) or {}
            evidence["peak_time_sec"] = max_at.get("time_sec", 0.0)
            evidence["peak_frame"] = max_at.get("frame", 0)

            parts.append(f"duration≈{evidence['duration_sec_est']}s")
            parts.append(f"on_screen_avg={evidence['avg_on_screen']}")
            parts.append(f"on_screen_peak={evidence['peak_on_screen']} (at {evidence['peak_time_sec']}s)")

            windows = stats.get("crowd_windows") or []
            if windows:
                evidence["crowd_windows"] = windows[:10]  # cap
                parts.append(f"high_density_windows={len(windows)}")

            answer = "Summary: " + ", ".join(parts) + "."
            return (intent, answer, evidence, 0.88)

        # fallback if no stats.json exists
        peak_text = ""
        peak_evidence = {}
        peak_conf = 0.0
        if os.path.exists(timeline_path):
            _, peak_text, peak_evidence, peak_conf = _summary_peak(run_dir)

        answer = f"Summary: unique_people={unique_people}, timeline_entries={timeline_count}. " + (peak_text if peak_text else "")
        evidence.update(peak_evidence)
        conf = 0.75 if peak_conf == 0.0 else min(0.9, 0.75 + peak_conf * 0.2)
        return (intent, answer.strip(), evidence, conf)

    # unknown
    return (
        "unknown",
        "I can answer questions about counts, peak crowd, timeline, and enter/exit events. "
        "Try: 'How many people?', 'When was the peak crowd?', 'Give me a summary'.",
        {"supported_intents": ["count_people", "peak_people", "timeline_info", "events", "summary"]},
        0.4
    )


def _summary_peak(run_dir: str):
    timeline_path = os.path.join(run_dir, "timeline.jsonl")

    max_people = 0
    max_at = None
    for row in _iter_jsonl(timeline_path):
        ppl = row.get("people", [])
        k = len(ppl)
        if k > max_people:
            max_people = k
            max_at = (row.get("time_sec"), row.get("frame"))

    if not max_at:
        return ("peak_people", "", {}, 0.0)

    t, fr = max_at
    return (
        "peak_people",
        f"Peak_on_screen={max_people} (at {t}s, frame {fr}).",
        {"peak_people": max_people, "peak_time_sec": t, "peak_frame": fr},
        0.85
    )
