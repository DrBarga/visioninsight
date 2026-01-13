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

    # --- intent handlers ---
    if intent == "count_people":
        n = summary.get("tracks_summary", {}).get("unique_people", 0)
        return (intent, f"Total unique people detected: {n}.", {"unique_people": n}, 0.9)

    if intent == "timeline_info":
        c = summary.get("timeline_count", 0)
        return (intent, f"Timeline entries saved: {c}.", {"timeline_count": c}, 0.85)

    if intent == "peak_people":
        if not os.path.exists(timeline_path):
            return (intent, "Timeline file not found.", {}, 0.3)

        max_people = 0
        max_at = None  # (time_sec, frame)
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
        return (intent, answer, {"max_people": max_people, "time_sec": t, "frame": fr, "scanned_rows": scanned}, 0.85)

    if intent == "events":
        # Prefer events.jsonl if you have it; otherwise could fall back to timeline events.
        if os.path.exists(events_path):
            # return a short digest
            entered = 0
            exited = 0
            sample: List[Dict[str, Any]] = []
            for ev in _iter_jsonl(events_path):
                et = ev.get("type")
                if et == "person_entered":
                    entered += 1
                elif et == "person_exited":
                    exited += 1
                if len(sample) < 10:
                    sample.append(ev)

            answer = f"Events summary: entered={entered}, exited={exited}. Sample (up to 10):"
            return (intent, answer, {"entered": entered, "exited": exited, "sample": sample}, 0.8)

        return (intent, "Events file not found for this analysis.", {}, 0.4)

    if intent == "summary":
        unique_people = summary.get("tracks_summary", {}).get("unique_people", 0)
        timeline_count = summary.get("timeline_count", 0)

        peak_text = ""
        peak_evidence = {}
        peak_conf = 0.0
        if os.path.exists(timeline_path):
            _, peak_text, peak_evidence, peak_conf = _summary_peak(run_dir)

        answer = (
            f"Summary: unique_people={unique_people}, timeline_entries={timeline_count}. "
            + (peak_text if peak_text else "")
        ).strip()

        evidence = {"unique_people": unique_people, "timeline_count": timeline_count}
        evidence.update(peak_evidence)

        conf = 0.75 if peak_conf == 0.0 else min(0.9, 0.75 + peak_conf * 0.2)
        return (intent, answer, evidence, conf)

    # unknown
    return (
        "unknown",
        "I can answer questions about counts, peak crowd, timeline, and enter/exit events in v0.3. "
        "Try: 'How many people?', 'When was the peak crowd?', 'Give me a summary'.",
        {"supported_intents": ["count_people", "peak_people", "timeline_info", "events", "summary"]},
        0.4
    )

def _summary_peak(run_dir: str):
    import os
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
    return ("peak_people", f"Peak_on_screen={max_people} (at {t}s, frame {fr}).", {"peak_people": max_people, "peak_time_sec": t, "peak_frame": fr}, 0.85)
