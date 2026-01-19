# app/query/engine.py
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

    # ---------------- intent handlers ----------------

    if intent == "count_people":
        unique_people = summary.get("tracks_summary", {}).get("unique_people", 0)
        evidence: Dict[str, Any] = {"unique_people": unique_people}

        if stats and "people_count" in stats:
            pc = stats.get("people_count") or {}
            evidence["avg_on_screen"] = pc.get("avg", 0.0)
            evidence["peak_on_screen"] = pc.get("max", 0)

        answer = f"Total unique people detected: {unique_people}."
        if "avg_on_screen" in evidence or "peak_on_screen" in evidence:
            answer += (
                f" On-screen crowd: avg={evidence.get('avg_on_screen', 0)}, "
                f"peak={evidence.get('peak_on_screen', 0)}."
            )

        return (intent, answer, evidence, 0.9)

    if intent == "timeline_info":
        timeline_count = summary.get("timeline_count", 0)
        evidence: Dict[str, Any] = {"timeline_count": timeline_count}

        if stats:
            evidence["duration_sec_est"] = stats.get("duration_sec_est", 0.0)
            evidence["fps"] = stats.get("fps", 0.0)

        answer = f"Timeline entries saved: {timeline_count}."
        if "duration_sec_est" in evidence:
            answer += f" Estimated duration: {evidence['duration_sec_est']}s."
        return (intent, answer, evidence, 0.85)

    if intent == "peak_people":
        # Prefer stats.json if available
        if stats and "people_count" in stats:
            pc = stats.get("people_count") or {}
            peak = pc.get("max", 0)
            max_at = pc.get("max_at") or {}
            t = max_at.get("time_sec", 0.0)
            fr = max_at.get("frame", 0)

            answer = f"Peak crowd on screen: {peak} people (at {t}s, frame {fr})."
            evidence = {"max_people": peak, "time_sec": t, "frame": fr, "source": "stats.json"}
            return (intent, answer, evidence, 0.9)

        # Fallback: scan timeline.jsonl
        if not os.path.exists(timeline_path):
            return (intent, "Timeline file not found.", {}, 0.3)

        max_people = 0
        max_at = None
        scanned = 0

        for row in _iter_jsonl(timeline_path):
            scanned += 1
            ppl = row.get("people") or []
            k = len(ppl)
            if k > max_people:
                max_people = k
                max_at = (row.get("time_sec"), row.get("frame"))

        if max_at is None:
            return (intent, "No people entries found in the timeline.", {}, 0.5)

        t, fr = max_at
        answer = f"Peak crowd on screen: {max_people} people (at {t}s, frame {fr})."
        evidence = {
            "max_people": max_people,
            "time_sec": t,
            "frame": fr,
            "scanned_rows": scanned,
            "source": "timeline.jsonl",
        }
        return (intent, answer, evidence, 0.8)

    if intent == "crowd_windows":
        if not stats:
            return (
                intent,
                "Stats not found for this analysis. Run analysis again to generate stats.json.",
                {},
                0.4,
            )

        windows = stats.get("crowd_windows") or []
        threshold = stats.get("crowd_threshold")

        pc = stats.get("people_count") or {}
        peak = pc.get("max", 0)

        if not windows:
            answer = (
                "No crowd windows detected with the current settings. "
                f"Peak on-screen crowd was {peak}. (threshold={threshold})"
            )
            evidence = {"crowd_windows": [], "peak_on_screen": peak, "threshold": threshold}
            return (intent, answer, evidence, 0.85)

        top = windows[:10]
        intervals = [f"{w['start_sec']}s–{w['end_sec']}s" for w in top]

        # detect window type
        types = set(w.get("type", "") for w in top)
        if "stable_crowd" in types:
            label = f"stable high-density windows (threshold={threshold})"
        elif "top_percentile_crowd" in types:
            label = "top-percentile crowd windows"
        else:
            label = f"crowd windows (threshold={threshold})"

        answer = f"Most crowded moments ({label}): " + ", ".join(intervals) + "."
        evidence = {"crowd_windows": top, "count": len(windows), "threshold": threshold}
        return (intent, answer, evidence, 0.9)

    if intent == "events":
        if os.path.exists(events_path):
            entered = 0
            exited = 0
            sample: List[Dict[str, Any]] = []

            for row in _iter_jsonl(events_path):
                evs = row.get("events") or []
                for ev in evs:
                    et = ev.get("type")
                    if et == "person_entered":
                        entered += 1
                    elif et == "person_exited":
                        exited += 1

                    if len(sample) < 10:
                        sample.append({"frame": row.get("frame"), "time_sec": row.get("time_sec"), **ev})

            answer = (
                f"Events summary: entered={entered}, exited={exited}. "
                "Sample (up to 10) included in evidence."
            )
            evidence = {"entered": entered, "exited": exited, "sample": sample, "source": "events.jsonl"}
            return (intent, answer, evidence, 0.85)

        return (intent, "Events file not found for this analysis.", {}, 0.4)

    if intent in ("crowd_growth", "crowd_drop", "most_dynamic"):
        if not stats:
            return (intent, "Stats not found for this analysis.", {}, 0.4)

        dyn = stats.get("crowd_dynamics") or {}

        if intent == "crowd_growth":
            g = dyn.get("fastest_growth")
            if not g:
                return (intent, "No clear crowd growth detected in this video.", {"crowd_dynamics": dyn}, 0.65)
            answer = (
                f"Fastest crowd growth: +{g['delta']} (from {g['from']} to {g['to']}) "
                f"during {g['start_sec']}s–{g['end_sec']}s."
            )
            return (intent, answer, {"fastest_growth": g}, 0.9)

        if intent == "crowd_drop":
            d = dyn.get("fastest_drop")
            if not d:
                return (intent, "No clear crowd drop detected in this video.", {"crowd_dynamics": dyn}, 0.65)
            answer = (
                f"Fastest crowd drop: {d['delta']} (from {d['from']} to {d['to']}) "
                f"during {d['start_sec']}s–{d['end_sec']}s."
            )
            return (intent, answer, {"fastest_drop": d}, 0.9)

        if intent == "most_dynamic":
            m = dyn.get("most_dynamic_window")
            if not m:
                return (
                    intent,
                    "No strong event burst detected (enter/exit). Try asking about crowd growth/peak.",
                    {"crowd_dynamics": dyn},
                    0.7,
                )
            answer = f"Most dynamic moment (enter/exit burst): {m['count']} events during {m['start_sec']}s–{m['end_sec']}s."
            return (intent, answer, {"most_dynamic_window": m}, 0.9)

    if intent == "summary":
        unique_people = summary.get("tracks_summary", {}).get("unique_people", 0)
        timeline_count = summary.get("timeline_count", 0)

        evidence: Dict[str, Any] = {"unique_people": unique_people, "timeline_count": timeline_count}
        parts: List[str] = [f"unique_people={unique_people}", f"timeline_entries={timeline_count}"]

        if stats:
            evidence["duration_sec_est"] = stats.get("duration_sec_est", 0.0)
            evidence["fps"] = stats.get("fps", 0.0)

            pc = stats.get("people_count") or {}
            evidence["avg_on_screen"] = pc.get("avg", 0.0)
            evidence["peak_on_screen"] = pc.get("max", 0)
            evidence["p95_on_screen"] = pc.get("p95", 0)

            max_at = pc.get("max_at") or {}
            evidence["peak_time_sec"] = max_at.get("time_sec", 0.0)
            evidence["peak_frame"] = max_at.get("frame", 0)

            parts.append(f"duration≈{evidence['duration_sec_est']}s")
            parts.append(f"on_screen_avg={evidence['avg_on_screen']}")
            parts.append(f"on_screen_peak={evidence['peak_on_screen']} (at {evidence['peak_time_sec']}s)")

            threshold = stats.get("crowd_threshold")
            windows = stats.get("crowd_windows") or []
            evidence["threshold"] = threshold
            evidence["crowd_windows_count"] = len(windows)

            if windows:
                evidence["crowd_windows"] = windows[:10]
                parts.append(f"crowd_windows={len(windows)} (threshold={threshold})")
            else:
                parts.append(f"crowd_windows=0 (threshold={threshold})")

            # include dynamics if present
            dyn = stats.get("crowd_dynamics") or {}
            if dyn.get("fastest_growth"):
                evidence["fastest_growth"] = dyn["fastest_growth"]
                parts.append(f"fastest_growth=+{dyn['fastest_growth'].get('delta')}")
            if dyn.get("fastest_drop"):
                evidence["fastest_drop"] = dyn["fastest_drop"]
                parts.append(f"fastest_drop={dyn['fastest_drop'].get('delta')}")
            if dyn.get("most_dynamic_window"):
                evidence["most_dynamic_window"] = dyn["most_dynamic_window"]
                parts.append(f"most_dynamic_events={dyn['most_dynamic_window'].get('count')}")

            answer = "Summary: " + ", ".join(parts) + "."
            return (intent, answer, evidence, 0.9)

        # fallback if no stats.json exists
        answer = f"Summary: unique_people={unique_people}, timeline_entries={timeline_count}."
        return (intent, answer, evidence, 0.75)

    return (
        "unknown",
        "I can answer questions about counts, peak crowd, crowded windows, crowd dynamics, timeline, and enter/exit events. "
        "Try: 'How many people?', 'When was the peak crowd?', 'When was it crowded?', "
        "'When did the crowd start growing?', 'What was the most dynamic moment?', 'Give me a summary'.",
        {
            "supported_intents": [
                "count_people",
                "peak_people",
                "crowd_windows",
                "crowd_growth",
                "crowd_drop",
                "most_dynamic",
                "timeline_info",
                "events",
                "summary",
            ]
        },
        0.4,
    )
