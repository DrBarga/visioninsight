import json
import os
import re
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


def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _safe_get_stats(run_dir: str) -> Optional[Dict[str, Any]]:
    return _safe_read_json(os.path.join(run_dir, "stats.json"))


def _safe_get_highlights(run_dir: str) -> Optional[Dict[str, Any]]:
    return _safe_read_json(os.path.join(run_dir, "highlights.json"))


def _safe_get_quality(run_dir: str) -> Optional[Dict[str, Any]]:
    return _safe_read_json(os.path.join(run_dir, "quality.json"))


def _safe_get_objects_stats(run_dir: str) -> Optional[Dict[str, Any]]:
    return _safe_read_json(os.path.join(run_dir, "objects_stats.json"))


def _match_object_class_from_question(question: str, available_classes: List[str]) -> Optional[str]:
    """
    Tries to find a class name mentioned in the question.
    Works even if the user writes plural: "cars" -> "car".
    """
    q = question.lower()

    # quick exact containment
    for cname in available_classes:
        if cname.lower() in q:
            return cname

    # plural handling: cars -> car, bicycles -> bicycle (naive)
    words = re.findall(r"[a-zA-Z_]+", q)
    normalized = set()
    for w in words:
        w = w.lower()
        if w.endswith("s") and len(w) > 3:
            normalized.add(w[:-1])
        normalized.add(w)

    for cname in available_classes:
        if cname.lower() in normalized:
            return cname

    return None


def _objects_stats_to_class_totals(obj_stats: Dict[str, Any]) -> Tuple[int, Dict[str, int]]:
    """
    Returns:
      (total_objects, totals_by_class)

    Supports TWO formats:

    A) CURRENT (your ObjectsStatsBuilder in objects_stats_builder.py):
       - objects_total
       - top_classes: [{class_name,total,max_in_frame}, ...]
       - objects_peak_in_frame: {max, max_at{time_sec,frame}} ...
       :contentReference[oaicite:2]{index=2}

    B) OLD/ALT (leftovers from previous iterations):
       - unique_total
       - unique_by_class: {class: count}
       :contentReference[oaicite:3]{index=3}
    """
    if not obj_stats:
        return 0, {}

    # Format B (old/alt)
    if isinstance(obj_stats.get("unique_by_class"), dict):
        totals_by_class = {str(k): int(v) for k, v in obj_stats.get("unique_by_class", {}).items()}
        total = int(obj_stats.get("unique_total", sum(totals_by_class.values())))
        return total, totals_by_class

    # Format A (current)
    totals_by_class: Dict[str, int] = {}
    top = obj_stats.get("top_classes") or []
    if isinstance(top, list):
        for item in top:
            if not isinstance(item, dict):
                continue
            cname = item.get("class_name")
            if cname is None:
                continue
            totals_by_class[str(cname)] = int(item.get("total", 0))

    total = int(obj_stats.get("objects_total", sum(totals_by_class.values())))
    return total, totals_by_class


def _available_object_classes(obj_stats: Dict[str, Any]) -> List[str]:
    _, totals_by_class = _objects_stats_to_class_totals(obj_stats)
    return list(totals_by_class.keys())


def answer_question(run_dir: str, question: str) -> Tuple[str, str, Dict[str, Any], float]:
    intent = detect_intent(question)

    summary_path = os.path.join(run_dir, "summary.json")
    timeline_path = os.path.join(run_dir, "timeline.jsonl")
    events_path = os.path.join(run_dir, "events.jsonl")

    if not os.path.exists(summary_path):
        return ("error", "Summary not found for this analysis.", {}, 0.2)

    summary = _read_json(summary_path)
    stats = _safe_get_stats(run_dir)
    highlights = _safe_get_highlights(run_dir)
    quality = _safe_get_quality(run_dir)
    obj_stats = _safe_get_objects_stats(run_dir)

    # ---------- PEOPLE ----------
    if intent == "count_people":
        unique_people = summary.get("tracks_summary", {}).get("unique_people", 0)
        evidence: Dict[str, Any] = {"unique_people": unique_people}

        if stats and "people_count" in stats:
            pc = stats["people_count"] or {}
            evidence["avg_on_screen"] = pc.get("avg", 0.0)
            evidence["peak_on_screen"] = pc.get("max", 0)

        answer = f"Total unique people detected: {unique_people}."
        if "avg_on_screen" in evidence or "peak_on_screen" in evidence:
            answer += f" On-screen crowd: avg={evidence.get('avg_on_screen', 0)}, peak={evidence.get('peak_on_screen', 0)}."
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
        if stats and "people_count" in stats:
            pc = stats["people_count"] or {}
            peak = pc.get("max", 0)
            max_at = pc.get("max_at", {}) or {}
            t = max_at.get("time_sec", 0.0)
            fr = max_at.get("frame", 0)
            answer = f"Peak crowd on screen: {peak} people (at {t}s, frame {fr})."
            evidence = {"max_people": peak, "time_sec": t, "frame": fr, "source": "stats.json"}
            return (intent, answer, evidence, 0.9)

        # fallback: scan timeline
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
        evidence = {"max_people": max_people, "time_sec": t, "frame": fr, "scanned_rows": scanned, "source": "timeline.jsonl"}
        return (intent, answer, evidence, 0.8)

    if intent == "crowd_windows":
        if not stats:
            return (intent, "Stats not found for this analysis. Run analysis again to generate stats.json.", {}, 0.4)

        windows = stats.get("crowd_windows") or []
        threshold = stats.get("crowd_threshold")

        pc = stats.get("people_count", {}) or {}
        peak = pc.get("max", 0)

        if not windows:
            answer = (
                "No high-density windows detected with the current threshold. "
                f"Peak on-screen crowd was {peak}. (threshold={threshold})"
            )
            return (intent, answer, {"crowd_windows": [], "peak_on_screen": peak, "threshold": threshold}, 0.85)

        top = windows[:10]
        intervals = [f"{w.get('start_sec')}s–{w.get('end_sec')}s" for w in top]

        types = set((w.get("type") or "") for w in top)
        if "stable_crowd" in types:
            label = f"stable high-density windows (threshold={threshold})"
        elif "top_percentile_crowd" in types:
            label = "top-percentile crowd windows"
        else:
            label = f"crowd windows (threshold={threshold})"

        answer = f"Most crowded moments ({label}): " + ", ".join(intervals) + "."
        return (intent, answer, {"crowd_windows": top, "count": len(windows), "threshold": threshold}, 0.9)

    if intent in ("crowd_growth", "crowd_drop", "most_dynamic"):
        if not stats:
            return (intent, "Stats not found for this analysis.", {}, 0.4)

        dyn = stats.get("crowd_dynamics") or {}

        if intent == "crowd_growth":
            g = dyn.get("fastest_growth")
            if not g:
                return (intent, "No clear crowd growth detected in this video.", {"crowd_dynamics": dyn}, 0.65)
            answer = f"Fastest crowd growth: +{g['delta']} (from {g['from']} to {g['to']}) during {g['start_sec']}s–{g['end_sec']}s."
            return (intent, answer, {"fastest_growth": g, "source": "stats.json"}, 0.9)

        if intent == "crowd_drop":
            d = dyn.get("fastest_drop")
            if not d:
                return (intent, "No clear crowd drop detected in this video.", {"crowd_dynamics": dyn}, 0.65)
            answer = f"Fastest crowd drop: {d['delta']} (from {d['from']} to {d['to']}) during {d['start_sec']}s–{d['end_sec']}s."
            return (intent, answer, {"fastest_drop": d, "source": "stats.json"}, 0.9)

        if intent == "most_dynamic":
            m = dyn.get("most_dynamic_window")
            if not m:
                return (intent, "No strong enter/exit burst detected. Try asking about crowd windows or highlights.", {"crowd_dynamics": dyn}, 0.7)
            answer = f"Most dynamic moment (enter/exit burst): {m['count']} events during {m['start_sec']}s–{m['end_sec']}s."
            return (intent, answer, {"most_dynamic_window": m, "source": "stats.json"}, 0.9)

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

            answer = f"Events summary: entered={entered}, exited={exited}. Sample (up to 10) included in evidence."
            return (intent, answer, {"entered": entered, "exited": exited, "sample": sample, "source": "events.jsonl"}, 0.85)

        return (intent, "Events file not found for this analysis.", {}, 0.4)

    # ---------- HIGHLIGHTS / QUALITY ----------
    if intent == "highlights":
        if not highlights:
            return (intent, "Highlights not found. Run analysis again to generate highlights.json.", {}, 0.4)

        items = highlights.get("highlights") or []
        if not items:
            return (intent, "No highlights found for this video.", {"highlights": []}, 0.75)

        top = items[:10]
        short = [f"{h.get('type')}: {h.get('start_sec')}s–{h.get('end_sec')}s" for h in top]
        answer = "Top highlights: " + ", ".join(short) + "."
        return (intent, answer, {"highlights": top, "count": len(items), "source": "highlights.json"}, 0.9)

    if intent == "quality":
        if not quality:
            return (intent, "Quality report not found. Run analysis again to generate quality.json.", {}, 0.4)

        qs = quality.get("quality_summary") or {}
        answer = (
            f"Tracking quality: tracks_total={qs.get('tracks_total')}, "
            f"avg_track_duration_sec={qs.get('avg_track_duration_sec')}, "
            f"short_tracks_pct={qs.get('short_tracks_pct')}% "
            f"(threshold={qs.get('short_track_threshold_sec')}s)."
        )
        return (intent, answer, {"quality_summary": qs, "source": "quality.json"}, 0.9)

    # ---------- OBJECTS ----------
    if intent in ("count_objects", "list_objects"):
        if not obj_stats:
            return (intent, "Objects stats not found. Run analysis again to generate objects_stats.json.", {}, 0.4)

        total_objects, totals_by_class = _objects_stats_to_class_totals(obj_stats)
        classes = list(totals_by_class.keys())
        matched = _match_object_class_from_question(question, classes)

        if intent == "count_objects":
            if matched:
                n = int(totals_by_class.get(matched, 0))
                answer = f"Total '{matched}' detected (aggregate over frames): {n}."
                evidence = {"class_name": matched, "total": n, "source": "objects_stats.json"}
                return (intent, answer, evidence, 0.9)

            answer = f"Total objects detected (aggregate over frames): {total_objects}."
            peak = obj_stats.get("objects_peak_in_frame") or {}
            if isinstance(peak, dict) and peak:
                max_on_screen = peak.get("max", 0)
                max_at = peak.get("max_at", {}) or {}
                answer += f" Peak on screen={max_on_screen} at {max_at.get('time_sec', 0.0)}s (frame {max_at.get('frame', 0)})."

            evidence = {
                "objects_total": total_objects,
                "top_classes": dict(list(totals_by_class.items())[:10]),
                "source": "objects_stats.json",
            }
            return (intent, answer, evidence, 0.9)

        if intent == "list_objects":
            if not totals_by_class:
                return (intent, "No objects were detected (non-person classes).", {"classes": []}, 0.85)

            # already sorted by builder (top_classes), but if we come from alt format, keep deterministic ordering
            top_items = list(totals_by_class.items())
            pretty = ", ".join([f"{k}={v}" for k, v in top_items[:10]])
            answer = f"Detected object classes (top): {pretty}."
            return (intent, answer, {"totals_by_class": totals_by_class, "source": "objects_stats.json"}, 0.9)

    # Fallback: questions like "How many cars?" without saying "objects"
    if intent == "unknown" and obj_stats:
        ql = question.lower()
        classes = _available_object_classes(obj_stats)
        matched = _match_object_class_from_question(question, classes)

        if matched and (("how many" in ql) or ("count" in ql) or ("total" in ql) or ("сколько" in ql)):
            total_objects, totals_by_class = _objects_stats_to_class_totals(obj_stats)
            n = int(totals_by_class.get(matched, 0))
            return (
                "count_objects",
                f"Total '{matched}' detected (aggregate over frames): {n}.",
                {"class_name": matched, "total": n, "source": "objects_stats.json"},
                0.85,
            )

    # ---------- SUMMARY ----------
    if intent == "summary":
        unique_people = summary.get("tracks_summary", {}).get("unique_people", 0)
        timeline_count = summary.get("timeline_count", 0)

        evidence: Dict[str, Any] = {"unique_people": unique_people, "timeline_count": timeline_count}
        parts: List[str] = [f"unique_people={unique_people}", f"timeline_entries={timeline_count}"]

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

            threshold = stats.get("crowd_threshold")
            windows = stats.get("crowd_windows") or []
            evidence["threshold"] = threshold
            evidence["crowd_windows_count"] = len(windows)
            parts.append(f"high_density_windows={len(windows)} (threshold={threshold})")

        if obj_stats:
            total_objects, totals_by_class = _objects_stats_to_class_totals(obj_stats)
            evidence["objects_total"] = total_objects
            evidence["objects_top_classes"] = dict(list(totals_by_class.items())[:10])
            parts.append(f"objects_total={total_objects}")

        answer = "Summary: " + ", ".join(parts) + "."
        conf = 0.9 if stats else 0.75
        return (intent, answer, evidence, conf)

    # ---------- UNKNOWN ----------
    return (
        "unknown",
        "I can answer questions about counts, peak crowd, crowded windows, crowd dynamics, highlights, tracking quality, "
        "and object analytics. Try: "
        "'How many people?', 'How many objects?', 'How many cars?', 'What objects were detected?', "
        "'When was it crowded?', 'When did the crowd start growing?', 'What was the most dynamic moment?', "
        "'Give me highlights', 'Tracking quality', 'Give me a summary'.",
        {"supported_intents": [
            "count_people", "count_objects", "list_objects",
            "peak_people", "crowd_windows", "crowd_growth", "crowd_drop", "most_dynamic",
            "highlights", "quality", "timeline_info", "events", "summary"
        ]},
        0.4
    )
