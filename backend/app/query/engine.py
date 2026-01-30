import json
import os
import re
from typing import Dict, Any, Tuple, List, Optional


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


def _normalize_word(w: str) -> str:
    w = (w or "").lower().strip()
    # naive plural normalization
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w


def _match_object_class_from_question(question: str, available_classes: List[str]) -> Optional[str]:
    q = (question or "").lower()
    # quick contains
    for cname in available_classes:
        if cname.lower() in q:
            return cname

    words = re.findall(r"[a-zA-Z_]+", q)
    norm = set(_normalize_word(w) for w in words)
    for cname in available_classes:
        if _normalize_word(cname) in norm:
            return cname
    return None


def _objects_stats_extract(obj_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unifies both old/new formats into a single dict.

    Returns keys:
      - frame_total_all: int
      - frame_totals_by_class: Dict[str,int]
      - unique_total_all: int
      - unique_by_class: Dict[str,int]
      - peak_in_frame: dict (max/max_at)
    """
    if not obj_stats:
        return {
            "frame_total_all": 0,
            "frame_totals_by_class": {},
            "unique_total_all": 0,
            "unique_by_class": {},
            "peak_in_frame": {},
        }

    # frame totals (current)
    frame_total_all = int(obj_stats.get("objects_total", 0) or 0)

    frame_totals_by_class: Dict[str, int] = {}
    top = obj_stats.get("top_classes") or []
    if isinstance(top, list):
        for item in top:
            if not isinstance(item, dict):
                continue
            cn = item.get("class_name")
            if cn is None:
                continue
            frame_totals_by_class[str(cn).lower()] = int(item.get("total", 0) or 0)

    # unique (new fix we just added)
    unique_by_class: Dict[str, int] = {}
    if isinstance(obj_stats.get("unique_by_class"), dict):
        for k, v in obj_stats.get("unique_by_class", {}).items():
            unique_by_class[str(k).lower()] = int(v or 0)

    unique_total_all = int(obj_stats.get("unique_total", 0) or sum(unique_by_class.values()))

    # peak
    peak = obj_stats.get("objects_peak_in_frame") or {}

    # Backward compatibility: if someone has old/alt format (unique_total/unique_by_class only)
    if frame_total_all == 0 and not frame_totals_by_class and isinstance(obj_stats.get("top_classes"), dict):
        # not expected, but keep safe
        for k, v in obj_stats.get("top_classes", {}).items():
            frame_totals_by_class[str(k).lower()] = int(v or 0)
        frame_total_all = int(sum(frame_totals_by_class.values()))

    return {
        "frame_total_all": frame_total_all,
        "frame_totals_by_class": frame_totals_by_class,
        "unique_total_all": unique_total_all,
        "unique_by_class": unique_by_class,
        "peak_in_frame": peak,
    }


def _question_wants_unique(question: str) -> bool:
    q = (question or "").lower()
    # English cues
    if any(x in q for x in ["unique", "distinct", "different", "how many cars were there", "how many were there"]):
        return True
    # Russian cues
    if any(x in q for x in ["уник", "разных", "сколько было", "всего было", "сколько всего было"]):
        return True
    return False


def _question_wants_frame_aggregate(question: str) -> bool:
    q = (question or "").lower()
    # English cues
    if any(x in q for x in ["detections", "detected in total", "aggregate", "total detections", "across frames"]):
        return True
    # Russian cues
    if any(x in q for x in ["детекц", "суммарно по кадрам", "в сумме по кадрам", "всего детекций"]):
        return True
    return False


# -------------------- MAIN API --------------------

from app.query.intents import detect_intent  # keep import here to avoid circular issues


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

    # ---------- CROWD WINDOWS / DYNAMICS ----------
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

    # ---------- EVENTS ----------
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
    if intent in ("count_objects", "list_objects", "peak_objects"):
        if not obj_stats:
            return (intent, "Objects stats not found. Run analysis again to generate objects_stats.json.", {}, 0.4)

        x = _objects_stats_extract(obj_stats)
        frame_total_all = x["frame_total_all"]
        frame_totals_by_class = x["frame_totals_by_class"]
        unique_total_all = x["unique_total_all"]
        unique_by_class = x["unique_by_class"]
        peak = x["peak_in_frame"] or {}

        # combine known class universe
        classes = sorted(set(list(frame_totals_by_class.keys()) + list(unique_by_class.keys())))
        matched = _match_object_class_from_question(question, classes)

        wants_unique = _question_wants_unique(question)
        wants_aggregate = _question_wants_frame_aggregate(question)
        if not wants_unique and not wants_aggregate:
            # default: if we have unique_by_class, prefer unique for "how many X were there"
            wants_unique = True if unique_by_class else False
            wants_aggregate = not wants_unique

        if intent == "list_objects":
            # show both (if available), but keep compact
            if not classes:
                return (intent, "No objects were detected (non-person classes).", {"classes": []}, 0.85)

            # sort by frame totals (more stable view), fallback to unique
            if frame_totals_by_class:
                items = sorted(frame_totals_by_class.items(), key=lambda kv: kv[1], reverse=True)
                pretty = ", ".join([f"{k}={v}" for k, v in items[:10]])
                answer = f"Detected object classes (frame-aggregate top): {pretty}."
                ev = {"frame_totals_by_class": frame_totals_by_class, "source": "objects_stats.json"}
                if unique_by_class:
                    ev["unique_by_class"] = unique_by_class
                return (intent, answer, ev, 0.9)

            items = sorted(unique_by_class.items(), key=lambda kv: kv[1], reverse=True)
            pretty = ", ".join([f"{k}={v}" for k, v in items[:10]])
            answer = f"Detected object classes (unique top): {pretty}."
            return (intent, answer, {"unique_by_class": unique_by_class, "source": "objects_stats.json"}, 0.9)

        if intent == "peak_objects":
            max_on_screen = peak.get("max", 0)
            max_at = peak.get("max_at", {}) or {}
            answer = f"Peak objects on screen: {max_on_screen} at {max_at.get('time_sec', 0.0)}s (frame {max_at.get('frame', 0)})."
            return (intent, answer, {"objects_peak_in_frame": peak, "source": "objects_stats.json"}, 0.9)

        if intent == "count_objects":
            metric = "unique" if wants_unique else "frame_aggregate"

            if matched:
                if wants_unique:
                    n = int(unique_by_class.get(matched, 0))
                    answer = f"Unique '{matched}' objects: {n}."
                    evidence = {"class_name": matched, "unique": n, "metric": metric, "source": "objects_stats.json"}
                    return (intent, answer, evidence, 0.9)
                else:
                    n = int(frame_totals_by_class.get(matched, 0))
                    answer = f"Total '{matched}' detections (aggregate over frames): {n}."
                    evidence = {"class_name": matched, "total": n, "metric": metric, "source": "objects_stats.json"}
                    return (intent, answer, evidence, 0.9)

            # no class
            if wants_unique:
                answer = f"Total unique objects: {unique_total_all}."
                ev = {"unique_total": unique_total_all, "metric": metric, "source": "objects_stats.json"}
                return (intent, answer, ev, 0.9)

            answer = f"Total objects detected (aggregate over frames): {frame_total_all}."
            if isinstance(peak, dict) and peak:
                max_on_screen = peak.get("max", 0)
                max_at = peak.get("max_at", {}) or {}
                answer += f" Peak on screen={max_on_screen} at {max_at.get('time_sec', 0.0)}s (frame {max_at.get('frame', 0)})."
            ev = {"objects_total": frame_total_all, "metric": metric, "source": "objects_stats.json"}
            return (intent, answer, ev, 0.9)

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
            x = _objects_stats_extract(obj_stats)
            evidence["objects_total_frame_aggregate"] = x["frame_total_all"]
            evidence["objects_total_unique"] = x["unique_total_all"]
            parts.append(f"objects_unique={x['unique_total_all']}")
            parts.append(f"objects_frame_aggregate={x['frame_total_all']}")

        answer = "Summary: " + ", ".join(parts) + "."
        conf = 0.9 if stats else 0.75
        return (intent, answer, evidence, conf)

    # ---------- UNKNOWN ----------
    return (
        "unknown",
        "I can answer about people counts, peak crowd, crowded windows, crowd dynamics, highlights, tracking quality, "
        "and object analytics (unique vs frame-aggregate). Try: "
        "'How many people?', 'Peak people', 'When was it crowded?', "
        "'What objects were detected?', 'How many cars were there? (unique)', "
        "'How many car detections? (aggregate)', 'Peak objects', 'Give me a summary'.",
        {
            "supported_intents": [
                "count_people", "peak_people", "crowd_windows", "crowd_growth", "crowd_drop", "most_dynamic",
                "highlights", "quality", "events", "summary",
                "list_objects", "count_objects", "peak_objects"
            ]
        },
        0.4,
    )
