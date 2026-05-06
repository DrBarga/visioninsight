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


def _safe_get_transcript_path(run_dir: str) -> str:
    return os.path.join(run_dir, "transcript.jsonl")


def _load_transcript_segments(transcript_path: str, limit: int = 20000) -> List[Dict[str, Any]]:
    """
    Returns normalized segments: {t_start, t_end, text, confidence?}
    Skips placeholders where available==False unless it's the only line.
    """
    if not os.path.exists(transcript_path):
        return []

    items: List[Dict[str, Any]] = []
    placeholders: List[Dict[str, Any]] = []
    for i, row in enumerate(_iter_jsonl(transcript_path)):
        if i >= limit:
            break

        # placeholder case
        if row.get("available") is False:
            placeholders.append(row)
            continue

        txt = (row.get("text") or "").strip()
        if not txt:
            continue

        items.append({
            "t_start": float(row.get("t_start", 0.0) or 0.0),
            "t_end": float(row.get("t_end", 0.0) or 0.0),
            "text": txt,
            "confidence": row.get("confidence"),
        })

    if items:
        return items

    if placeholders:
        return placeholders

    return []


def _match_object_class_from_question(question: str, available_classes: List[str]) -> Optional[str]:
    """
    Tries to find a class name mentioned in the question.
    Works even if the user writes plural: "cars" -> "car".
    """
    q = question.lower()

    for cname in available_classes:
        if cname in q:
            return cname

    words = re.findall(r"[a-zA-Z_]+", q)
    normalized = set()
    for w in words:
        w = w.lower()
        if w.endswith("s") and len(w) > 3:
            normalized.add(w[:-1])
        normalized.add(w)

    for cname in available_classes:
        if cname in normalized:
            return cname

    return None


def _extract_transcript_query(question: str) -> Optional[str]:
    """
    Extracts a search query from natural language:
      - "Where do they mention X?"
      - "Find where they talk about X"
      - "Где говорят про X?"
      - quoted strings preferred
    """
    q = (question or "").strip()

    m = re.search(r"['\"]([^'\"]{2,120})['\"]", q)
    if m:
        return m.group(1).strip()

    low = q.lower()

    for pat in [
        r"\bmention\b\s+(.*)$",
        r"\btalk about\b\s+(.*)$",
        r"\bdiscuss\b\s+(.*)$",
        r"\bsay\b\s+(.*)$",
        r"\babout\b\s+(.*)$",
    ]:
        mm = re.search(pat, low)
        if mm:
            s = mm.group(1).strip()
            s = re.sub(r"[?.!,]+$", "", s).strip()
            if len(s) >= 2:
                return s

    for pat in [
        r"\bпро\b\s+(.*)$",
        r"\bупомян[а-я]*\b\s+(.*)$",
        r"\bобсужда[а-я]*\b\s+(.*)$",
        r"\bговорят\b\s+про\s+(.*)$",
    ]:
        mm = re.search(pat, low)
        if mm:
            s = mm.group(1).strip()
            s = re.sub(r"[?.!,]+$", "", s).strip()
            if len(s) >= 2:
                return s

    tokens = re.findall(r"[A-Za-zА-Яа-я0-9_]+", q)
    if len(tokens) >= 1:
        tail = " ".join(tokens[-4:]).strip()
        if len(tail) >= 2:
            return tail

    return None


def _format_time(t: float) -> str:
    t = max(0.0, float(t))
    m = int(t // 60)
    s = int(round(t - m * 60))
    return f"{m:02d}:{s:02d}"


# ===========================
# Transcript improvements (safe helpers)
# ===========================

def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize(s: str) -> List[str]:
    # English-ish tokenizer; good enough for fuzzy transcript search
    return re.findall(r"[a-zA-Z0-9]+", _normalize_text(s))


def _fuzzy_score(query_tokens: List[str], text_tokens: List[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    qs = set(query_tokens)
    ts = set(text_tokens)
    inter = len(qs & ts)
    return inter / max(1, len(qs))


def _search_transcript(segments: List[Dict[str, Any]], query: str, max_hits: int = 8) -> List[Dict[str, Any]]:
    """
    1) substring match (strong)
    2) token-overlap fuzzy match (fallback)
    """
    q_norm = _normalize_text(query)
    q_tokens = _tokenize(query)

    hits: List[Dict[str, Any]] = []

    for seg in segments:
        txt = seg.get("text") or ""
        t_norm = _normalize_text(txt)

        if q_norm and q_norm in t_norm:
            hits.append({
                "t_start": seg.get("t_start"),
                "t_end": seg.get("t_end"),
                "text": txt,
                "match": "substring",
                "score": 1.0,
            })
        else:
            # fuzzy: require >=2 tokens to avoid garbage hits for 1-word queries
            if len(q_tokens) >= 2:
                score = _fuzzy_score(q_tokens, _tokenize(txt))
                if score >= 0.5:
                    hits.append({
                        "t_start": seg.get("t_start"),
                        "t_end": seg.get("t_end"),
                        "text": txt,
                        "match": "token_overlap",
                        "score": round(score, 3),
                    })

        if len(hits) >= max_hits * 5:
            break

    hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return hits[:max_hits]


def _build_transcript_chunks(segments: List[Dict[str, Any]], chunk_sec: float = 20.0) -> List[Dict[str, Any]]:
    """
    Merge segments into time chunks for better summarization without LLM.
    """
    if not segments:
        return []

    chunks: List[Dict[str, Any]] = []
    cur_text: List[str] = []
    cur_start = float(segments[0].get("t_start", 0.0) or 0.0)
    cur_end = float(segments[0].get("t_end", cur_start) or cur_start)

    for seg in segments:
        s = float(seg.get("t_start", 0.0) or 0.0)
        e = float(seg.get("t_end", s) or s)

        if (e - cur_start) > chunk_sec and cur_text:
            chunks.append({
                "t_start": cur_start,
                "t_end": cur_end,
                "text": " ".join(cur_text).strip(),
            })
            cur_text = []
            cur_start = s
            cur_end = e

        cur_end = max(cur_end, e)
        cur_text.append((seg.get("text") or "").strip())

    if cur_text:
        chunks.append({
            "t_start": cur_start,
            "t_end": cur_end,
            "text": " ".join(cur_text).strip(),
        })

    return chunks


def _summarize_rule_based(chunks: List[Dict[str, Any]], max_points: int = 8) -> List[Dict[str, Any]]:
    """
    Extractive summary:
      - picks chunks with the most unique tokens (proxy information density)
    """
    scored = []
    for c in chunks:
        toks = set(_tokenize(c.get("text", "")))
        scored.append((len(toks), c))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = [c for _, c in scored[:max_points]]
    selected.sort(key=lambda x: x["t_start"])
    return selected


def _is_lyrics_like(segments: List[Dict[str, Any]]) -> bool:
    """
    Heuristic: repeated lines + many short lines OR profanity spikes.
    Helps warn that 'summary' might be more like lyrics than story.
    """
    if not segments:
        return False

    texts = [seg.get("text", "") for seg in segments[:140]]
    norm_lines = [_normalize_text(t) for t in texts if (t or "").strip()]
    if not norm_lines:
        return False

    unique = set(norm_lines)
    repetition_ratio = 1.0 - (len(unique) / max(1, len(norm_lines)))
    short_ratio = sum(1 for t in norm_lines if len(t.split()) <= 6) / max(1, len(norm_lines))
    prof_ratio = sum(1 for t in norm_lines if any(w in t for w in ["fuck", "shit", "goddamn"])) / max(1, len(norm_lines))

    return (short_ratio > 0.55 and repetition_ratio > 0.15) or (prof_ratio > 0.08)

def _safe_get_objects_refined_stats(run_dir: str) -> Optional[Dict[str, Any]]:
    return _safe_read_json(os.path.join(run_dir, "objects_refined_stats.json"))


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
    obj_ref_stats = _safe_get_objects_refined_stats(run_dir)


    # ---------- TRANSCRIPT ----------
    if intent in ("transcript_search", "summarize_video"):
        transcript_path = _safe_get_transcript_path(run_dir)
        segments = _load_transcript_segments(transcript_path)

        if not segments:
            return (
                intent,
                "Transcript not found for this analysis. Re-run analysis with transcript enabled to generate transcript.jsonl.",
                {"source": "transcript.jsonl", "found": False},
                0.45,
            )

        # placeholder only
        if len(segments) == 1 and segments[0].get("available") is False:
            return (
                intent,
                "Transcript is not available on this machine (audio extraction/ASR missing). "
                "Install ffmpeg and an ASR backend (recommended: faster-whisper) and re-run analysis.",
                {"source": "transcript.jsonl", "available": False, "reason": segments[0].get("reason")},
                0.55,
            )

        if intent == "transcript_search":
            query = _extract_transcript_query(question)
            if not query:
                return (
                    intent,
                    "I couldn't extract what to search for. Try: \"Where do they mention 'X'?\"",
                    {"source": "transcript.jsonl"},
                    0.5,
                )

            hits = _search_transcript(segments, query, max_hits=8)

            if not hits:
                return (
                    intent,
                    f"No transcript matches found for: '{query}'.",
                    {"query": query, "matches": [], "source": "transcript.jsonl"},
                    0.85,
                )

            pretty = "; ".join([f"{_format_time(h['t_start'])}–{_format_time(h['t_end'])}" for h in hits[:5]])
            answer = f"Found {len(hits)} transcript matches for '{query}'. Top timestamps: {pretty}."
            return (
                intent,
                answer,
                {"query": query, "matches": hits, "source": "transcript.jsonl"},
                0.92,
            )

        if intent == "summarize_video":
            chunks = _build_transcript_chunks(segments, chunk_sec=20.0)
            selected = _summarize_rule_based(chunks, max_points=8)
            lyrics_like = _is_lyrics_like(segments)

            hl_short = []
            if highlights and isinstance(highlights.get("highlights"), list):
                for h in highlights["highlights"][:6]:
                    hl_short.append({
                        "type": h.get("type"),
                        "start_sec": h.get("start_sec"),
                        "end_sec": h.get("end_sec"),
                    })

            lines = []
            if lyrics_like:
                lines.append("Note: transcript looks music/lyrics-like; summary may be less meaningful as 'story'.")

            if hl_short:
                lines.append("Key visual analytics highlights:")
                for h in hl_short:
                    lines.append(f"- {h['type']}: {h['start_sec']}s–{h['end_sec']}s")

            lines.append("Transcript summary (extractive):")
            for c in selected:
                lines.append(f"- {_format_time(c['t_start'])}–{_format_time(c['t_end'])}: {c['text'][:180]}")

            answer = "\n".join(lines)

            evidence = {
                "lyrics_like": lyrics_like,
                "highlights_sample": hl_short,
                "transcript_chunks_total": len(chunks),
                "transcript_selected_chunks": selected,
                "source": "transcript.jsonl",
            }
            if highlights:
                evidence["highlights_source"] = "highlights.json"

            return (intent, answer, evidence, 0.88)

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

        unique_total = obj_stats.get("unique_total", 0)
        unique_by_class = obj_stats.get("unique_by_class") or {}

        available_classes = list(unique_by_class.keys())
        matched = _match_object_class_from_question(question, available_classes)

        if intent == "count_objects":
            if matched:
                n = int(unique_by_class.get(matched, 0))
                answer = f"Total unique '{matched}' detected: {n}."
                evidence = {"class_name": matched, "unique": n, "source": "objects_stats.json"}
                return (intent, answer, evidence, 0.9)

            on_screen = obj_stats.get("on_screen") or {}
            answer = f"Total unique objects detected: {unique_total}."
            if on_screen:
                answer += f" On-screen objects: avg={on_screen.get('avg', 0)}, peak={on_screen.get('max', 0)}."
            evidence = {"unique_total": unique_total, "unique_by_class_top": dict(list(unique_by_class.items())[:10]), "source": "objects_stats.json"}
            return (intent, answer, evidence, 0.9)

        if intent == "list_objects":
            if not unique_by_class:
                return (intent, "No objects were detected (non-person classes).", {"unique_by_class": {}}, 0.85)

            top = list(unique_by_class.items())[:10]
            pretty = ", ".join([f"{k}={v}" for k, v in top])
            answer = f"Detected object classes (top): {pretty}."
            return (intent, answer, {"unique_by_class": unique_by_class, "source": "objects_stats.json"}, 0.9)

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
            evidence["objects_unique_total"] = obj_stats.get("unique_total", 0)
            top_classes = obj_stats.get("unique_by_class") or {}
            evidence["objects_top_classes"] = dict(list(top_classes.items())[:10])
            parts.append(f"objects_unique_total={evidence['objects_unique_total']}")

        answer = "Summary: " + ", ".join(parts) + "."
        conf = 0.9 if stats else 0.75
        return (intent, answer, evidence, conf)

    # ---------- UNKNOWN ----------
    return (
        "unknown",
        "I can answer questions about: people, crowd dynamics, highlights, tracking quality, objects, and transcript (search + summary). "
        "Try: "
        "'Summarize the video', "
        "'Where do they mention \"X\"?', "
        "'How many people?', 'When was it crowded?', "
        "'How many objects?', 'What objects were detected?', "
        "'Give me highlights', 'Tracking quality', 'Give me a summary'.",
        {"supported_intents": [
            "summarize_video", "transcript_search",
            "count_people", "peak_people", "crowd_windows", "crowd_growth", "crowd_drop", "most_dynamic",
            "highlights", "quality", "timeline_info", "events", "summary",
            "count_objects", "list_objects"
        ]},
        0.4
    )
