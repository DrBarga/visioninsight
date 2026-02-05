import re


def detect_intent(question: str) -> str:
    q = (question or "").lower().strip()

    # ---- transcript: summarize / notes ----
    if re.search(r"\b(summarize|summary|notes|recap|tl;dr)\b", q) and re.search(r"\b(video|clip|recording)\b", q):
        return "summarize_video"
    if re.search(r"\b(make|create|write)\b.*\b(summary|notes|recap)\b", q) and re.search(r"\b(video|clip)\b", q):
        return "summarize_video"
    if re.search(r"\b(конспект|кратко|перескажи|итог|резюме)\b", q) and re.search(r"\b(видео|ролик)\b", q):
        return "summarize_video"

    # ---- transcript: search ----
    if re.search(r"\b(where)\b.*\b(mention|talk about|say|discuss)\b", q):
        return "transcript_search"
    if re.search(r"\b(find)\b.*\b(where|moment|timestamp)\b", q) and re.search(r"\b(mention|talk|say|discuss)\b", q):
        return "transcript_search"
    if re.search(r"\b(где)\b.*\b(говорят|сказали|упомян|обсужда|про)\b", q):
        return "transcript_search"
    if re.search(r"\b(найди)\b.*\b(где|момент|таймкод)\b", q) and re.search(r"\b(говорят|упомян|про)\b", q):
        return "transcript_search"

    # ---- people ----
    if re.search(r"\b(how many|count|total)\b.*\b(people|persons)\b", q):
        return "count_people"

    if re.search(r"\b(peak|max|most)\b.*\b(people|crowd)\b", q):
        return "peak_people"

    # ---- crowd windows ----
    if re.search(r"\b(crowded|crowding|dense|high density|most crowded)\b", q) or \
       re.search(r"\b(when)\b.*\b(crowd|crowded)\b", q):
        return "crowd_windows"

    # ---- dynamics ----
    if re.search(r"\b(grow|growing|increase|start growing)\b.*\b(crowd|people)\b", q) or \
       re.search(r"\b(when)\b.*\b(crowd)\b.*\b(grow|increase)\b", q):
        return "crowd_growth"

    if re.search(r"\b(drop|decrease|fall)\b.*\b(crowd|people)\b", q):
        return "crowd_drop"

    if re.search(r"\b(dynamic|most dynamic)\b", q):
        return "most_dynamic"

    # ---- highlights / quality ----
    if re.search(r"\b(highlight|highlights|best moments|top moments)\b", q):
        return "highlights"

    if re.search(r"\b(quality|tracking quality|tracker quality)\b", q):
        return "quality"

    # ---- objects ----
    if re.search(r"\b(what)\b.*\b(objects|items)\b.*\b(detected|found|seen)\b", q) or \
       re.search(r"\b(objects detected|detected objects)\b", q):
        return "list_objects"

    if re.search(r"\b(how many|count|total)\b.*\b(objects|items)\b", q):
        return "count_objects"

    # ---- generic ----
    if re.search(r"\b(summary|overview|what happened)\b", q):
        return "summary"

    if re.search(r"\b(enter|entered|exit|exited|events)\b", q):
        return "events"

    if re.search(r"\b(timeline|frames|frame count|timeline count)\b", q):
        return "timeline_info"

    return "unknown"
