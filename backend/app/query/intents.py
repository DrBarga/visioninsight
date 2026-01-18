import re

def detect_intent(question: str) -> str:
    q = question.lower().strip()

    if re.search(r"\b(how many|count|total)\b.*\b(people|persons)\b", q):
        return "count_people"

    if re.search(r"\b(how many|count|total)\b.*\b(objects|items)\b", q):
        return "count_objects"

    if re.search(r"\b(peak|max|most)\b.*\b(people|crowd)\b", q):
        return "peak_people"

    # v0.3.3 crowd windows intent
    if re.search(r"\b(crowded|crowding|dense|high density|most crowded)\b", q) or \
       re.search(r"\b(when)\b.*\b(crowd|crowded)\b", q):
        return "crowd_windows"

    if re.search(r"\b(summary|summarize|overview|what happened)\b", q):
        return "summary"

    if re.search(r"\b(enter|entered|exit|exited|events)\b", q):
        return "events"

    if re.search(r"\b(timeline|frames|frame count|timeline count)\b", q):
        return "timeline_info"

    return "unknown"
