import re

def detect_intent(question: str) -> str:
    q = question.lower().strip()

    # counts
    if re.search(r"\b(how many|count|total)\b.*\b(people|persons)\b", q):
        return "count_people"

    if re.search(r"\b(how many|count|total)\b.*\b(objects|items)\b", q):
        return "count_objects"

    # peak crowd / max people on screen
    if re.search(r"\b(peak|max|most)\b.*\b(people|crowd)\b", q):
        return "peak_people"

    # summary
    if re.search(r"\b(summary|summarize|overview|what happened)\b", q):
        return "summary"

    # events
    if re.search(r"\b(enter|entered|exit|exited|events)\b", q):
        return "events"

    # timeline info
    if re.search(r"\b(timeline|frames|frame count|timeline count)\b", q):
        return "timeline_info"

    return "unknown"
