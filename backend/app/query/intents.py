import re

def detect_intent(question: str) -> str:
    q = question.lower().strip()

    # count people
    if re.search(r"\b(how many|count|total)\b", q) and re.search(r"\b(people|persons)\b", q):
        return "count_people"

    # peak people
    if re.search(r"\b(peak|max|most)\b", q) and re.search(r"\b(people|crowd)\b", q):
        return "peak_people"

    # crowd windows (crowded moments)
    if re.search(r"\b(crowded|crowding|dense|packed|busy)\b", q):
        return "crowd_windows"

    #catches: "most crowded moments", "crowd moments", "crowded moments"
    if re.search(r"\b(most\s+crowded|crowded\s+moments|crowd\s+moments|most\s+crowd)\b", q):
        return "crowd_windows"

    #catches: "most crowded moment" (singular)
    if re.search(r"\b(most)\b.*\b(crowded)\b.*\b(moment)\b", q):
        return "crowd_windows"

    #catches: "when was it crowded"
    if re.search(r"\b(when)\b.*\b(crowd|crowded|packed|busy)\b", q):
        return "crowd_windows"

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
