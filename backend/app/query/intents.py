import re


def detect_intent(question: str) -> str:
    q = question.lower().strip()

    # counts
    if re.search(r"\b(how many|count|total)\b", q) and re.search(r"\b(people|persons)\b", q):
        return "count_people"

    # peak
    if re.search(r"\b(peak|max|most)\b", q) and re.search(r"\b(people|crowd)\b", q):
        return "peak_people"

    # crowded windows
    if re.search(r"\b(crowded|crowding|dense|packed|busy|most crowded|crowd windows|crowded moments)\b", q):
        return "crowd_windows"

    # crowd growth (covers "start growing")
    if (
        re.search(r"\b(grow|growing|grew|increase|increasing|rise|rising|build up|building up)\b", q)
        and re.search(r"\b(crowd|people)\b", q)
    ):
        return "crowd_growth"

    # crowd drop / dispersal
    if (
        re.search(r"\b(drop|dropping|decrease|decreasing|fell|falling|shrink|shrinking|disperse|dispersing|leave|leaving)\b", q)
        and re.search(r"\b(crowd|people)\b", q)
    ):
        return "crowd_drop"

    # dynamic moment
    if re.search(r"\b(most dynamic|most intense|event burst|highest activity|highlight)\b", q):
        return "most_dynamic"

    # summary / events / timeline
    if re.search(r"\b(summary|summarize|overview|what happened)\b", q):
        return "summary"

    if re.search(r"\b(enter|entered|exit|exited|events)\b", q):
        return "events"

    if re.search(r"\b(timeline|frames|frame count|timeline count)\b", q):
        return "timeline_info"

    # highlights
    if re.search(r"\b(highlights?|interesting moments?|best moments?|top moments?|key moments?)\b", q):
        return "highlights"


    return "unknown"
