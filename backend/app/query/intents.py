import re


def detect_intent(question: str) -> str:
    q = (question or "").lower().strip()

    # ---------- PEOPLE ----------
    if re.search(r"\b(how many|count|total)\b", q) and re.search(r"\b(people|persons)\b", q):
        return "count_people"

    if re.search(r"\b(peak|max|most)\b", q) and re.search(r"\b(people|crowd)\b", q):
        return "peak_people"

    # ---------- CROWD WINDOWS ----------
    if re.search(r"\b(crowded|crowding|dense|high density|most crowded|busy|packed)\b", q) or \
       re.search(r"\b(when)\b.*\b(crowd|crowded|busy|packed)\b", q) or \
       re.search(r"\b(a lot of people|lots of people)\b", q):
        return "crowd_windows"

    # ---------- DYNAMICS ----------
    if re.search(r"\b(grow|growing|increase|start growing)\b", q) and re.search(r"\b(crowd|people)\b", q):
        return "crowd_growth"

    if re.search(r"\b(drop|decrease|fall)\b", q) and re.search(r"\b(crowd|people)\b", q):
        return "crowd_drop"

    if re.search(r"\b(dynamic|most dynamic)\b", q):
        return "most_dynamic"

    # ---------- HIGHLIGHTS / QUALITY ----------
    if re.search(r"\b(highlight|highlights|best moments|top moments)\b", q):
        return "highlights"

    if re.search(r"\b(quality|tracking quality|tracker quality)\b", q):
        return "quality"

    # ---------- OBJECTS ----------
    # explicit list
    if re.search(r"\b(what)\b.*\b(objects|items)\b.*\b(detected|found|seen)\b", q) or \
       re.search(r"\b(objects detected|detected objects)\b", q):
        return "list_objects"

    # explicit count
    if re.search(r"\b(how many|count|total)\b", q) and re.search(r"\b(objects|items)\b", q):
        return "count_objects"

    # IMPORTANT: allow "How many cars/trucks/buses/..." etc.
    # Heuristic: "how many/count/total" + NOT people/crowd => treat as objects count.
    if re.search(r"\b(how many|count|total)\b", q) and not re.search(r"\b(people|persons|crowd)\b", q):
        return "count_objects"

    # ---------- GENERIC ----------
    if re.search(r"\b(summary|summarize|overview|what happened)\b", q):
        return "summary"

    if re.search(r"\b(enter|entered|exit|exited|events)\b", q):
        return "events"

    if re.search(r"\b(timeline|frames|frame count|timeline count)\b", q):
        return "timeline_info"

    return "unknown"
