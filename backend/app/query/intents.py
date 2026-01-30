import re


def detect_intent(question: str) -> str:
    q = (question or "").lower().strip()

    # summary / overview
    if re.search(r"\b(summary|overview|summarize|сводка|итог|обзор)\b", q):
        return "summary"

    # quality
    if re.search(r"\b(quality|tracking quality|качество|качество трекинга|качество отслеживания)\b", q):
        return "quality"

    # highlights
    if re.search(r"\b(highlights?|key moments|best moments|важные моменты|ключевые моменты|хайлайты)\b", q):
        return "highlights"

    # events
    if re.search(r"\b(events|entered|exited|enter|exit|события|вход|выход|заш[её]л|вышел)\b", q):
        return "events"

    # crowd windows
    if re.search(r"\b(when)\b.*\b(crowd|crowded|busy|dense)\b", q) or re.search(r"\b(людно|много людей|толпа)\b", q):
        return "crowd_windows"

    # dynamics
    if re.search(r"\b(grow|growing|increase|start growing|рост|увелич|начал расти)\b", q) and re.search(r"\b(crowd|people|люди|толпа)\b", q):
        return "crowd_growth"
    if re.search(r"\b(drop|decrease|fall|decline|паден|уменьш|спал)\b", q) and re.search(r"\b(crowd|people|люди|толпа)\b", q):
        return "crowd_drop"
    if re.search(r"\b(most dynamic|dynamic|burst|всплеск|динамич)\b", q):
        return "most_dynamic"

    # people
    if re.search(r"\b(peak|max|most|пик|максимум)\b", q) and re.search(r"\b(people|person|crowd|люди|человек|толпа)\b", q):
        return "peak_people"
    if re.search(r"\b(how many|count|total|сколько|количество|всего)\b", q) and re.search(r"\b(people|person|persons|люди|человек)\b", q):
        return "count_people"

    # objects: peak
    if re.search(r"\b(peak|max|most|пик|максимум)\b", q) and re.search(r"\b(objects|items|объект|объекты|предмет|предметы)\b", q):
        return "peak_objects"
    if re.search(r"\b(peak)\b.*\b(cars?|trucks?|buses?|trains?)\b", q):
        return "peak_objects"
    if re.search(r"\b(пик)\b.*\b(машин|машины|авто|поезд|поезда|автобус|грузовик)\b", q):
        return "peak_objects"

    # objects: list
    if re.search(r"\b(what|which|list)\b.*\b(objects|items)\b", q) or re.search(r"\b(какие)\b.*\b(объект|объекты|предмет|предметы)\b", q):
        return "list_objects"
    if re.search(r"\b(objects detected|detected objects|what objects were detected)\b", q):
        return "list_objects"

    # objects: count
    if re.search(r"\b(how many|count|total|сколько|количество|всего)\b", q):
        # if explicitly about people, handled above; otherwise treat as objects count
        if not re.search(r"\b(people|person|persons|люди|человек|толпа)\b", q):
            return "count_objects"

    return "unknown"
