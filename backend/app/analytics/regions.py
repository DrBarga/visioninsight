from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


Point = Tuple[float, float]
IDENTIFIER = re.compile(r"^[a-zA-Z0-9_-]{1,40}$")


def _point(raw: Any) -> Point:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("Region points must be [x, y]")
    x, y = float(raw[0]), float(raw[1])
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError("Region coordinates must be between 0 and 1")
    return x, y


def validate_regions(regions: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"zones": [], "lines": []}
    for category in ("zones", "lines"):
        items = regions.get(category) or []
        if not isinstance(items, list) or len(items) > 20:
            raise ValueError(f"{category} must contain at most 20 items")
        seen = set()
        for item in items:
            if not isinstance(item, dict) or not IDENTIFIER.fullmatch(str(item.get("id", ""))):
                raise ValueError(f"Each {category} item needs a short alphanumeric id")
            if item["id"] in seen:
                raise ValueError(f"Duplicate {category} id")
            seen.add(item["id"])
            if category == "zones":
                points = [_point(point) for point in item.get("points", [])]
                if len(points) < 3 or len(points) > 32:
                    raise ValueError("Zone needs 3 to 32 points")
                result[category].append({"id": item["id"], "points": points})
            else:
                start, end = _point(item.get("start")), _point(item.get("end"))
                if start == end:
                    raise ValueError("Line endpoints must differ")
                result[category].append({"id": item["id"], "start": start, "end": end})
    return result


def _inside(point: Point, polygon: List[Point]) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            intersection = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < intersection:
                inside = not inside
        previous = current
    return inside


def _side(point: Point, start: Point, end: Point) -> float:
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])


def _segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    return _side(a, c, d) * _side(b, c, d) < 0 and _side(c, a, b) * _side(d, a, b) < 0


@dataclass
class RegionAnalyzer:
    regions: Dict[str, Any]
    width: int
    height: int
    fps: float
    frame_stride: int = 1
    previous_points: Dict[int, Point] = field(default_factory=dict)
    zone_observations: Dict[Tuple[str, int], int] = field(default_factory=dict)
    zone_peak: Dict[str, int] = field(default_factory=dict)
    line_events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.regions = validate_regions(self.regions)

    def update(self, frame: int, time_sec: float, people: List[Dict[str, Any]]) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        current: Dict[int, Point] = {}
        for person in people:
            bbox = person.get("bbox") or []
            if len(bbox) != 4:
                continue
            track_id = int(person["track_id"])
            current[track_id] = ((float(bbox[0]) + float(bbox[2])) / (2 * self.width),
                                 float(bbox[3]) / self.height)

        occupancy: Dict[str, int] = {}
        for zone in self.regions["zones"]:
            zone_id = zone["id"]
            inside = [track_id for track_id, point in current.items() if _inside(point, zone["points"])]
            occupancy[zone_id] = len(inside)
            self.zone_peak[zone_id] = max(self.zone_peak.get(zone_id, 0), len(inside))
            for track_id in inside:
                key = (zone_id, track_id)
                self.zone_observations[key] = self.zone_observations.get(key, 0) + 1

        events: List[Dict[str, Any]] = []
        for line in self.regions["lines"]:
            start, end = line["start"], line["end"]
            for track_id, point in current.items():
                previous = self.previous_points.get(track_id)
                if previous is None or not _segments_cross(previous, point, start, end):
                    continue
                direction = "in" if _side(point, start, end) > 0 else "out"
                event = {"type": "line_crossed", "line_id": line["id"], "direction": direction,
                         "track_id": track_id, "frame": frame, "time_sec": time_sec}
                events.append(event)
                self.line_events.append(event)
        self.previous_points = current
        return occupancy, events

    def summary(self) -> Dict[str, Any]:
        zone_summary = []
        for zone in self.regions["zones"]:
            zone_id = zone["id"]
            dwell = [
                round(observations * self.frame_stride / self.fps, 2)
                for (name, _track_id), observations in self.zone_observations.items() if name == zone_id
            ]
            zone_summary.append({
                "id": zone_id, "peak_occupancy": self.zone_peak.get(zone_id, 0),
                "observed_tracks": len(dwell),
                "average_observed_dwell_sec": round(sum(dwell) / len(dwell), 2) if dwell else 0.0,
            })
        return {"schema_version": "1.0", "zones": zone_summary,
                "lines": self.regions["lines"], "line_crossings": self.line_events}
