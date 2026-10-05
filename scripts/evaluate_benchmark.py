from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def match_crossings(predicted: list[dict], expected: list[dict], tolerance_sec: float) -> int:
    used: set[int] = set()
    matches = 0
    for truth in expected:
        candidates = [
            (index, abs(float(item["time_sec"]) - float(truth["time_sec"])))
            for index, item in enumerate(predicted)
            if index not in used and item["line_id"] == truth["line_id"]
            and item["direction"] == truth["direction"]
        ]
        if candidates:
            index, distance = min(candidates, key=lambda pair: pair[1])
            if distance <= tolerance_sec:
                used.add(index)
                matches += 1
    return matches


def evaluate(manifest: dict[str, Any], tolerance_sec: float = 1.0) -> dict[str, Any]:
    cases = manifest.get("cases") or []
    if not cases:
        raise ValueError("Benchmark manifest needs at least one labeled case")
    results = []
    peak_errors = []
    predicted_crossings = expected_crossings = matched_crossings = 0
    expected_moments = matched_moments = 0
    for case in cases:
        directory = Path(case["run_dir"])
        truth = case["truth"]
        stats = read_json(directory / "stats.json")
        zones = read_json(directory / "zones.json")
        moments = read_json(directory / "moments.json")["moments"]
        summary = read_json(directory / "summary.json")
        peak = int(stats["people_count"]["max"])
        error = abs(peak - int(truth["peak_visible_people"]))
        peak_errors.append(error)
        actual_crossings = zones.get("line_crossings") or []
        expected = truth.get("line_crossings") or []
        matched = match_crossings(actual_crossings, expected, tolerance_sec)
        predicted_crossings += len(actual_crossings)
        expected_crossings += len(expected)
        matched_crossings += matched
        intervals = truth.get("notable_intervals") or []
        moment_hits = sum(any(
            max(float(moment["start_sec"]), float(interval[0])) <=
            min(float(moment["end_sec"]), float(interval[1]))
            for moment in moments
        ) for interval in intervals)
        expected_moments += len(intervals)
        matched_moments += moment_hits
        duration = float(summary["source"]["duration_sec_est"] or 0)
        elapsed = float(summary["timings"]["total_sec"] or 0)
        results.append({
            "name": case["name"], "peak_absolute_error": error,
            "crossings": {"matched": matched, "predicted": len(actual_crossings), "expected": len(expected)},
            "notable_intervals_recalled": moment_hits,
            "notable_intervals_total": len(intervals),
            "processing_to_video_ratio": round(elapsed / duration, 3) if duration else None,
        })
    if not predicted_crossings and not expected_crossings:
        precision = recall = f1 = None
    else:
        precision = matched_crossings / predicted_crossings if predicted_crossings else 0.0
        recall = matched_crossings / expected_crossings if expected_crossings else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "schema_version": "1.0", "cases": results,
        "peak_people_mae": round(sum(peak_errors) / len(peak_errors), 3),
        "line_crossing_precision": round(precision, 3) if precision is not None else None,
        "line_crossing_recall": round(recall, 3) if recall is not None else None,
        "line_crossing_f1": round(f1, 3) if f1 is not None else None,
        "notable_interval_recall": round(matched_moments / expected_moments, 3) if expected_moments else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate annotated VisionInsight analyses")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--crossing-tolerance-sec", type=float, default=1.0)
    parser.add_argument("--max-peak-mae", type=float)
    parser.add_argument("--min-crossing-f1", type=float)
    parser.add_argument("--min-moment-recall", type=float)
    args = parser.parse_args()
    report = evaluate(read_json(args.manifest), args.crossing_tolerance_sec)
    print(json.dumps(report, indent=2))
    failures = []
    if args.max_peak_mae is not None and report["peak_people_mae"] > args.max_peak_mae:
        failures.append("peak people MAE")
    if args.min_crossing_f1 is not None and (report["line_crossing_f1"] is None or
                                              report["line_crossing_f1"] < args.min_crossing_f1):
        failures.append("line crossing F1")
    if args.min_moment_recall is not None and (report["notable_interval_recall"] is None or
                                               report["notable_interval_recall"] < args.min_moment_recall):
        failures.append("notable interval recall")
    if failures:
        parser.exit(1, "Benchmark gates failed: " + ", ".join(failures) + "\n")


if __name__ == "__main__":
    main()
