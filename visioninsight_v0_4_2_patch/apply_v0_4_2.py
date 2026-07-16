from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPLACEMENTS = [
    "backend/app/version.py",
    "backend/app/main.py",
    "backend/app/video/analysis_profiles.py",
    "backend/app/video/processor.py",
    "backend/app/tracking/iou_tracker.py",
    "backend/app/detection/yolo.py",
    "backend/app/analytics/stats_builder.py",
    "backend/app/analytics/quality_builder.py",
    "backend/app/analytics/transcript_builder.py",
    "backend/tests/test_pipeline_stabilization.py",
    ".gitignore",
    "requirements.txt",
    "requirements-dev.txt",
]

OLD_OBJECT_BLOCK = '''    # ---------- OBJECTS ----------
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
'''

NEW_OBJECT_BLOCK = '''    # ---------- OBJECTS ----------
    if intent in ("count_objects", "list_objects"):
        refined_available = bool(obj_ref_stats and obj_ref_stats.get("available"))
        refined_labels = (obj_ref_stats or {}).get("unique_by_label") or {}

        if refined_available and refined_labels:
            unique_total = int(obj_ref_stats.get("unique_total", 0))
            unique_by_class = refined_labels
            source_name = "objects_refined_stats.json"
            label_kind = "refined_label"
        elif obj_stats:
            unique_total = int(obj_stats.get("unique_total", 0))
            unique_by_class = obj_stats.get("unique_by_class") or {}
            source_name = "objects_stats.json"
            label_kind = "class_name"
        else:
            return (intent, "Objects stats not found. Run analysis again with object analysis enabled.", {}, 0.4)

        available_classes = list(unique_by_class.keys())
        matched = _match_object_class_from_question(question, available_classes)

        if intent == "count_objects":
            if matched:
                count = int(unique_by_class.get(matched, 0))
                answer = f"Total unique '{matched}' detected: {count}."
                evidence = {label_kind: matched, "unique": count, "source": source_name}
                return (intent, answer, evidence, 0.92 if refined_available else 0.9)

            answer = f"Total unique objects detected: {unique_total}."
            evidence = {
                "unique_total": unique_total,
                "unique_by_label_top": dict(list(unique_by_class.items())[:10]),
                "source": source_name,
                "refined": refined_available,
            }
            return (intent, answer, evidence, 0.92 if refined_available else 0.9)

        if not unique_by_class:
            return (intent, "No objects were detected (non-person classes).", {"unique_by_label": {}}, 0.85)

        top = list(unique_by_class.items())[:10]
        pretty = ", ".join([f"{name}={count}" for name, count in top])
        prefix = "Refined object labels" if refined_available else "Detected object classes"
        answer = f"{prefix} (top): {pretty}."
        return (
            intent,
            answer,
            {"unique_by_label": unique_by_class, "source": source_name, "refined": refined_available},
            0.92 if refined_available else 0.9,
        )
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply VisionInsight v0.4.2 stabilization patch")
    parser.add_argument("repo", nargs="?", default=".", help="Path to the VisionInsight repository root")
    args = parser.parse_args()

    bundle_root = Path(__file__).resolve().parent
    repo_root = Path(args.repo).resolve()
    if not (repo_root / "backend" / "app").exists():
        raise SystemExit(f"Not a VisionInsight repository root: {repo_root}")

    backup_root = repo_root / ".patch_backups" / "v0.4.2"
    copied = 0
    for relative in REPLACEMENTS:
        source = bundle_root / relative
        destination = repo_root / relative
        if destination.exists():
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1

    engine_path = repo_root / "backend" / "app" / "query" / "engine.py"
    engine_patched = False
    if engine_path.exists():
        content = engine_path.read_text(encoding="utf-8")
        if OLD_OBJECT_BLOCK in content:
            backup = backup_root / "backend/app/query/engine.py"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(engine_path, backup)
            engine_path.write_text(content.replace(OLD_OBJECT_BLOCK, NEW_OBJECT_BLOCK), encoding="utf-8")
            engine_patched = True
        elif NEW_OBJECT_BLOCK in content:
            engine_patched = True
            print("query/engine.py already contains the refined-object patch")
        else:
            print("WARNING: query/engine.py object block differed; it was left unchanged")

    print(f"Copied {copied} replacement files")
    print(f"Query engine refined-object patch: {'applied' if engine_patched else 'not applied'}")
    print(f"Backups: {backup_root}")
    print("Next: cd backend && python -m unittest discover -s tests -v")


if __name__ == "__main__":
    main()
