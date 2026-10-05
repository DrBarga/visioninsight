from __future__ import annotations

import shutil
import uuid
from pathlib import Path


class LocalStorage:
    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, analysis_id: str) -> Path:
        canonical = str(uuid.UUID(analysis_id))
        if canonical != analysis_id:
            raise ValueError("Invalid analysis ID")
        return self.runs_dir / canonical

    def delete(self, analysis_id: str) -> None:
        directory = self.run_dir(analysis_id)
        if directory.is_dir():
            shutil.rmtree(directory)

    def artifact(self, analysis_id: str, name: str) -> Path:
        if name not in {
            "input.mp4", "output.mp4", "meta.json", "summary.json", "stats.json",
            "highlights.json", "quality.json", "timeline.jsonl", "people.jsonl",
            "events.jsonl", "objects.jsonl", "objects_stats.json", "transcript.jsonl",
            "audio.wav", "object_refinements.json", "objects_refined_stats.json",
            "moments.json", "evidence.json", "zones.json",
        }:
            raise ValueError("Unknown artifact")
        return self.run_dir(analysis_id) / name
