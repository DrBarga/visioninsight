# VisionInsight v0.4.2 — Pipeline Stabilization

## Fixed

- Isolated tracker state for every video analysis.
- Added explicit `visible` tracking semantics so retained/ghost tracks are not counted as on-screen people or objects.
- Changed the sampled timeline to retain empty analyzed frames, producing a complete sampled time axis.
- Corrected statistics and tracking-quality calculations when `frame_stride > 1`.
- Unified the backend and artifact version under `0.4.2`.

## Added

- Analysis modes: `fast`, `balanced`, and `full`.
- Optional transcript, CLIP refinement, object analysis, output video, and frame sampling controls.
- Per-stage timings in `summary.json` and `meta.json`.
- Internal YOLO class filtering for people-only and vehicle profiles.
- Cached Whisper and CLIP model instances across sequential analyses.
- API endpoints for meta, transcript, audio, output video, object refinements, and refined object statistics.
- Regression tests for tracker visibility, tracker reset, analysis modes, sampling-aware stats, and sampling-aware quality.

## Behavior changes

- `balanced` no longer runs Whisper or CLIP by default.
- `full` runs the complete multimodal pipeline.
- `fast` defaults to people-only detection, no annotated output video, and `frame_stride=2`.
- `timeline.jsonl` now contains one row for every analyzed frame, including zero-person frames.
