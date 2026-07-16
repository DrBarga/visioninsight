from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AnalysisMode:
    name: str
    detection_profile: str
    include_objects: bool
    enable_transcript: bool
    enable_object_refinement: bool
    save_output_video: bool
    frame_stride: int


ANALYSIS_MODES: Dict[str, AnalysisMode] = {
    "fast": AnalysisMode(
        name="fast",
        detection_profile="people_strict",
        include_objects=False,
        enable_transcript=False,
        enable_object_refinement=False,
        save_output_video=False,
        frame_stride=2,
    ),
    "balanced": AnalysisMode(
        name="balanced",
        detection_profile="balanced",
        include_objects=True,
        enable_transcript=False,
        enable_object_refinement=False,
        save_output_video=True,
        frame_stride=1,
    ),
    "full": AnalysisMode(
        name="full",
        detection_profile="balanced",
        include_objects=True,
        enable_transcript=True,
        enable_object_refinement=True,
        save_output_video=True,
        frame_stride=1,
    ),
}


@dataclass(frozen=True)
class ResolvedAnalysisOptions:
    mode: str
    detection_profile: str
    include_objects: bool
    enable_transcript: bool
    enable_object_refinement: bool
    save_output_video: bool
    frame_stride: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _pick(override: Optional[Any], default: Any) -> Any:
    return default if override is None else override


def resolve_analysis_options(
    mode: str = "balanced",
    detection_profile: Optional[str] = None,
    include_objects: Optional[bool] = None,
    enable_transcript: Optional[bool] = None,
    enable_object_refinement: Optional[bool] = None,
    save_output_video: Optional[bool] = None,
    frame_stride: Optional[int] = None,
) -> ResolvedAnalysisOptions:
    mode_name = (mode or "balanced").strip().lower()
    if mode_name not in ANALYSIS_MODES:
        valid = ", ".join(sorted(ANALYSIS_MODES))
        raise ValueError(f"Unknown analysis mode '{mode}'. Expected one of: {valid}")

    base = ANALYSIS_MODES[mode_name]
    stride = int(_pick(frame_stride, base.frame_stride))
    if stride < 1 or stride > 10:
        raise ValueError("frame_stride must be between 1 and 10")

    resolved_include_objects = bool(_pick(include_objects, base.include_objects))
    resolved_refinement = bool(_pick(enable_object_refinement, base.enable_object_refinement))

    # CLIP refinement requires object tracks. Make the dependency explicit and safe.
    if resolved_refinement:
        resolved_include_objects = True

    return ResolvedAnalysisOptions(
        mode=mode_name,
        detection_profile=(detection_profile or base.detection_profile).strip(),
        include_objects=resolved_include_objects,
        enable_transcript=bool(_pick(enable_transcript, base.enable_transcript)),
        enable_object_refinement=resolved_refinement,
        save_output_video=bool(_pick(save_output_video, base.save_output_video)),
        frame_stride=stride,
    )
