from __future__ import annotations

import json
import math
import asyncio
import shutil
import subprocess
import tempfile
import threading
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.api.v1.auth import current_identity, require_verified
from app.analytics.regions import validate_regions
from app.db.store import QuotaExceeded


router = APIRouter(prefix="/v1", tags=["analyses"])
CLIP_SEMAPHORE = threading.BoundedSemaphore(1)
PLANS = {
    "free": {"monthly_minutes": 30, "max_video_minutes": 10, "max_upload_mb": 100, "storage_gb": 1, "retention_days": 7},
    "developer": {"monthly_minutes": 600, "max_video_minutes": 60, "max_upload_mb": 500, "storage_gb": 20, "retention_days": 30},
    "team": {"monthly_minutes": 3000, "max_video_minutes": 120, "max_upload_mb": 1000, "storage_gb": 100, "retention_days": 90},
}
RECIPES = {
    "quick_scan": {"analysis_mode": "fast", "detection_profile": "balanced",
                   "include_objects": True, "save_output_video": False},
    "event_pulse": {"analysis_mode": "balanced", "include_objects": True, "save_output_video": True},
}
ARTIFACT_MEDIA = {
    "summary.json": "application/json", "stats.json": "application/json",
    "highlights.json": "application/json", "quality.json": "application/json",
    "moments.json": "application/json", "evidence.json": "application/json",
    "objects_stats.json": "application/json", "objects_refined_stats.json": "application/json",
    "timeline.jsonl": "application/x-ndjson", "people.jsonl": "application/x-ndjson",
    "events.jsonl": "application/x-ndjson", "objects.jsonl": "application/x-ndjson",
    "transcript.jsonl": "application/x-ndjson", "output.mp4": "video/mp4",
    "zones.json": "application/json",
}


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def _owned(request: Request, analysis_id: str, identity: Dict[str, Any]) -> Dict[str, Any]:
    job = request.app.state.store.get_analysis(analysis_id)
    if job is None or job["user_id"] != identity["user"]["id"]:
        raise HTTPException(404, "Analysis not found")
    return job


def _completed(request: Request, analysis_id: str, identity: Dict[str, Any]) -> Dict[str, Any]:
    job = _owned(request, analysis_id, identity)
    if job["status"] != "completed":
        raise HTTPException(409, "Analysis is not complete")
    return job


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(404, "Analysis artifact not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _derived_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*")
               if path.is_file() and path.name != "input.mp4")


def _probe_video(path: Path, max_pixels: int, max_fps: int, max_frames: int) -> Tuple[float, float]:
    executable = shutil.which("ffprobe")
    if not executable:
        raise HTTPException(503, "Video inspection requires ffprobe")
    try:
        result = subprocess.run([
            executable, "-v", "error", "-protocol_whitelist", "file",
            "-select_streams", "v:0", "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
            "-of", "json", str(path),
        ], capture_output=True, text=True, timeout=15, check=True)
        metadata = json.loads(result.stdout)
        stream = next(item for item in metadata.get("streams", []) if item.get("codec_type") == "video")
        width, height = int(stream["width"]), int(stream["height"])
        rate = stream.get("avg_frame_rate")
        if not rate or rate == "0/0":
            rate = stream.get("r_frame_rate")
        fps = float(Fraction(rate))
        durations = [float(value) for value in (stream.get("duration"),
                                                 (metadata.get("format") or {}).get("duration")) if value is not None]
        duration = max(durations)
        advertised_frames = int(stream.get("nb_frames") or 0) if str(stream.get("nb_frames") or "0").isdigit() else 0
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError, TypeError,
            KeyError, StopIteration, ZeroDivisionError):
        raise HTTPException(422, "Video cannot be inspected") from None
    if not math.isfinite(fps) or not math.isfinite(duration) or fps <= 0 or duration <= 0 or width <= 0 or height <= 0:
        raise HTTPException(422, "Video metadata is invalid")
    if width * height > max_pixels or fps > max_fps or max(advertised_frames, math.ceil(duration * fps)) > max_frames:
        raise HTTPException(413, "Video resolution, frame rate, or frame count exceeds the service limit")
    return duration, fps


@router.get("/plans")
def plans(request: Request):
    settings = request.app.state.settings
    if settings.environment == "production" and not settings.model_license_cleared:
        raise HTTPException(503, "Video processing is unavailable pending model license clearance")
    return {"legal": {"terms_url": settings.legal_terms_url or None,
                      "privacy_url": settings.legal_privacy_url or None,
                      "contact_email": settings.business_contact_email or None},
            "plans": [
        {"id": name, **limits,
         "display_price": (settings.paddle_developer_display_price if name == "developer" else
                           settings.paddle_team_display_price if name == "team" else "Free"),
         "checkout_available": settings.billing_ready(name)}
        for name, limits in PLANS.items()
    ]}


@router.post("/analyses", status_code=202)
async def create_analysis(
    request: Request,
    recipe: str = "quick_scan",
    zones: Optional[str] = None,
    lines: Optional[str] = None,
    identity: Dict[str, Any] = Depends(current_identity),
):
    if recipe not in RECIPES:
        raise HTTPException(422, "Unknown analysis recipe")
    require_verified(identity)
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "video/mp4":
        raise HTTPException(422, "Upload an MP4 video")
    regions: Dict[str, Any] = {}
    for field, value in (("zones", zones), ("lines", lines)):
        if value:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                raise HTTPException(422, f"Invalid {field} JSON") from None
            if not isinstance(parsed, list) or len(parsed) > 20:
                raise HTTPException(422, f"{field} must be a list of up to 20 items")
            regions[field] = parsed
    try:
        regions = validate_regions(regions)
    except (TypeError, ValueError) as error:
        raise HTTPException(422, str(error)) from error

    settings = request.app.state.settings
    if settings.environment == "production" and not settings.model_license_cleared:
        raise HTTPException(503, "Video processing is unavailable pending model license clearance")
    user = identity["user"]
    if not request.app.state.store.allow_request("upload:" + user["id"], 3):
        raise HTTPException(429, "Too many video uploads")
    plan = PLANS.get(user["plan"], PLANS["free"])
    upload_limit = min(settings.max_upload_bytes, plan["max_upload_mb"] * 1024 * 1024)
    declared_size = request.headers.get("content-length")
    if declared_size and declared_size.isdigit() and int(declared_size) > upload_limit:
        raise HTTPException(413, "Video exceeds upload limit")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(settings.data_dir).free < settings.min_free_disk_bytes:
        raise HTTPException(507, "Video storage is temporarily unavailable")
    with tempfile.TemporaryDirectory(dir=settings.data_dir) as temp:
        upload_path = Path(temp) / "upload.mp4"
        uploaded = 0
        with upload_path.open("wb") as output:
            async for chunk in request.stream():
                uploaded += len(chunk)
                if uploaded > upload_limit:
                    raise HTTPException(413, "Video exceeds upload limit")
                output.write(chunk)
                if uploaded % (8 * 1024 * 1024) < len(chunk) and \
                        shutil.disk_usage(settings.data_dir).free < settings.min_free_disk_bytes:
                    raise HTTPException(507, "Video storage is temporarily unavailable")
        if uploaded == 0:
            raise HTTPException(422, "Video is empty")
        duration_sec, source_fps = await asyncio.to_thread(
            _probe_video, upload_path, settings.max_video_pixels,
            settings.max_video_fps, settings.max_video_frames,
        )
        if duration_sec > min(settings.max_duration_sec, plan["max_video_minutes"] * 60):
            raise HTTPException(413, "Video exceeds the plan duration limit")
        recipe_settings = RECIPES[recipe]
        minutes = max(1, math.ceil(duration_sec / 60))
        options = recipe_settings.copy()
        options["regions"] = regions
        options["max_source_frames"] = min(
            settings.max_video_frames,
            math.ceil(duration_sec * source_fps * 1.1) + math.ceil(source_fps * 2),
            plan["max_video_minutes"] * 60 * settings.max_video_fps,
        )
        options["max_wall_seconds"] = settings.max_processing_seconds
        options["max_source_pixels"] = settings.max_video_pixels
        options["max_source_fps"] = settings.max_video_fps
        options["max_derived_bytes"] = settings.max_derived_bytes
        options["min_free_disk_bytes"] = settings.min_free_disk_bytes
        try:
            job = request.app.state.store.create_analysis(
                user["id"], recipe, options, minutes, plan["monthly_minutes"],
                uploaded, plan["storage_gb"] * 1024 * 1024 * 1024,
            )
        except QuotaExceeded as error:
            raise HTTPException(402, str(error)) from error
        run_dir = request.app.state.storage.run_dir(job["id"])
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            shutil.move(str(upload_path), str(run_dir / "input.mp4"))
            request.app.state.store.enqueue(job["id"])
        except Exception:
            request.app.state.store.finish(job["id"], "failed", error_code="UPLOAD_STORAGE_FAILED",
                                           error_message="Could not save uploaded video")
            raise HTTPException(500, "Could not save uploaded video") from None
    return {"schema_version": "1.0", "analysis_id": job["id"], "status": "queued",
            "status_url": f"/v1/analyses/{job['id']}"}


@router.get("/analyses")
def list_analyses(request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    return {"schema_version": "1.0", "analyses": request.app.state.store.list_analyses(identity["user"]["id"])}


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str, request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    return {"schema_version": "1.0", **_owned(request, analysis_id, identity)}


@router.post("/analyses/{analysis_id}/cancel")
def cancel_analysis(analysis_id: str, request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    _owned(request, analysis_id, identity)
    request.app.state.store.cancel(identity["user"]["id"], analysis_id)
    return {"status": request.app.state.store.get_analysis(analysis_id)["status"]}


@router.delete("/analyses/{analysis_id}")
def delete_analysis(analysis_id: str, request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    job = _owned(request, analysis_id, identity)
    if job["status"] == "queued":
        request.app.state.store.cancel(identity["user"]["id"], analysis_id)
        job = request.app.state.store.get_analysis(analysis_id)
    if job["status"] not in ("completed", "failed", "cancelled"):
        raise HTTPException(409, "Wait for the analysis to stop before deletion")
    request.app.state.storage.delete(analysis_id)
    if not request.app.state.store.delete_analysis(identity["user"]["id"], analysis_id):
        raise HTTPException(409, "Analysis state changed during deletion")
    return {"deleted": True}


@router.get("/analyses/{analysis_id}/overview")
def overview(analysis_id: str, request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    job = _completed(request, analysis_id, identity)
    directory = request.app.state.storage.run_dir(analysis_id)
    summary = _read_json(directory / "summary.json")
    return {
        "schema_version": "1.0", "analysis_id": analysis_id, "recipe": job["recipe"],
        "source": summary.get("source"), "tracks_summary": summary.get("tracks_summary"),
        "objects_summary": summary.get("objects_summary"),
        "stats": _read_json(directory / "stats.json"),
        "zones": _read_json(directory / "zones.json"),
        "quality": _read_json(directory / "quality.json").get("quality_summary"),
        "moments": _read_json(directory / "moments.json").get("moments", [])[:5],
    }


@router.get("/analyses/{analysis_id}/moments")
def moments(analysis_id: str, request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    _completed(request, analysis_id, identity)
    return _read_json(request.app.state.storage.artifact(analysis_id, "moments.json"))


@router.get("/analyses/{analysis_id}/metrics")
def metrics(analysis_id: str, request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    _completed(request, analysis_id, identity)
    return _read_json(request.app.state.storage.artifact(analysis_id, "stats.json"))


@router.get("/analyses/{analysis_id}/evidence/{evidence_id}")
def evidence(analysis_id: str, evidence_id: str, request: Request,
             identity: Dict[str, Any] = Depends(current_identity)):
    _completed(request, analysis_id, identity)
    cards = _read_json(request.app.state.storage.artifact(analysis_id, "evidence.json")).get("evidence", [])
    card = next((item for item in cards if item["id"] == evidence_id), None)
    if card is None:
        raise HTTPException(404, "Evidence not found")
    return {"schema_version": "1.0", **card}


@router.get("/analyses/{analysis_id}/evidence/{evidence_id}/image")
def evidence_image(analysis_id: str, evidence_id: str, request: Request,
                   identity: Dict[str, Any] = Depends(current_identity)):
    card = evidence(analysis_id, evidence_id, request, identity)
    if not card.get("image"):
        raise HTTPException(404, "Evidence image unavailable")
    path = request.app.state.storage.run_dir(analysis_id) / "evidence" / f"{evidence_id}.jpg"
    if not path.is_file():
        raise HTTPException(404, "Evidence image unavailable")
    return FileResponse(path, media_type="image/jpeg", content_disposition_type="inline")


@router.get("/analyses/{analysis_id}/video")
def video(analysis_id: str, request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    _completed(request, analysis_id, identity)
    path = request.app.state.storage.artifact(analysis_id, "input.mp4")
    if not path.is_file():
        raise HTTPException(404, "Video unavailable")
    return FileResponse(path, media_type="video/mp4", filename="input.mp4",
                        content_disposition_type="inline")


@router.get("/analyses/{analysis_id}/artifacts")
def artifacts(analysis_id: str, request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    _completed(request, analysis_id, identity)
    directory = request.app.state.storage.run_dir(analysis_id)
    return {"artifacts": [name for name in ARTIFACT_MEDIA if (directory / name).is_file()]}


@router.get("/analyses/{analysis_id}/artifacts/{name}")
def artifact(analysis_id: str, name: str, request: Request,
             identity: Dict[str, Any] = Depends(current_identity)):
    _completed(request, analysis_id, identity)
    if name not in ARTIFACT_MEDIA:
        raise HTTPException(404, "Artifact not found")
    path = request.app.state.storage.artifact(analysis_id, name)
    if not path.is_file():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(path, media_type=ARTIFACT_MEDIA[name], filename=name)


@router.get("/analyses/{analysis_id}/report")
def report(analysis_id: str, request: Request, identity: Dict[str, Any] = Depends(current_identity)):
    _completed(request, analysis_id, identity)
    directory = request.app.state.storage.run_dir(analysis_id)
    payload = {"schema_version": "1.0", "overview": overview(analysis_id, request, identity),
               "moments": _read_json(directory / "moments.json")["moments"],
               "evidence": _read_json(directory / "evidence.json")["evidence"]}
    return JSONResponse(payload, headers={"Content-Disposition": "attachment; filename=visioninsight-report.json"})


@router.get("/analyses/{analysis_id}/clip")
def clip(analysis_id: str, start_sec: float, end_sec: float, request: Request,
         identity: Dict[str, Any] = Depends(current_identity)):
    job = _completed(request, analysis_id, identity)
    if not math.isfinite(start_sec) or not math.isfinite(end_sec):
        raise HTTPException(422, "Clip timestamps must be finite")
    start_sec, end_sec = round(start_sec, 1), round(end_sec, 1)
    if start_sec < 0 or end_sec - start_sec < 0.1 or end_sec - start_sec > 30:
        raise HTTPException(422, "Clip must be between 0.1 and 30 seconds")
    if job["duration_sec"] is not None and end_sec > job["duration_sec"] + 0.5:
        raise HTTPException(422, "Clip extends beyond video")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise HTTPException(503, "Clip export requires ffmpeg and ffprobe")
    directory = request.app.state.storage.run_dir(analysis_id)
    output = directory / "clips" / f"{round(start_sec * 10)}-{round(end_sec * 10)}.mp4"
    if not output.exists():
        if not CLIP_SEMAPHORE.acquire(blocking=False):
            raise HTTPException(429, "Clip export is busy; try again shortly")
        temporary = output.with_name(f"{uuid.uuid4()}.tmp.mp4")
        try:
            if output.exists():
                return FileResponse(output, media_type="video/mp4", filename="clip.mp4")
            output.parent.mkdir(parents=True, exist_ok=True)
            if len(list(output.parent.glob("*.mp4"))) >= 20:
                raise HTTPException(429, "Clip export limit reached for this analysis")
            remaining = request.app.state.settings.max_derived_bytes - _derived_bytes(directory)
            if remaining < 1024 * 1024:
                raise HTTPException(507, "Analysis output storage limit reached")
            clip_limit = min(remaining, 100 * 1024 * 1024)
            subprocess.run([
                ffmpeg, "-nostdin", "-v", "error", "-ss", str(start_sec),
                "-i", str(directory / "input.mp4"), "-t", str(end_sec - start_sec),
                "-c:v", "libx264", "-c:a", "aac", "-fs", str(clip_limit),
                "-y", str(temporary),
            ], timeout=180, check=True, capture_output=True)
            probe = subprocess.run([
                ffprobe, "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                str(temporary),
            ], timeout=15, check=True, capture_output=True, text=True)
            clip_duration = float(probe.stdout.strip())
            if not math.isfinite(clip_duration) or clip_duration < end_sec - start_sec - 0.5:
                raise HTTPException(507, "Clip exceeds the output size limit")
            if _derived_bytes(directory) > request.app.state.settings.max_derived_bytes:
                raise HTTPException(507, "Analysis output storage limit reached")
            temporary.replace(output)
        except HTTPException:
            temporary.unlink(missing_ok=True)
            raise
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            temporary.unlink(missing_ok=True)
            raise HTTPException(500, "Could not export clip") from None
        finally:
            CLIP_SEMAPHORE.release()
    return FileResponse(output, media_type="video/mp4", filename="clip.mp4")


@router.post("/analyses/{analysis_id}/query")
def query(analysis_id: str, payload: Question, request: Request,
          identity: Dict[str, Any] = Depends(current_identity)):
    _completed(request, analysis_id, identity)
    from app.query.v1 import answer_with_evidence

    return answer_with_evidence(request.app.state.storage.run_dir(analysis_id), payload.question)
