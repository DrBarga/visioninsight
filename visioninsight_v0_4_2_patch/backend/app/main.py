from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from app.query.engine import answer_question
from app.query.schemas import AskRequest, AskResponse
from app.version import __version__
from app.video.processor import VideoProcessor

app = FastAPI(title="VisionInsight API", version=__version__)

RUNS_DIR = Path("runs")
processor = VideoProcessor(runs_dir=str(RUNS_DIR), model_path="yolov8n.pt")


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


@app.get("/")
def root():
    return PlainTextResponse("VisionInsight API is running. Go to /docs")


@app.post("/analyze-video/")
async def analyze_video(
    file: UploadFile = File(...),
    mode: str = Form("balanced"),
    profile: Optional[str] = Form(None),
    include_objects: Optional[bool] = Form(None),
    enable_transcript: Optional[bool] = Form(None),
    enable_object_refinement: Optional[bool] = Form(None),
    save_output_video: Optional[bool] = Form(None),
    frame_stride: Optional[int] = Form(None),
    transcript_backend: str = Form("auto"),
    transcript_model: str = Form("base"),
    transcript_language: Optional[str] = Form(None),
):
    analysis_id = str(uuid.uuid4())
    run_dir = RUNS_DIR / analysis_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / "input.mp4"
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = processor.process(
            str(input_path),
            analysis_id=analysis_id,
            analysis_mode=mode,
            detection_profile=profile,
            include_objects=include_objects,
            enable_transcript=enable_transcript,
            enable_object_refinement=enable_object_refinement,
            save_output_video=save_output_video,
            frame_stride=frame_stride,
            transcript_backend=transcript_backend,
            transcript_model=transcript_model,
            transcript_language=transcript_language,
        )
        return JSONResponse(content=result)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


def _file_or_404(path: Path, detail: str, media_type: str, filename: str):
    if not path.exists():
        raise HTTPException(status_code=404, detail=detail)
    return FileResponse(str(path), media_type=media_type, filename=filename)


@app.get("/analysis/{analysis_id}/meta")
def get_meta(analysis_id: str):
    return _file_or_404(RUNS_DIR / analysis_id / "meta.json", "meta not found", "application/json", "meta.json")


@app.get("/analysis/{analysis_id}/summary")
def get_summary(analysis_id: str):
    return _file_or_404(RUNS_DIR / analysis_id / "summary.json", "summary not found", "application/json", "summary.json")


@app.get("/analysis/{analysis_id}/timeline")
def get_timeline(analysis_id: str):
    return _file_or_404(RUNS_DIR / analysis_id / "timeline.jsonl", "timeline not found", "application/x-ndjson", "timeline.jsonl")


@app.get("/analysis/{analysis_id}/people")
def get_people(analysis_id: str):
    return _file_or_404(RUNS_DIR / analysis_id / "people.jsonl", "people not found", "application/x-ndjson", "people.jsonl")


@app.get("/analysis/{analysis_id}/events")
def get_events(analysis_id: str):
    return _file_or_404(RUNS_DIR / analysis_id / "events.jsonl", "events not found", "application/x-ndjson", "events.jsonl")


@app.get("/analysis/{analysis_id}/stats")
def get_stats(analysis_id: str):
    return _file_or_404(RUNS_DIR / analysis_id / "stats.json", "stats not found", "application/json", "stats.json")


@app.get("/analysis/{analysis_id}/highlights")
def get_highlights(analysis_id: str):
    return _file_or_404(RUNS_DIR / analysis_id / "highlights.json", "highlights not found", "application/json", "highlights.json")


@app.get("/analysis/{analysis_id}/quality")
def get_quality(analysis_id: str):
    return _file_or_404(RUNS_DIR / analysis_id / "quality.json", "quality not found", "application/json", "quality.json")


@app.get("/analysis/{analysis_id}/objects")
def get_objects(analysis_id: str):
    return _file_or_404(RUNS_DIR / analysis_id / "objects.jsonl", "objects not found", "application/x-ndjson", "objects.jsonl")


@app.get("/analysis/{analysis_id}/objects-stats")
def get_objects_stats(analysis_id: str):
    return _file_or_404(RUNS_DIR / analysis_id / "objects_stats.json", "objects_stats not found", "application/json", "objects_stats.json")


@app.get("/analysis/{analysis_id}/object-refinements")
def get_object_refinements(analysis_id: str):
    return _file_or_404(
        RUNS_DIR / analysis_id / "object_refinements.json",
        "object refinements not found",
        "application/json",
        "object_refinements.json",
    )


@app.get("/analysis/{analysis_id}/objects-refined-stats")
def get_objects_refined_stats(analysis_id: str):
    return _file_or_404(
        RUNS_DIR / analysis_id / "objects_refined_stats.json",
        "objects refined stats not found",
        "application/json",
        "objects_refined_stats.json",
    )


@app.get("/analysis/{analysis_id}/transcript")
def get_transcript(analysis_id: str):
    return _file_or_404(
        RUNS_DIR / analysis_id / "transcript.jsonl",
        "transcript not found or transcript was disabled",
        "application/x-ndjson",
        "transcript.jsonl",
    )


@app.get("/analysis/{analysis_id}/audio")
def get_audio(analysis_id: str):
    return _file_or_404(
        RUNS_DIR / analysis_id / "audio.wav",
        "audio not found or transcript was disabled",
        "audio/wav",
        "audio.wav",
    )


@app.get("/analysis/{analysis_id}/output-video")
def get_output_video(analysis_id: str):
    return _file_or_404(
        RUNS_DIR / analysis_id / "output.mp4",
        "output video not found or save_output_video was disabled",
        "video/mp4",
        "output.mp4",
    )


@app.post("/analysis/{analysis_id}/ask", response_model=AskResponse)
def ask(analysis_id: str, request: AskRequest):
    run_dir = RUNS_DIR / analysis_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="analysis_id not found")

    intent, answer, evidence, confidence = answer_question(str(run_dir), request.question)
    return AskResponse(
        analysis_id=analysis_id,
        question=request.question,
        answer=answer,
        intent=intent,
        evidence=evidence,
        confidence=confidence,
    )
