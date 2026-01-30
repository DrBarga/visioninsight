import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse

from app.video.processor import VideoProcessor
from app.query.schemas import AskRequest, AskResponse
from app.query.engine import answer_question

app = FastAPI(title="VisionInsight API", version="0.3.9+profiles")

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
    profile: str = Form("balanced"),  # NEW: balanced | crowd_people | people_strict | vehicles
):
    analysis_id = str(uuid.uuid4())
    run_dir = RUNS_DIR / analysis_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / "input.mp4"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = processor.process(str(input_path), analysis_id=analysis_id, detection_profile=profile)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _file_or_404(path: Path, detail: str, media_type: str, filename: str):
    if not path.exists():
        raise HTTPException(status_code=404, detail=detail)
    return FileResponse(str(path), media_type=media_type, filename=filename)


@app.get("/analysis/{analysis_id}/summary")
def get_summary(analysis_id: str):
    path = RUNS_DIR / analysis_id / "summary.json"
    return _file_or_404(path, "summary not found", "application/json", "summary.json")


@app.get("/analysis/{analysis_id}/timeline")
def get_timeline(analysis_id: str):
    path = RUNS_DIR / analysis_id / "timeline.jsonl"
    return _file_or_404(path, "timeline not found", "application/x-ndjson", "timeline.jsonl")


@app.get("/analysis/{analysis_id}/people")
def get_people(analysis_id: str):
    path = RUNS_DIR / analysis_id / "people.jsonl"
    return _file_or_404(path, "people not found", "application/x-ndjson", "people.jsonl")


@app.get("/analysis/{analysis_id}/events")
def get_events(analysis_id: str):
    path = RUNS_DIR / analysis_id / "events.jsonl"
    return _file_or_404(path, "events not found", "application/x-ndjson", "events.jsonl")


@app.get("/analysis/{analysis_id}/stats")
def get_stats(analysis_id: str):
    path = RUNS_DIR / analysis_id / "stats.json"
    return _file_or_404(path, "stats not found", "application/json", "stats.json")


@app.get("/analysis/{analysis_id}/highlights")
def get_highlights(analysis_id: str):
    path = RUNS_DIR / analysis_id / "highlights.json"
    return _file_or_404(path, "highlights not found", "application/json", "highlights.json")


@app.get("/analysis/{analysis_id}/quality")
def get_quality(analysis_id: str):
    path = RUNS_DIR / analysis_id / "quality.json"
    return _file_or_404(path, "quality not found", "application/json", "quality.json")


@app.get("/analysis/{analysis_id}/objects")
def get_objects(analysis_id: str):
    path = RUNS_DIR / analysis_id / "objects.jsonl"
    return _file_or_404(path, "objects not found", "application/x-ndjson", "objects.jsonl")


@app.get("/analysis/{analysis_id}/objects-stats")
def get_objects_stats(analysis_id: str):
    path = RUNS_DIR / analysis_id / "objects_stats.json"
    return _file_or_404(path, "objects_stats not found", "application/json", "objects_stats.json")


@app.post("/analysis/{analysis_id}/ask", response_model=AskResponse)
def ask(analysis_id: str, req: AskRequest):
    run_dir = RUNS_DIR / analysis_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="analysis_id not found")

    intent, answer, evidence, confidence = answer_question(str(run_dir), req.question)
    return AskResponse(
        analysis_id=analysis_id,
        question=req.question,
        answer=answer,
        intent=intent,
        evidence=evidence,
        confidence=confidence,
    )