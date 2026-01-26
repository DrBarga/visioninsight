import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse

from app.video.processor import VideoProcessor
from app.query.schemas import AskRequest, AskResponse
from app.query.engine import answer_question

app = FastAPI(title="VisionInsight API", version="0.3.8")

RUNS_DIR = Path("runs")
processor = VideoProcessor(runs_dir=str(RUNS_DIR))


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.3.8"}


@app.get("/")
def root():
    return PlainTextResponse("VisionInsight API is running. Go to /docs")


@app.post("/analyze-video/")
async def analyze_video(file: UploadFile = File(...)):
    analysis_id = str(uuid.uuid4())
    run_dir = RUNS_DIR / analysis_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / "input.mp4"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = processor.process(str(input_path), analysis_id=analysis_id)
        return JSONResponse(content=result)  # result is dict → safe
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _file_or_404(path: Path, msg: str):
    if not path.exists():
        raise HTTPException(status_code=404, detail=msg)
    return path


@app.get("/analysis/{analysis_id}/summary")
def get_summary(analysis_id: str):
    path = _file_or_404(RUNS_DIR / analysis_id / "summary.json", "summary not found")
    return FileResponse(str(path), media_type="application/json", filename="summary.json")


@app.get("/analysis/{analysis_id}/timeline")
def get_timeline(analysis_id: str):
    path = _file_or_404(RUNS_DIR / analysis_id / "timeline.jsonl", "timeline not found")
    return FileResponse(str(path), media_type="application/x-ndjson", filename="timeline.jsonl")


@app.get("/analysis/{analysis_id}/events")
def get_events(analysis_id: str):
    path = _file_or_404(RUNS_DIR / analysis_id / "events.jsonl", "events not found")
    return FileResponse(str(path), media_type="application/x-ndjson", filename="events.jsonl")


@app.get("/analysis/{analysis_id}/people")
def get_people(analysis_id: str):
    path = _file_or_404(RUNS_DIR / analysis_id / "people.jsonl", "people not found")
    return FileResponse(str(path), media_type="application/x-ndjson", filename="people.jsonl")


@app.get("/analysis/{analysis_id}/stats")
def get_stats(analysis_id: str):
    path = _file_or_404(RUNS_DIR / analysis_id / "stats.json", "stats not found")
    return FileResponse(str(path), media_type="application/json", filename="stats.json")


@app.get("/analysis/{analysis_id}/highlights")
def get_highlights(analysis_id: str):
    path = _file_or_404(RUNS_DIR / analysis_id / "highlights.json", "highlights not found")
    return FileResponse(str(path), media_type="application/json", filename="highlights.json")


@app.get("/analysis/{analysis_id}/quality")
def get_quality(analysis_id: str):
    path = _file_or_404(RUNS_DIR / analysis_id / "quality.json", "quality not found")
    return FileResponse(str(path), media_type="application/json", filename="quality.json")


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
