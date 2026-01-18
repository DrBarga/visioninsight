import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse

from app.video.processor import VideoProcessor
from app.query.schemas import AskRequest, AskResponse
from app.query.engine import answer_question

app = FastAPI(title="VisionInsight API", version="0.3.1")

RUNS_DIR = Path("runs")
processor = VideoProcessor(runs_dir=str(RUNS_DIR))


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.3.1"}


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
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/{analysis_id}/summary")
def get_summary(analysis_id: str):
    summary_path = RUNS_DIR / analysis_id / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="analysis_id not found")
    return FileResponse(str(summary_path), media_type="application/json", filename="summary.json")


@app.get("/analysis/{analysis_id}/timeline")
def get_timeline(analysis_id: str):
    timeline_path = RUNS_DIR / analysis_id / "timeline.jsonl"
    if not timeline_path.exists():
        raise HTTPException(status_code=404, detail="timeline not found")

    return FileResponse(
        str(timeline_path),
        media_type="application/x-ndjson",
        filename="timeline.jsonl"
    )


@app.get("/analysis/{analysis_id}/stats")
def get_stats(analysis_id: str):
    stats_path = RUNS_DIR / analysis_id / "stats.json"
    if not stats_path.exists():
        raise HTTPException(status_code=404, detail="stats not found")

    return FileResponse(str(stats_path), media_type="application/json", filename="stats.json")


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
        confidence=confidence
    )
