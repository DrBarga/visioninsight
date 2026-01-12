import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse

from app.video.processor import VideoProcessor

app = FastAPI(title="VisionInsight API", version="0.2")

processor = VideoProcessor(runs_dir="runs")


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2"}


@app.post("/analyze-video/")
async def analyze_video(file: UploadFile = File(...)):
    analysis_id = str(uuid.uuid4())
    run_dir = Path("runs") / analysis_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / "input.mp4"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = processor.process(str(input_path), analysis_id=analysis_id)
        # result already compact
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/{analysis_id}/summary")
def get_summary(analysis_id: str):
    summary_path = Path("runs") / analysis_id / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="analysis_id not found")
    return FileResponse(str(summary_path), media_type="application/json")

@app.get("/analysis/{analysis_id}/timeline", response_class=FileResponse)
def get_timeline(analysis_id: str):
    run_dir = Path("runs") / analysis_id
    path = run_dir / "timeline.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="timeline not found")

    return FileResponse(
        path=str(path),
        media_type="application/x-ndjson",
        filename="timeline.jsonl"
    )

@app.get("/analysis/{analysis_id}/timeline")
def get_timeline(analysis_id: str):
    timeline_path = Path("runs") / analysis_id / "timeline.jsonl"
    if not timeline_path.exists():
        raise HTTPException(status_code=404, detail="timeline not found")
    # JSONL is plain text lines; client can parse line-by-line
    return FileResponse(str(timeline_path), media_type="application/jsonl")


@app.get("/")
def root():
    return PlainTextResponse("VisionInsight API is running. Go to /docs")
