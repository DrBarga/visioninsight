from fastapi import FastAPI, UploadFile, File
import shutil
import uuid

from app.video.processor import VideoProcessor

app = FastAPI(title="VisionInsight API")

_processor: VideoProcessor | None = None


def get_processor() -> VideoProcessor:
    global _processor
    if _processor is None:
        _processor = VideoProcessor()  # YOLO init произойдёт тут, при первом запросе
    return _processor


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-video/")
async def analyze_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())

    input_path = f"input_{video_id}.mp4"
    output_path = f"output_{video_id}.mp4"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    processor = get_processor()
    events = processor.process(input_path, output_path)

    return {
        "video_id": video_id,
        "events_count": len(events),
        "events": events,
        "output_video": output_path
    }
