VisionInsight — Video Analytics Platform (v0.1)

VisionInsight is an end-to-end AI-powered video analytics backend that performs object detection on uploaded videos and returns structured detection data along with an annotated output video.
This project is designed as a production-oriented MVP, demonstrating how to build a complete Computer Vision pipeline using modern ML tooling and a REST API.

----------------------------------------------------------------

1) Features (v0.1)

FastAPI-based backend with automatic API documentation
Video upload and processing via REST API
Frame-by-frame object detection using YOLOv8 (CPU)
Bounding box visualization on output video
Structured JSON output with detection metadata
Health-check endpoint for service monitoring

2) Tech Stack

Backend: FastAPI, Uvicorn
Computer Vision: OpenCV
Object Detection: Ultralytics YOLOv8 (yolov8n, CPU-friendly)
Language: Python 3.10+
Deployment (local): Uvicorn ASGI server

3)Project Structure

visioninsight/
  backend/
     app/
       main.py              # FastAPI entry point
        detection/
          yolo.py           # YOLOv8 detector
        video/
          processor.py      # Video processing pipeline
        tracking/             # (reserved for v0.2)
        requirements.txt

4) API Endpoints

1.Health Check
GET /health

2.Response:
{
  "status": "ok"
}

3.Analyze Video
POST /analyze-video/

4.Request
Content-Type: multipart/form-data
Field: file (MP4 video)

5. Response (200 OK)
{
  "video_id": "uuid",
  "events_count": 128,
  "events": [
    {
      "frame": 12,
      "objects": [
        {
          "class_id": 0,
          "confidence": 0.84,
          "bbox": [119, 360, 714, 1280]
        }
      ]
    }
  ],
  "output_video": "output_<uuid>.mp4"
}

5) How to Run Locally

1. Create virtual environment

python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

2. Install dependencies

pip install -r requirements.txt

3. Start server

cd backend
python -m uvicorn app.main:app

Server will be available at:
http://127.0.0.1:8000

Swagger UI:
http://127.0.0.1:8000/docs

----------------------------------------------------------------

Notes & Limitations (v0.1)

Detection is performed per frame (no object tracking yet)
Output JSON may be large for long videos
Results are stored locally (no database in v0.1)
Optimized for CPU inference, not GPU

----------------------------------------------------------------

Roadmap
v0.2 (next)
Object tracking (track IDs across frames)
High-level events (person entered / exited)
Timeline-based output instead of raw frame data

v0.3
LLM-based video summarization
Natural language Q&A over detected events

v0.4+
Web frontend (Next.js)
Result storage and retrieval
API authentication & rate limiting

----------------------------------------------------------------

focus on extensibility and scalability
