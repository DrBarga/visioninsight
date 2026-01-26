# VisionInsight

VisionInsight is a computer vision system for **crowd and event video analytics**.  
It transforms raw videos into structured data, analytics, highlights, and natural-language answers.

The system is designed for videos with many people (festivals, public events, concerts, streets) and focuses on **explainable intelligence**, not black-box predictions.

---

## Core Capabilities

- Person detection (YOLOv8)
- Multi-object tracking (IOU-based)
- Crowd statistics and dynamics
- High-density crowd window detection
- Automatic video highlights extraction
- Natural-language Q&A over video analytics (no LLMs yet)
- Fully traceable results (JSON / JSONL artifacts)

---

## Project Structure

backend/
├── app/
│ ├── video/ # video processing & tracking
│ ├── analytics/ # stats, highlights, crowd dynamics
│ ├── query/ # intents, Q&A engine
│ ├── detection/ # YOLO detector
│ └── tracking/ # IOU tracker
├── runs/ # analysis outputs (auto-generated)
└── main.py # FastAPI entry point

---

## Each video analysis creates a folder:

runs/<analysis_id>/
├── input.mp4
├── output.mp4
├── summary.json
├── stats.json
├── highlights.json
├── timeline.jsonl
├── events.jsonl
└── people.jsonl

---

## Requirements

- Python 3.10+
- CPU is sufficient (GPU optional)
- OS: Windows / Linux / macOS

Install dependencies:
```bash```
pip install -r requirements.txt

---

How to Run
From the backend directory:
python -m uvicorn app.main:app

Server will start at:
http://127.0.0.1:8000

Interactive API documentation:
http://127.0.0.1:8000/docs

---

## API Usage

1. Analyze a Video

POST /analyze-video/

Upload a video file.
Response contains analysis_id and summary info

2. Ask Questions About the Video

POST /analysis/{analysis_id}/ask

POST /analysis/{analysis_id}/ask

Example body:
{
  "question": "When was it crowded?"
}

## Supported questions include:
How many people?
When was it crowded?
When did the crowd start growing?
What was the most dynamic moment?
Give me highlights
Give me a summary

3. Get Generated Artifacts
GET /analysis/{analysis_id}/summary
GET /analysis/{analysis_id}/stats
GET /analysis/{analysis_id}/highlights
GET /analysis/{analysis_id}/timeline

---

# Design Principles

Deterministic and explainable logic
No hidden decisions
Every answer backed by data
Built for extension (LLMs, databases, dashboards)

LLMs will be added after structured understanding is complete
