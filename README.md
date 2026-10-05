# VisionInsight

VisionInsight turns recorded MP4 footage into timestamped moments, crowd and object metrics, optional region measurements, evidence cards, and exportable reports. It is designed for reviewing events and public spaces. It does not identify people or infer age, gender, or emotion.

This repository is a v1.0 release candidate. Do not enable public paid access until the gates in [docs/RELEASE_GATE.md](docs/RELEASE_GATE.md) are complete. In particular, the current Ultralytics YOLOv8 dependency needs a commercial-license decision for a closed-source service.

The current implementation against the supplied product blueprint is tracked in [docs/BLUEPRINT_STATUS.md](docs/BLUEPRINT_STATUS.md).

## Local development

Use Python 3.12 and FFmpeg. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:PYTHONPATH = "backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Development mode uses SQLite, automatically verifies local test accounts, and runs one embedded video worker. It is not a production configuration.

Run the tests:

```powershell
$env:PYTHONPATH = "backend"
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```

## Container deployment

Copy `.env.example` to `.env`, replace the database password, and run `docker compose up --build`. The API binds to `127.0.0.1:8000`; put an HTTPS reverse proxy in front of it for public access. The API and worker share a persistent analysis volume. PostgreSQL uses a separate volume.

For production, set `VISIONINSIGHT_ENV=production`, an HTTPS `VISIONINSIGHT_PUBLIC_BASE_URL`, `VISIONINSIGHT_COOKIE_SECURE=true`, SMTP settings, legal URLs and business contact. Do not set `VISIONINSIGHT_MODEL_LICENSE_CLEARED=true` until the model licensing route is documented and approved. Paid checkout also requires an approved Paddle account, live API key, webhook secret, two recurring price IDs, and matching public price labels. Missing values keep checkout disabled.

See [operations](docs/OPERATIONS.md) for deployment, backup, restore, and incident procedures.

## Product flow

1. Register and verify an email address in production.
2. Upload an MP4 using Quick Scan or Event Pulse. Upload size, video duration, source-video storage, and monthly processing minutes depend on the plan.
3. Poll the analysis resource until it completes or fails. The worker writes versioned artifacts and source-linked evidence.
4. Review metrics and moments, ask supported questions, inspect the original video, and export JSON, an annotated video, or a short clip.
5. Delete an analysis manually or let retention remove it after the plan's retention period.

The built-in Ask flow answers known metric questions deterministically. For other questions it retrieves relevant moments; it does not generate unsupported factual claims. Semantic retrieval is optional and must be configured and evaluated separately. Tracking IDs are estimates, not identities. A track appearing or disappearing is not a physical entrance or exit; configure a line to measure crossings.

## API

The OpenAPI specification is at `/docs`. All video and artifact endpoints require a session or API key and check ownership. Browser sessions require `X-CSRF-Token` for writes. API keys are available only on paid plans and are shown once when created.

Main endpoints:

- `POST /v1/auth/register`, `POST /v1/auth/login`, `GET /v1/auth/me`
- `POST /v1/analyses`, `GET /v1/analyses/{id}`, `POST /v1/analyses/{id}/cancel`, `DELETE /v1/analyses/{id}`
- `GET /v1/analyses/{id}/overview`, `/moments`, `/metrics`, `/report`, `/artifacts`
- `POST /v1/analyses/{id}/query`
- `POST /v1/billing/checkout`, `/portal`, `/webhooks/paddle`
- `GET /health`, `GET /ready`

Upload with `POST /v1/analyses?recipe=quick_scan` and a raw MP4 request body (`Content-Type: video/mp4`). Event Pulse uses `recipe=event_pulse`. Optional `zones` and `lines` are JSON-encoded query parameters with normalized 0–1 coordinates. API details and examples are in [architecture](docs/ARCHITECTURE.md).

## Repository

- `backend/app/api`: authentication, analyses, billing
- `backend/app/jobs`: persistent job worker
- `backend/app/video`, `detection`, `tracking`, `analytics`, `moments`, `query`: analysis pipeline
- `backend/app/db`, `storage`, `core`: state, files, settings, security
- `frontend`: browser application
- `backend/tests`: unit and integration tests
- `scripts/evaluate_benchmark.py`: labeled-video evaluation

Test media, local run output, credentials, and local databases are ignored by Git. Previous commits may still contain historical media; removing it from Git history requires a separate coordinated history rewrite.
