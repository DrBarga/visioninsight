# VisionInsight

VisionInsight helps you review recorded video through crowd and object metrics, timestamped moments, and answers linked to the source footage. Upload an MP4, define an area or crossing line if needed, and inspect the observations alongside the original video.

The current version is **1.0.0-rc.1** (`1.0.0rc1` in Python). It is available for development and evaluation. The paid public launch remains pending the [release requirements](docs/RELEASE_GATE.md), including model licensing, production validation, and merchant setup. See the [release notes](docs/releases/1.0.0-rc.1.md) and [changelog](CHANGELOG.md).

## What you can do

- Analyze recorded MP4 files in the browser or through the API.
- Review visible-person counts, object detections, tracking activity, and selected moments.
- Draw zones and crossing lines on a video frame before uploading.
- Open the source video and evidence behind a result, and ask supported questions about the analysis.
- Export structured JSON, annotated Event Pulse footage, and short clips.
- Keep analyses tied to an account, with upload limits, monthly usage, deletion, and retention.

| Recipe | Processing | Output |
| --- | --- | --- |
| Quick Scan | Samples frames for a faster overview; includes people and basic objects | Metrics, moments, source video, and structured exports |
| Event Pulse | More detailed analysis with optional zones and lines | Quick Scan outputs plus annotated video and region measurements |

VisionInsight does not identify people or infer age, gender, or emotion. Counts are estimates from tracked detections. Camera movement, occlusion, scene changes, and poor visibility can affect them.

## Local development

Use Python 3.12, FFmpeg, and FFprobe. Make sure both video tools are on `PATH`. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0+cpu torchvision==0.25.0+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:PYTHONPATH = "backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Development mode uses SQLite, automatically verifies local test accounts, and runs one embedded video worker. Use it on a local machine. Linux and macOS use the same commands with `python3.12` and `.venv/bin/python` in place of the Windows paths.

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

Test media, local run output, credentials, and local databases are ignored by Git. Historical commits still contain older test media; their removal from the current tree does not erase the history.

## Project documentation

- [Architecture and API contracts](docs/ARCHITECTURE.md)
- [Deployment, backup, and recovery](docs/OPERATIONS.md)
- [Security policy and private vulnerability reporting](SECURITY.md)
- [Release requirements](docs/RELEASE_GATE.md)
- [Product implementation status](docs/BLUEPRINT_STATUS.md)
- [Third-party dependencies and model licensing](THIRD_PARTY_NOTICES.md)
- [Contribution guidelines](CONTRIBUTING.md)

The original project code is proprietary. Public access to this repository does not grant a license to reuse it. Third-party packages and models retain their own licenses; see [LICENSE](LICENSE).
