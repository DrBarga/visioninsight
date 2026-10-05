# Architecture

## Request and processing path

The FastAPI process handles registration, sessions, API keys, uploads, account-scoped reads, and Paddle webhooks. Uploads are validated, probed with OpenCV, copied to private storage, and inserted into the persistent database queue. The worker claims one queued analysis transactionally and runs detection, tracking, derived metrics, moment selection, keyframe extraction, and optional exports. Clients poll the job status while it runs.

Production uses PostgreSQL for users, billing state, quotas, and the job queue. The initial deployment uses a shared local volume for videos and artifacts and one worker. This is a single-host design; multi-host scaling needs object storage and distributed job ownership/leases before adding workers. Development defaults to SQLite and an embedded worker.

Each analysis directory is named by a UUID and stores the source video and versioned JSON/JSONL artifacts. The API uses an explicit artifact allowlist and owner checks. Returned artifact names are relative, not server filesystem paths. No uploaded filename becomes a path component. A finished analysis can produce:

`summary.json`, `meta.json`, `stats.json`, `zones.json`, `highlights.json`, `quality.json`, `moments.json`, `evidence.json`, `timeline.jsonl`, `people.jsonl`, `objects.jsonl`, `events.jsonl`, `objects_stats.json`, and optionally `output.mp4`.

## Measurement semantics

- A person track is a sequence of detections. Re-identification is not performed. Occlusion and missed frames may split one person into multiple tracks.
- `track_started` and `track_lost` describe tracker state, not physical entrances and exits.
- A configured line crossing is emitted only when consecutive observed track positions cross the finite segment. Direction depends on the line's start/end orientation.
- Zone occupancy is the number of visible confirmed tracks whose foot points lie inside a normalized polygon. Dwell is observed sampled time, not guaranteed real dwell.
- Moment quality is a track-continuity heuristic, not a calibrated probability.
- CLIP refinement is optional; the default product recipes use the detector label and do not silently replace it with a low-margin text match.

## Regions example

Send these as two JSON-encoded query parameters in `POST /v1/analyses`. The request body is the raw MP4 file with `Content-Type: video/mp4`:

```json
{"zones":[{"id":"entrance","points":[[0.1,0.1],[0.45,0.1],[0.45,0.8],[0.1,0.8]]}],"lines":[{"id":"gate","start":[0.5,0.2],"end":[0.5,0.85]}]}
```

The `zones` field receives only the `zones` array and the `lines` field only the `lines` array. Coordinates are relative to the image; `[0,0]` is top-left and `[1,1]` is bottom-right. Validate placement on a representative frame before relying on counts.

## Access and quota controls

Session and API-key tokens are stored as hashes. Browser mutations require a CSRF token. Production registration needs email verification; password recovery revokes existing sessions and API keys. API keys are restricted to paid plans. Analyses, videos, evidence images, reports, and exports check the owning user.

Source-video bytes are reserved against the plan at upload, and processing minutes are reserved transactionally per UTC month. Successful analyses consume reserved minutes; failed/cancelled analyses release them. Source-video storage is released on deletion or retention expiry. Derived output bytes are not counted against the displayed source-video quota. Each analysis has a separate generated-output cap and the worker stops when the configured free-disk reserve is reached; production still needs host-volume monitoring and capacity alerts.

Paddle checkout records the server-created transaction ID. Subscription webhooks must have a valid signature, recent timestamp, matching transaction or existing subscription, known price ID, and a newer event timestamp. Only an active subscription grants paid access. Checkout is disabled until provider, legal, and license settings are complete.

## Known boundaries

The service is for recorded MP4 files, not live streams. It does not support facial recognition or sensitive trait inference. It is not designed for safety-critical decisions without human review. Scene cuts, camera movement, dense occlusion, low light, and distant people can reduce quality. A benchmark with labeled representative footage is required before publishing accuracy claims.
