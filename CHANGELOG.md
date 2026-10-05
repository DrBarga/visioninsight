# Changelog

## 1.0.0-rc.1 — 2026-10-05

This release candidate introduces the account-based web service and browser workflow. The Python version is `1.0.0rc1`; the Git tag is `v1.0.0-rc.1`.

### Added

- Browser registration, login, email verification, password recovery, and account-scoped analyses.
- Persistent asynchronous analysis jobs with progress, cancellation, usage reservations, and retention cleanup.
- Quick Scan and Event Pulse recipes for recorded MP4 files.
- Moment selection, evidence frames, deterministic metric answers, and source-video review.
- Optional browser drawing for occupancy zones and crossing lines.
- Structured JSON reports, annotated Event Pulse video, and bounded clip exports.
- Paid-plan API keys, Paddle checkout and customer portal integration, and signed billing callbacks. Checkout is disabled until configured.
- Docker and Compose deployment, CI checks, operations documentation, and a labeled-video benchmark evaluator.

### Changed

- Video uploads now use a bounded raw `video/mp4` body. Zones and lines are JSON-encoded query parameters.
- Quick Scan includes basic object detections while retaining its sampled processing mode.
- Tracking events distinguish track appearance and loss from measured physical crossings.
- API errors report useful status without exposing internal processing exceptions.

### Fixed

- Upload and callback parsing limits, video metadata inspection, worker processing budgets, and derived-output limits.
- Generic production registration responses for new and existing email addresses.
- Browser video-preview permissions, region submission, and missing favicon requests.
- Empty videos no longer produce a zero-person peak-crowd highlight.

### Publication status

The automated suite passes 32 tests. A local browser check completed upload, region submission, processing, results, and a metric question. The container was built and its health, readiness, and plan endpoints were checked.

Paid public deployment, provider-backed purchases, representative accuracy evaluation, production load testing, and the remaining [release requirements](docs/RELEASE_GATE.md) are outstanding. This tag is a prerelease for evaluation.
