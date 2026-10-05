# Security policy

## Reporting a vulnerability

Report a suspected vulnerability through [GitHub private vulnerability reporting](https://github.com/DrBarga/visioninsight/security/advisories/new). Do not include credentials, personal video, customer data, or exploit details in a public issue.

Include the affected version or commit, the component, the conditions needed to reproduce the problem, and the observed impact. Use synthetic data where possible. The maintainer will review the report and coordinate any fix and disclosure with the reporter. This project does not currently offer a response-time guarantee or a bug bounty.

## Supported versions

The latest 1.0 release candidate is the maintained development version. Earlier 0.x versions are historical and do not receive regular security fixes. A release-candidate tag is not a production security certification.

## System and scope

VisionInsight is a service for recorded MP4 analysis. The security boundary includes the FastAPI API, browser application, authentication, PostgreSQL or local development SQLite, job worker, private artifact storage, video processing, exports, email flows, and Paddle billing integration.

Production is intended to run behind an HTTPS reverse proxy on one host with a shared analysis volume and one worker. The operator controls the host, database, proxy, SMTP service, model files, and provider credentials. Development settings are intended for local evaluation.

## Trust boundaries

Treat uploaded media and its metadata, request bodies, query parameters, region coordinates, account credentials, and billing callback payloads as untrusted. An authenticated account can still submit hostile input. A valid media extension or container does not make its decoder input safe.

Source videos, evidence, reports, account records, session tokens, API keys, email tokens, billing state, and quota reservations are protected assets. Worker code and model files are trusted deployment inputs; users must not be able to select arbitrary model or filesystem paths.

## Required security properties

- Every analysis, video, evidence image, report, and export must be accessible only to its owning account. An identifier alone must never grant access.
- Authorization must happen before protected reads and writes. Browser-session mutations must validate a CSRF token. API keys must be checked server-side and restricted to eligible accounts.
- Passwords must be salted and hashed. Session tokens, API keys, and email tokens must not be stored or logged in plaintext. Account recovery must revoke existing sessions and API keys.
- Uploads and callback bodies must have enforced byte limits. Video processing must have geometry, frame, time, output-size, and free-disk budgets. Repeated export requests must not create unbounded work or output.
- Filenames and artifact requests must not escape the analysis directory. Only approved artifact paths may be returned by the API.
- Paid access must depend on a verified, correlated server-side billing event. A browser redirect, claimed price, or unsigned callback must not grant access. Duplicate and stale events must not overwrite newer billing state.
- Quota reservations must be transactional and released after failed or cancelled processing. Deletion and retention must remove the analysis files and account-scoped record consistently.
- Client-facing errors must not expose credentials, stack traces, or server paths. Production must use HTTPS and secure cookies.

## Findings and severity

Account isolation failures, authentication or recovery bypasses, forged billing entitlements, secret disclosure, path traversal, unsafe parser execution, and realistically reachable resource exhaustion are reportable. Include the prerequisites and practical impact so the maintainer can assess severity. A failure that requires authentication can still be serious.

There are no blanket exclusions or accepted security risks that suppress these findings. Missing product features and inaccurate detections are product issues unless they also break a security or privacy boundary.

## Current limitations

The release candidate has automated control tests and local processing checks. These do not establish security on a deployed host. The release requirements include dependency and parser review, deployment validation, load testing, and backup restoration.

Media processing runs inside the worker container; application budgets and container resource limits are not a dedicated per-upload sandbox. The shared local volume relies on host permissions and infrastructure protection. Application-level storage encryption, multi-host processing, and a versioned database migration system are not implemented.

Development mode automatically verifies accounts and can use insecure cookies and SQLite. Production configuration validates HTTPS, secure cookies, legal contact settings, and PostgreSQL. Model processing and paid checkout remain gated while licensing or merchant configuration is incomplete.

Historical Git commits contain older test media. Current `.gitignore` rules and removal from the current tree do not remove those historical objects. See [release requirements](docs/RELEASE_GATE.md) and [operations](docs/OPERATIONS.md) for the remaining deployment work.
