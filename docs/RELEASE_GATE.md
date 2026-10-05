# v1.0 release gate

This file separates implementation from verified production readiness. A green test suite is necessary but is not proof that customers can buy and use the service.

## Implemented and checked locally

- Account-scoped API, CSRF-protected browser sessions, hashed API keys, email verification and password recovery paths.
- Asynchronous persistent jobs, quota reservations, cancellation, private artifact access, retention cleanup.
- Quick Scan and Event Pulse, moments, source-linked evidence, region and line analytics, question answering, JSON and clip export.
- Responsive browser application and disabled-by-default paid checkout.
- Signed Paddle webhook validation, transaction correlation, duplicate and out-of-order event handling.
- Automated tests and a direct smoke run using the actual YOLOv8 model.
- Uploads and webhooks use bounded streaming; video geometry, frame count, processing time, generated output, and minimum free disk have application limits. The Compose worker also has CPU, memory, and process limits. The offline security audit was a pre-fix snapshot; these changes need renewed review and adversarial validation on the deployed stack.

## Required before a paid public v1.0

- [ ] Resolve Ultralytics code and YOLOv8 model licensing for a closed-source paid service, or replace the detector with a verified compatible alternative. Do not set the license clearance flag based only on technical tests.
- [ ] Obtain the seller identity, business contact, domain, reviewed Terms/Privacy/Imprint as applicable, and confirm applicable German tax/business obligations with a qualified adviser.
- [ ] Obtain Paddle account approval for this product and the seller, live keys, recurring prices, checkout domain/default payment link, and webhook destination. Verify checkout, renewal, failed payment, cancellation, refund, and customer portal in sandbox and live test purchases.
- [ ] Run a labeled evaluation on representative consented videos using `scripts/evaluate_benchmark.py`; agree release thresholds for visible-person MAE, line-crossing F1, moment recall, and throughput before publishing performance claims.
- [ ] Decide and validate the remaining product scope in [BLUEPRINT_STATUS.md](BLUEPRINT_STATUS.md): default semantic retrieval and narrator, outbound customer webhooks, optional transcript/refinement, and storage portability. Do not advertise unfinished features.
- [ ] Load-test uploads, queue pressure, clip export, disk budget, failure recovery, and retention on the actual production host.
- [ ] Configure and test SMTP, HTTPS, reverse-proxy body/time limits, backups, restore, monitoring, alerts, and incident contact.
- [ ] Add a versioned database migration process before evolving an existing production schema. The current initial schema is suitable for a fresh deployment only.
- [ ] Complete a security review of all dependencies and video parsers and verify no historical customer media remains reachable in the repository history if the repository will be shared.
- [ ] Validate the full customer journey on the public domain from signup through payment, processing, evidence review, export, portal, and deletion.

The commercial launch remains closed until every required item is checked with evidence. The current application version is `1.0.0rc1`, not a public 1.0.0 release.
