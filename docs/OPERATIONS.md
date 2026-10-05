# Operations

## First deployment

1. Use a host with enough CPU, RAM, and disk for the expected video volume. Test throughput on that host before pricing plans.
2. Copy `.env.example` to `.env` and set a strong URL-safe PostgreSQL password. Keep `.env` outside backups that are shared with third parties.
3. Configure HTTPS in a reverse proxy with an upload body limit matching the largest enabled plan, a request timeout appropriate for streamed uploads, and trusted client-IP forwarding for rate limits. Forward only the API port from localhost. Set `VISIONINSIGHT_PUBLIC_BASE_URL`, secure cookies, and production mode.
4. Configure SMTP, verified sender domain, and the real Terms, Privacy, and business contact information. Test email verification and password recovery with a real mailbox.
5. Resolve the detector and model license before setting `VISIONINSIGHT_MODEL_LICENSE_CLEARED=true`. The current Ultralytics dependency is not covered by this repository's proprietary license.
6. Create and verify a Paddle account, approved checkout domain and default payment link, recurring prices, and webhook destination `/v1/billing/webhooks/paddle`. Use sandbox first, then live credentials. Do not grant paid access from the browser redirect; the signed webhook is authoritative.
7. Start with `docker compose up --build -d`. Check `docker compose ps`, `GET /ready`, worker logs, and a complete upload-to-export flow.

Do not expose the development configuration to the public internet. The Docker Compose file binds port 8000 to localhost; an external reverse proxy and TLS configuration are deliberately deployment-specific.

## Backups and restore

Back up both persistent volumes: PostgreSQL and analysis data. A database-only snapshot loses videos and evidence; a files-only snapshot loses ownership and billing state. Take a coordinated backup when writes are quiet, encrypt it, store it off-host, and test a restore on a separate host. Keep at least daily backups and a documented retention policy appropriate for the service.

Example database export from the Compose directory:

```powershell
docker compose exec -T db pg_dump -U visioninsight -Fc visioninsight > visioninsight-db.dump
```

Also snapshot the `analysis_data` volume through the host's backup system. Do not publish either artifact. Restore to a fresh environment before pointing public traffic at it, run `/ready`, compare sample analysis ownership and artifact access, and verify a new job completes.

## Monitoring

Monitor API readiness, worker restarts, queue age, analysis failures, disk usage, database volume, SMTP failures, and webhook delivery failures. Logs include request IDs and processing failures without printing credentials or uploaded video. Alert if jobs remain `queued` beyond normal processing time or if Paddle deliveries fail. On worker restart, in-progress jobs are marked failed and their minute reservations are released; users can retry from their source file.

The cleanup loop removes terminal analyses after the plan retention period. It deletes files before the database record, so a filesystem error leaves the record available for a retry. Monitor cleanup errors and audit both the database and storage before declaring a deletion incident closed. Keep one worker in this single-host release candidate.

## Release and rollback

Pin the image tag and record the database backup and code revision before a deployment. Deploy the API and worker from the same image version. Run the release gate and a real customer flow in sandbox after deployment. For rollback, restore both application image and any incompatible database state from the paired backup; `create_all` only creates missing tables and is not a full migration system. Do not rely on an untested rollback.
