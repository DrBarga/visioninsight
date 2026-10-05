# Blueprint implementation status

This status compares the current `1.0.0rc1` code with the Master Blueprint, Powerful Release Blueprint, and release checklist dated 29 September 2026. It records product behavior, not a public-launch claim.

| Product requirement | Current state | Release evidence or remaining work |
| --- | --- | --- |
| Upload, progress, overview, moments, evidence, export | Implemented | API and browser flows exist; automated API and processor tests pass. Public-domain usability remains untested. |
| Quick Scan with people and basic objects | Implemented | The recipe samples frames and includes object detections without transcript, CLIP refinement, or annotated output. |
| Event Pulse with zones, lines, occupancy, and crossings | Implemented | API calculations and optional browser drawing are present. Direction depends on the order of line endpoints; validate against representative footage. |
| Account ownership, API keys, quotas, retention, deletion | Implemented | Automated tests cover access and reservations. Production backup and deletion operations still need host validation. |
| Paid Free, Developer, and Team plans | Integrated, disabled | Paddle checkout, portal, and signed entitlement webhooks are coded. Seller approval, live prices, legal pages, and real customer-flow tests are missing. Team is a higher-limit plan; shared workspaces are not implemented and are not advertised. |
| Semantic Ask over moments | Partial | Deterministic factual answers cite structured evidence. Moment search falls back to lexical matching unless a separately installed and configured embedding model is available. No default licensed embedding model is bundled. |
| LLM narrator with evidence contract | Not implemented | There is no production-safe narrator or evaluation suite. The current Ask flow does not claim to generate open-ended factual answers. |
| Optional transcript and CLIP object refinement in the product | Partial | Pipeline stages exist but are not exposed by the two public recipes. They need measured quality, resource budgets, and model licensing before customer use. |
| Outbound completion/failure webhooks | Not implemented | Only inbound Paddle billing webhooks exist. Do not promise developer callbacks. |
| Storage abstraction | Partial | A local adapter and persistent volume are in use. S3-compatible storage and a tested migration path are not available. |
| Labeled benchmark and release quality thresholds | Not complete | An evaluator exists, but no representative consented 20–30-video dataset, baseline report, or accepted thresholds are available. |
| Production operations | Not complete | Container, CI, health checks, limits, and operational instructions exist. Actual HTTPS host, SMTP, backups, alerts, load testing, migrations, and incident procedures must be validated before launch. |

The commercial release gate in [RELEASE_GATE.md](RELEASE_GATE.md) remains authoritative. Do not change the version to `1.0.0` or enable payments while these launch requirements are open.
