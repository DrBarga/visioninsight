# Contributing

Bug reports and focused product suggestions are welcome. Use the issue templates and include the version, environment, expected behavior, and observed result. Use synthetic media or describe the footage; do not attach personal or customer video without permission.

For a security issue, use the private reporting channel in [SECURITY.md](SECURITY.md).

The project is proprietary. Public repository access does not grant permission to modify or redistribute it. Discuss a proposed code contribution with the maintainer and obtain the required written permission before submitting changes.

## Reviewing an authorized change

- Keep changes focused on the reported behavior and preserve existing API contracts where possible.
- Use clear names and comments that explain decisions. Keep credentials, local databases, uploaded media, and generated analyses out of Git.
- Run the checks relevant to the change. Backend behavior changes should include a meaningful regression test. Browser changes should be checked through the affected user flow.
- Describe the trigger, resulting behavior, verification, and remaining limitations in the pull request. Do not claim production or payment verification from local tests.

The [README](README.md) contains setup and test commands. Deployment changes should also update [operations](docs/OPERATIONS.md) and the [release requirements](docs/RELEASE_GATE.md) when applicable.
