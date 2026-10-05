# Third-party dependencies and model licensing

VisionInsight's original code is covered by [LICENSE](LICENSE). That notice does not replace the licenses of dependencies, pretrained models, or their bundled components.

The table below records license information declared by the direct dependency versions used for this release candidate. It is an inventory, not a complete license bundle for redistributing a Python environment or container. Such a distribution must retain the applicable upstream notices for direct and transitive packages, native libraries, and model files.

| Dependency | Version | Declared license |
| --- | --- | --- |
| FastAPI | 0.129.0 | MIT |
| opencv-python | 4.13.0.92 | Apache 2.0; wheel components have additional notices |
| Ultralytics | 8.4.14 | AGPL-3.0 |
| PyTorch | 2.10.0 | BSD-3-Clause |
| torchvision | 0.25.0 | BSD |
| NumPy | 2.4.2 | BSD-3-Clause, 0BSD, MIT, Zlib, and CC0-1.0 |
| Pydantic | 2.12.5 | MIT |
| Uvicorn | 0.41.0 | BSD-3-Clause |
| faster-whisper | 1.2.1 | MIT |
| open-clip-torch | 3.2.0 | MIT |
| Pillow | 12.1.1 | MIT-CMU |
| SQLAlchemy | 2.0.54 | MIT |
| psycopg | 3.3.6 | LGPL-3.0-only; binary packages have separate bundled-library notices |
| HTTPX | 0.28.1 | BSD-3-Clause |

FFmpeg and PostgreSQL are installed separately through the host or container distribution. Their licenses and enabled components must also be considered when redistributing an image.

## Detector and model files

The current detector uses Ultralytics YOLOv8 and the `yolov8n.pt` model. These third-party works are not licensed under VisionInsight's proprietary notice. Ultralytics describes its licensing options on the [official license page](https://www.ultralytics.com/license) and publishes the [AGPL-3.0 terms](https://www.ultralytics.com/legal/agpl-3-0-software-license).

The commercial licensing route for a proprietary VisionInsight service is unresolved. The production processing flag must remain disabled until the applicable rights and obligations are documented, or a compatible detector and model are substituted and validated. Publishing a source prerelease does not establish commercial model clearance.

Optional transcription, refinement, and embedding models may have licenses different from the packages that load them. Review the exact model and revision before enabling or distributing it.
