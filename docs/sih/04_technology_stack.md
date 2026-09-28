# 04 - Technology stack (SIH26078)

## Backend

| Layer | Choice | Version | Why |
| --- | --- | --- | --- |
| Language | Python | 3.13 (≥3.11 supported) | The scientific stack's native language |
| Web framework | FastAPI | current | Typed, OpenAPI-generated, fast |
| ASGI server | uvicorn | with `[standard]` | The reference server; container-friendly |
| Settings | pydantic-settings | — | Every setting is an environment variable, validated |
| Array processing | NumPy, Xarray, Dask | current | Labelled n-dimensional data is the right abstraction |
| I/O | NetCDF4, pandas | current | The interchange format for gridded meteorology |
| Statistics | SciPy, MetPy | current | Unit-aware meteorological quantities |
| Deep learning | PyTorch | current | Already a core dependency; the prototypes use it |

**Dependencies were not added in Phase 11.** The scientific and web stack was
fixed in Phase 0 and reused throughout.

## Frontend

| Choice | Version | Why |
| --- | --- | --- |
| React | 19.1 | Component model; mature ecosystem |
| TypeScript | 5.9 | The API contract is typed end to end |
| Vite | 7.1 | Fast dev server, straightforward production build |
| Leaflet + react-leaflet | 1.9 / 5.0 | Geographic display without a paid tile provider |
| axios | 1.7 | One HTTP client, one error-normalisation point |

No UI framework, no state-management library, no charting package. The
dashboard is plain components against typed API responses, which keeps the
bundle small and the contract visible.

## Infrastructure

| Choice | Why |
| --- | --- |
| Docker Compose | Two services, explicit volumes, no orchestrator needed |
| `python:3.13-slim` | Small, matches the validated interpreter |
| `node:22-alpine` | Small, matches the frontend toolchain |
| Non-root container user (uid 1001) | A container that runs as root is a container that is one misconfiguration away from compromise |
| Healthcheck on `/health` | The frontend waits for `service_healthy`, not merely for start-up |

## Testing

| Tool | Use |
| --- | --- |
| pytest | The suite: 1816 tests, 0 skipped, 0 xfailed |
| `fastapi.testclient` (httpx) | In-process API tests, no port bound |
| `TestClient` with `raise_server_exceptions=False` | Assert the error envelope, not just the happy path |

There is no mocking framework in the suite. Tests run the real services against
a temporary `DATA_DIR`, because a mocked pipeline cannot tell you that the
orchestrator writes to the wrong directory — which is exactly the defect Phase
11 found.

## Deliberate non-choices

| Not used | Why |
| --- | --- |
| A database | Every artifact is a file with a checksum. A database would add a failure mode without adding a capability. |
| A message queue | The pipeline is synchronous and second-scale for the data sizes involved. |
| Kubernetes manifests | Premature. Compose expresses the deployment completely. |
| A frontend charting library | The charts needed are simple; a dependency would add more than it saves. |
| An authentication library | The API is unauthenticated and says so. A half-built auth is worse than a documented boundary. |
