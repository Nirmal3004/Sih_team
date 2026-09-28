# Security (SIH26078, Phase 10 - IMPLEMENTED)

> **Status: IMPLEMENTED and AUDITED.**
> `src/extreme_weather/security.py`, `src/extreme_weather/integrity.py`,
> `src/extreme_weather/api/middleware.py`, `Dockerfile`, `docker-compose.yml`.
>
> **This system is a research prototype and is not an operational
> disaster-warning system.** It has **no authentication**. That is stated in
> plain terms below rather than glossed over, because a security document that
> implies otherwise is worse than none.

## The honest summary

| Control | State |
| --- | --- |
| Filesystem boundary enforcement | **Implemented** |
| Path traversal refusal | **Implemented** |
| Extension allow-listing | **Implemented** |
| Subprocess hardening | **Implemented** |
| Input validation | **Implemented** |
| Credential redaction in logs and responses | **Implemented** |
| CORS configuration | **Implemented**, wildcard off by default |
| Non-root container | **Implemented** |
| Artifact integrity (SHA-256) | **Implemented** |
| Sanitised API errors | **Implemented** |
| **Authentication / authorisation** | **NOT IMPLEMENTED** |
| **TLS termination** | **NOT IMPLEMENTED** (expected of a reverse proxy) |
| **Rate limiting** | **NOT IMPLEMENTED** |
| **Audit log with actor identity** | **NOT IMPLEMENTED** (there are no actors) |
| **Secrets manager** | **NOT IMPLEMENTED** (environment variables only) |

**No fake authentication was added.** A token check that trusts a client-sent
header, or a "login" that sets a boolean, is theatre: it looks like a control
and stops nobody. The gap is documented instead.

### What the missing controls mean in practice

This stack is suitable for a **trusted internal network, a research group, or a
demo machine**. It is **not** suitable for the public internet. Exposed
publicly, anyone can read every dataset, event, track and alert, and can drive
the pipeline and alerting endpoints. A reverse proxy providing TLS, rate
limiting and an authenticating gate in front of it is a **prerequisite**, not
an enhancement.

## Filesystem boundary enforcement

Every caller-supplied path is untrusted.

```python
from extreme_weather.security import safe_path

safe_path(settings.data_dir, user_input)   # resolves, or raises
```

* Both sides are `resolve()`d, so a symlink pointing out of the tree is caught
  rather than followed.
* The resolved path must be the root or below it.
* **An absolute path is refused** unless it already resolves inside the root.
  Reinterpreting `/etc/passwd` as `<root>/etc/passwd` would silently answer a
  different question than the caller asked.
* Extension allow-list: `.nc`, `.nc4`, `.grb`, `.grib`, `.grb2` by default.
* `must_exist=True` additionally requires the file.

Tested against `../../secret`, `../../../etc/passwd`, `raw/../../outside.nc`,
`..\..\secret.nc`, `./../secret.nc`, an absolute path outside the root, and a
symlink escaping the root.

Turn off only with `ENFORCE_DATA_DIR_BOUNDARY=false`, which is not recommended
and is documented as such.

## Filename validation

`validate_filename()` rejects empty names, names over 255 characters, names
containing a path separator, `.`/`..`/leading `..`, control characters, and the
characters `< > : " | ? *`.

## Subprocess safety

`safe_subprocess()` enforces:

* an **argument list**, never a string - `shell=False` always;
* a **mandatory timeout**, so a hung child cannot pin a worker thread;
* captured output rather than inherited stdio.

A filename containing a space or a semicolon is therefore one argument, not a
second command.

## Credential handling

* No credential is stored in the repository, the compose file or any Dockerfile.
* `WEATHER_API_TOKEN` and `OPEN_METEO_API_KEY` are read from the environment.
* A token is sent as an **`Authorization` header**, never as a query parameter,
  because a URL ends up in logs and crash reports.
* Every log line and structured field is passed through `redact()`.
* URLs are redacted in manifests, logs and API responses: userinfo stripped and
  credential-shaped query values masked, keeping the parameter *name* so a
  reader can still see a key was meant to be there.

Tested with `api_key=`, `API-KEY:`, `token=`, `password=`,
`Authorization: Bearer`, `?apikey=`, `{"api_key": ...}` and a field literally
named `api_key`.

## CORS

`CORS_ORIGINS` is an explicit comma-separated allow-list. A wildcard requires
`ALLOW_CORS_WILDCARD=true`, which defaults to **false** - a wildcard origin lets
any website read this API from a browser, and `allow_credentials=True` alongside
it is worse still.

The container sets the origins explicitly rather than inheriting a default.

## API error sanitisation

* **500** never exposes a stack trace, a filesystem path or an internal
  exception message. The detail goes to the log; the client receives the
  request id.
* **4xx** and **503** pass the handler's own message through, redacted. A 503
  saying "run `scripts/train_gnn.py`" is actionable guidance, not a leak.

## Artifact integrity

`check_checkpoint()` returns `VALID` / `INCOMPATIBLE` / `CORRUPT` / `MISSING`
and **never raises**, so a caller can report the state rather than crashing
inside a tensor constructor.

A hash match proves the bytes are intact; it does not prove they are the right
*model*. That is why the checkpoint's recorded `model_version` is compared
against what the caller expects - an intact file holding the wrong model is
`INCOMPATIBLE`, a different problem from `CORRUPT`.

Manifests are verified the same way, including re-hashing the dataset the
manifest describes, which catches a file modified after ingestion.

Tests deliberately corrupt a checkpoint (garbage bytes, truncation) and assert
it is refused.

## Container hardening

| Control | Detail |
| --- | --- |
| Non-root runtime | uid/gid 1001, `nologin` shell |
| No baked secrets | every setting via the environment |
| Explicit volumes | `./data` and `./models`; chown guidance in the compose file |
| Liveness vs readiness | `/health` for restart policy, `/ready` for traffic gating |
| Graceful stop | `stop_grace_period: 30s` so a request is not severed |
| Frontend gating | `depends_on: service_healthy` |

The `user: "1001:1001"` line pairs with a host directory owned by that uid; a
bind mount owned by another uid is the usual reason a hardened container cannot
start.

## Input validation

* Pydantic models with `extra="forbid"` reject unrecognised body fields rather
  than ignoring them.
* Bounded numeric ranges on path, port, threshold and duration settings.
* `LOG_FORMAT` and other enums fail at startup, not at first use.
* `api_max_detail_chars` bounds error text.

## Report of what was audited and found

| Finding | Severity | Resolution |
| --- | --- | --- |
| Container ran as root | High | Non-root user added |
| No readiness probe distinct from liveness | Medium | `/ready` added; frontend now waits for `service_healthy` |
| 503 messages indistinguishable from internal errors | Medium | Authorship-based masking rule |
| Long paths unbounded in logs | Low | Truncated at 512 characters |
| No artifact integrity check | Medium | SHA-256 + metadata check added |
| CORS defaults permissive in container | Low | Explicit origins, wildcard off |

## Outstanding, and why

| Gap | Why it is not closed here |
| --- | --- |
| Authentication | Out of scope for a prototype; faking it would be worse than its absence. Requires a real identity provider and a session model. |
| Authorisation | Depends on authentication. |
| Rate limiting | Belongs at the edge; the app is single-process and prototype-scale. |
| TLS | Terminates at a reverse proxy, which the deployment must provide. |
| Secret storage | Environment variables are the honest minimum; a secrets manager is a deployment decision. |
| Dependency scanning | No new dependencies were added; a scanner belongs in CI. |
| Multi-process state | Run history is file-backed, so two workers would race. Documented in PRODUCTION_READINESS.md. |

## Verification

```bash
python -m pytest tests/test_phase10_core.py -v      # security + integrity
python -m pytest tests/test_phase10_operations.py -v # error schema, readiness
docker compose config                                # compose renders
```

The security tests are the ones to read first: they attempt the traversal shapes
and the corruption cases rather than asserting a flag is set.

## Non-operational disclaimer

**This system is a research prototype and is not an operational
disaster-warning system.** The controls above protect a research deployment. They
do not make this production-ready, and passing them does not establish forecast
skill, accuracy, or any operational capability.
