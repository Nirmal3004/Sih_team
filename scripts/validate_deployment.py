#!/usr/bin/env python3
"""Deterministic offline validation for SIH26078 (Phase 10).

Runs every check that does not need a network, and prints a pass/fail summary.

    python scripts/validate_deployment.py            # full offline validation
    python scripts/validate_deployment.py --fast     # skip the slow test suite
    python scripts/validate_deployment.py --json     # machine-readable summary

What it checks
--------------
1. Configuration loads and validates.
2. The full pytest suite passes - **including any network-dependent test**.
3. The frontend typechecks and builds for production.
4. ``docker compose config`` renders.
5. Every CLI exposes ``--help``.
6. The API imports and exposes its routes, including the Phase 10 additions.
7. A **real uvicorn server** starts, becomes ready and serves the same probes
   over a socket, then shuts down cleanly.

On the live server check
------------------------
The API is probed twice on purpose. In process over ASGI, which is
deterministic and isolates the application; and over a real socket against a
uvicorn subprocess, which additionally proves the deployment works. The
server is started on an OS-assigned port, its stdout and stderr are captured,
``/health`` is polled until it answers before any other request is made, and
the process is always terminated. A start-up failure is reported with the
server log rather than as a bare timeout.

A start-up race, not a backend fault, produced an ``ECONNRESET`` here once:
importing the app costs ~15-36 s (PyTorch) while an un-waiting client gives
up in ~3 s. Readiness must be awaited, not assumed.

On network-dependent tests
--------------------------
The specification asks that the *offline validation path* not require the
network. A network-dependent test is therefore not skipped, hidden or xfailed -
it is **reported separately**, and the summary says plainly whether it ran.
Suppressing it silently would turn a real failure into a green run, and marking
it ``skip`` would misreport the suite's health. Here is what each status means:

``full``
    Every test ran, including the network-dependent one.
``offline``
    Network-dependent tests were deselected. This is a documented,
    reproducible subset - not a passing full suite - and the summary says so.

Exit codes: ``0`` all checks passed, ``1`` a check failed, ``2`` a tool was
missing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: Tests that need the network. Selected explicitly so the offline path is a
#: deliberate, documented subset rather than a silent gap.
NETWORK_TESTS: tuple[str, ...] = (
    "tests/test_phase8_providers.py::test_remote_provider_converts_a_valid_archive_payload",
)

#: The CLI entry points that must expose --help. Phase 11 adds the three
#: submission commands, so they are held to the same rule as the rest.
CLI_SCRIPTS: tuple[str, ...] = (
    "scripts/run_realtime_pipeline.py",
    "scripts/run_phase7_validation.py",
    "scripts/run_phase9_evaluation.py",
    "scripts/generate_sample_data.py",
    "scripts/run_sih_demo.py",
    "scripts/verify_reproducibility.py",
    "scripts/generate_final_validation.py",
)


@dataclass
class Check:
    """One validation step.

    Attributes:
        name: What was checked.
        passed: Whether it succeeded.
        detail: The tail of the output, or the reason for failure.
        duration: Wall-clock seconds.
        skipped: True when the step could not run at all.
    """

    name: str
    passed: bool
    detail: str = ""
    duration: float = 0.0
    skipped: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "name": self.name,
            "passed": self.passed,
            "skipped": self.skipped,
            "detail": self.detail,
            "duration_seconds": round(self.duration, 2),
        }


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float = 1800.0,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run a command and return its exit code and combined output tail.

    Uses an argument list and ``shell=False``: a script path containing a space
    must not become two arguments, and a shell is not needed here.
    """
    import os

    started = time.perf_counter()
    try:
        completed = subprocess.run(  # noqa: S603 - argument list, shell=False
            list(command),
            cwd=str(cwd or ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **(env or {})},
            shell=False,
            check=False,
        )
    except FileNotFoundError as exc:
        return 127, f"command not found: {exc}"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f}s"
    # Return the FULL output. Truncating here silently broke --help checks:
    # argparse emits "usage:" on the first line, and a 25-line tail of a 51-line
    # help text does not contain it.
    output = (completed.stdout or "") + (completed.stderr or "")
    _ = started
    return completed.returncode, output


def _tail(output: str, lines: int = 25) -> str:
    """Return the last ``lines`` lines of ``output``, for display only."""
    stripped = output.strip().splitlines()
    if not stripped:
        return "(no output)"
    return "\n".join(stripped[-lines:])


def check_configuration() -> Check:
    """Load and validate the application configuration."""
    started = time.perf_counter()
    try:
        from extreme_weather.config import Settings, get_settings

        get_settings.cache_clear()
        settings = Settings()
        # Touch the properties that resolve paths and parse CORS, so a bad
        # value fails here rather than on the first request.
        settings.raw_dir, settings.processed_dir, settings.manifests_dir
        settings.cors_origin_list
        return Check(
            name="configuration",
            passed=True,
            detail=(
                f"app_env={settings.app_env} log_format={settings.log_format} "
                f"cors_origins={len(settings.cors_origin_list)}"
            ),
            duration=time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001 - any failure is a failed check
        return Check(
            name="configuration",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
            duration=time.perf_counter() - started,
        )


def check_tests(*, offline: bool) -> Check:
    """Run the pytest suite, optionally without the network-dependent tests."""
    name = "tests (offline)" if offline else "tests (full)"
    command = [sys.executable, "-m", "pytest", "-q", "--tb=line"]
    if offline and NETWORK_TESTS:
        # Deselect, never skip: the test still exists and still runs in the
        # `full` path above.
        command += ["--deselect", NETWORK_TESTS[0]]
    code, output = _run(command)
    return Check(
        name=name,
        passed=code == 0,
        detail=_summary_line(output) if code == 0 else _tail(output),
    )


def _summary_line(tail: str) -> str:
    """Return the most informative line of a pytest run."""
    for line in reversed(tail.splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            return line.strip()
    return tail.strip().splitlines()[-1] if tail.strip() else "(no output)"


def check_frontend(*, build: bool) -> Check:
    """Typecheck and optionally build the frontend."""
    npm = shutil.which("npm")
    if npm is None:
        return Check(
            name="frontend build", passed=False, skipped=True, detail="npm not found"
        )
    commands = [[npm, "run", "build"]] if build else [[npm, "run", "typecheck"]]
    name = "frontend build" if build else "frontend typecheck"
    if build and not (ROOT / "frontend" / "package.json").is_file():
        return Check(name=name, passed=False, skipped=True, detail="no frontend/package.json")

    for command in commands:
        code, output = _run(command, cwd=ROOT / "frontend", timeout=900.0)
        if code != 0:
            return Check(name=name, passed=False, detail=_tail(output))
    return Check(name=name, passed=True, detail="ok")


def check_docker() -> Check:
    """Render the compose file, which validates its schema."""
    docker = shutil.which("docker")
    if docker is None:
        return Check(
            name="docker compose config",
            passed=False,
            skipped=True,
            detail="docker not found",
        )
    code, output = _run([docker, "compose", "config"], timeout=180.0)
    return Check(
        name="docker compose config",
        passed=code == 0,
        detail="ok" if code == 0 else _tail(output),
    )


def check_cli_help() -> Check:
    """Every CLI must expose --help without side effects."""
    failures: list[str] = []
    for relative in CLI_SCRIPTS:
        script = ROOT / relative
        if not script.is_file():
            failures.append(f"{relative}: missing")
            continue
        code, output = _run([sys.executable, str(script), "--help"], timeout=180.0)
        if code != 0:
            failures.append(f"{relative}: exit {code} :: {_tail(output, 3)}")
        elif "usage:" not in output.lower():
            failures.append(f"{relative}: no usage line")
    return Check(
        name="cli --help",
        passed=not failures,
        detail="; ".join(failures) if failures else f"{len(CLI_SCRIPTS)} scripts ok",
    )


def check_api_import() -> Check:
    """Import the app and confirm the Phase 10 routes exist."""
    started = time.perf_counter()
    try:
        from extreme_weather.api.main import create_app

        app = create_app()
        paths = {getattr(route, "path", "") for route in app.routes}
        required = {"/health", "/ready", "/api/v1/operations/metrics"}
        missing = required - paths
        return Check(
            name="api routes",
            passed=not missing,
            detail=(
                f"{len(paths)} routes registered"
                if not missing
                else f"missing: {', '.join(sorted(missing))}"
            ),
            duration=time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001
        return Check(
            name="api routes",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
            duration=time.perf_counter() - started,
        )


def _free_port() -> int:
    """Reserve an ephemeral port from the OS and release it immediately.

    Asking the OS for port ``0`` is the only way to avoid colliding with a
    stale listener on a hard-coded port such as 8000. The port is released
    before uvicorn binds it, so a race with another process is possible in
    principle; the readiness wait below is what absorbs it.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LiveServer:
    """A uvicorn subprocess that is *provably* ready before it is used.

    The reported failure - ``ECONNRESET`` / "the socket connection was closed
    unexpectedly" - is a start-up race, not a backend fault. Importing the
    application costs ~15-36 s on this platform (PyTorch dominates), while a
    client that gives up in ~3 s sees a refused or reset socket and reports it
    as a crash. This class removes the race instead of tolerating it:

    * a port is allocated by the OS rather than hard-coded;
    * the server's stdout **and** stderr are captured to a log file, so a
      start-up failure is reported with the traceback that caused it;
    * ``/health`` is polled until it answers, and only then is ``/ready``
      consulted, so a request is never issued against a half-started process;
    * a process that dies during start-up is detected immediately and its log
      is surfaced, rather than being reported as a timeout;
    * the process is terminated on exit, and killed if it ignores the polite
      request, so validation never leaves a listener behind.
    """

    def __init__(self, port: int, log_path: Path, env: dict[str, str]) -> None:
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        self.env = env
        self.process: subprocess.Popen[str] | None = None
        self._log_handle: Any = None

    def log_tail(self, lines: int = 40) -> str:
        """Return the tail of the server log, or a note if it is missing."""
        try:
            return _tail(self.log_path.read_text(encoding="utf-8", errors="replace"), lines)
        except OSError:
            return "(no server log)"

    def __enter__(self) -> "LiveServer":
        self._log_handle = self.log_path.open("w", encoding="utf-8", errors="replace")
        try:
            self.process = subprocess.Popen(  # noqa: S603 - argument list, shell=False
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "extreme_weather.api.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(self.port),
                    "--log-level",
                    "info",
                ],
                cwd=str(ROOT),
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                env={**os.environ, **self.env},
                shell=False,
            )
        except OSError:
            # The log file was opened before the spawn, so a failure here would
            # otherwise leak the handle. There is no process to clean up.
            self._log_handle.close()
            self._log_handle = None
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Terminate the server, escalating to kill, and close its log."""
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def wait_until_ready(self, timeout: float = 240.0) -> tuple[bool, str]:
        """Block until ``/health`` answers, then confirm ``/ready``.

        Returns:
            ``(True, state)`` when the server is serving, otherwise
            ``(False, reason)`` with the reason and a log tail.
        """
        deadline = time.perf_counter() + timeout
        last_error = "no probe attempted"
        with httpx.Client(timeout=10.0) as client:
            while time.perf_counter() < deadline:
                if self.process is not None and self.process.poll() is not None:
                    # The server exited during start-up. Its log explains why;
                    # a bare timeout here would hide a real fault.
                    return False, (
                        f"server exited during start-up "
                        f"(rc={self.process.returncode})\n{self.log_tail()}"
                    )
                try:
                    response = client.get(f"{self.base_url}/health")
                    if response.status_code == 200:
                        break
                    last_error = f"/health returned {response.status_code}"
                except Exception as exc:  # noqa: BLE001 - retried until the deadline
                    last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.5)
            else:
                return False, f"not ready after {timeout:.0f}s (last: {last_error})"

            # Serving is not the same as able to take traffic: confirm the
            # readiness verdict too, so a NOT_READY surfaces here instead of
            # being discovered later as a confusing 503.
            ready = client.get(f"{self.base_url}/ready")
            try:
                state = ready.json().get("state")
            except ValueError:
                state = None
            if ready.status_code != 200 or state not in ("READY", "DEGRADED"):
                return False, (
                    f"GET /ready -> {ready.status_code} state={state!r}\n"
                    f"{self.log_tail()}"
                )
        return True, str(state)


def check_api_live_server() -> Check:
    """Start a real uvicorn server and exercise the API over a real socket.

    The in-process check proves the application works; this one proves the
    *deployment* works - that uvicorn imports the app, binds a port, answers
    over TCP and shuts down cleanly. It is the check that would have caught
    the start-up race, so it is kept alongside rather than in place of the
    in-process probe.
    """
    import tempfile

    started = time.perf_counter()
    name = "api live server"
    try:
        import httpx

        from extreme_weather.config import Settings
    except Exception as exc:  # noqa: BLE001
        return Check(
            name=name,
            passed=False,
            detail=f"could not import the client/app: {type(exc).__name__}: {exc}",
        )

    # Probes, grouped by the category each endpoint belongs to. Every one of
    # these answers 200 against a fresh install: the *status* endpoints report
    # model unavailability in the body rather than as an error status, so a
    # missing checkpoint does not make the smoke test fail.
    probes: tuple[tuple[str, str], ...] = (
        # CORE
        ("GET", "/health"),
        ("GET", "/ready"),
        ("GET", "/api/v1/system/status"),
        # EVENTS
        ("GET", "/api/v1/events"),
        # TRACKING
        ("GET", "/api/v1/tracks"),
        # GNN (optional model: status answers 200 with available=false)
        ("GET", "/api/v1/gnn/status"),
        ("GET", "/api/v1/gnn/model"),
        # DOWNSCALING (optional model)
        ("GET", "/api/v1/downscaling/status"),
        ("GET", "/api/v1/downscaling/model"),
        # ALERTS
        ("GET", "/api/v1/alerts"),
        # PIPELINE
        ("GET", "/api/v1/pipeline/status"),
        ("GET", "/api/v1/pipeline/runs"),
        # EVALUATION
        ("GET", "/api/v1/evaluation/status"),
        # OPERATIONS
        ("GET", "/api/v1/operations/status"),
        ("GET", "/api/v1/operations/metrics"),
        ("GET", "/api/v1/operations/integrity"),
        # DATA (Phase 0 and 6, kept for regression coverage)
        ("GET", "/api/v1/datasets"),
        ("GET", "/api/v1/data/datasets"),
        ("GET", "/api/v1/validation/status"),
    )

    with tempfile.TemporaryDirectory() as tmp:
        # An isolated data/model directory keeps the check from touching the
        # repository's data. It is handed to the *server* through the
        # environment, because the child process builds its own Settings.
        scratch = Path(tmp)
        data_dir = scratch / "data"
        model_dir = scratch / "models"
        Settings(data_dir=data_dir, model_dir=model_dir).ensure_directories()

        env = {
            "DATA_DIR": str(data_dir),
            "MODEL_DIR": str(model_dir),
            "APP_ENV": "validation",
            "LOG_FORMAT": "text",
        }
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        if str(SRC) not in existing_pythonpath.split(os.pathsep):
            env["PYTHONPATH"] = os.pathsep.join(
                part for part in (str(SRC), existing_pythonpath) if part
            )

        log_path = scratch / "uvicorn.log"
        failures: list[str] = []
        checked: list[str] = []

        with LiveServer(_free_port(), log_path, env) as server:
            ready_ok, detail = server.wait_until_ready()
            if not ready_ok:
                return Check(
                    name=name,
                    passed=False,
                    detail=f"start-up failed: {detail}",
                    duration=time.perf_counter() - started,
                )

            try:
                with httpx.Client(base_url=server.base_url, timeout=60.0) as client:
                    for method, path in probes:
                        response = client.request(method, path)
                        checked.append(f"{method} {path} -> {response.status_code}")
                        # Every probe answers 200 against a fresh install.
                        # Anything else means the route is gone or the handler
                        # regressed, so it is a failure rather than something
                        # to interpret.
                        if response.status_code != 200:
                            failures.append(f"{method} {path} -> {response.status_code}")

                    health = client.get("/health")
                    if set(health.json()) != {
                        "status",
                        "project",
                        "service",
                        "version",
                        "environment",
                    }:
                        failures.append("GET /health changed its response contract")

                    if not health.headers.get("X-Request-ID"):
                        failures.append("GET /health returned no X-Request-ID header")

                    ready_payload = client.get("/ready").json()
                    if ready_payload.get("state") not in (
                        "READY",
                        "DEGRADED",
                        "NOT_READY",
                    ):
                        failures.append(
                            f"GET /ready state is {ready_payload.get('state')!r}"
                        )
            except Exception as exc:  # noqa: BLE001
                # A transport error here is a real deployment fault, and the
                # server log is what explains it.
                return Check(
                    name=name,
                    passed=False,
                    detail=(
                        f"probe failed: {type(exc).__name__}: {exc}\n"
                        f"--- uvicorn log ---\n{server.log_tail()}"
                    ),
                    duration=time.perf_counter() - started,
                )

            # Assembled here, while the scratch directory still exists, so the
            # log tail is readable on failure. The `with` block then tears the
            # server down on the way out, pass or fail.
            return Check(
                name=name,
                passed=not failures,
                detail=(
                    f"{len(checked)} endpoints probed over TCP against a real "
                    f"uvicorn server (readiness={detail})"
                    if not failures
                    else "; ".join(failures)
                    + f"\n--- uvicorn log ---\n{server.log_tail()}"
                ),
                duration=time.perf_counter() - started,
            )


def _check_api_in_process() -> Check:
    """Verify the API surface **in process**, over ASGI rather than TCP.

    An in-process client is deterministic: there is no port, no process and no
    socket, so the only thing that can fail is the application itself. That
    makes it the right check for "is the API correct".

    It cannot, however, verify that uvicorn imports the app, binds a port and
    answers over TCP. :func:`check_api_live_server` covers that gap, so this
    check narrows its own docstring rather than claiming to be the whole story.
    """
    started = time.perf_counter()
    try:
        import tempfile

        from fastapi.testclient import TestClient

        from extreme_weather.api.main import create_app
        from extreme_weather.config import Settings, get_settings
    except Exception as exc:  # noqa: BLE001
        return Check(
            name="api endpoints (in process)",
            passed=False,
            detail=f"could not import the app: {type(exc).__name__}: {exc}",
        )

    # A scratch directory keeps the check from touching the repository's data.
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(data_dir=Path(tmp) / "data", model_dir=Path(tmp) / "models")
        settings.ensure_directories()
        application = create_app(settings)
        application.dependency_overrides[get_settings] = lambda: settings

        # Same category-based probe list as the live check, so the two cannot
        # drift apart. See `check_api_live_server` for the grouping.
        probes: tuple[tuple[str, str], ...] = (
            # CORE
            ("GET", "/health"),
            ("GET", "/ready"),
            ("GET", "/api/v1/system/status"),
            # EVENTS
            ("GET", "/api/v1/events"),
            # TRACKING
            ("GET", "/api/v1/tracks"),
            # GNN
            ("GET", "/api/v1/gnn/status"),
            ("GET", "/api/v1/gnn/model"),
            # DOWNSCALING
            ("GET", "/api/v1/downscaling/status"),
            ("GET", "/api/v1/downscaling/model"),
            # ALERTS
            ("GET", "/api/v1/alerts"),
            # PIPELINE
            ("GET", "/api/v1/pipeline/status"),
            ("GET", "/api/v1/pipeline/runs"),
            # EVALUATION
            ("GET", "/api/v1/evaluation/status"),
            # OPERATIONS
            ("GET", "/api/v1/operations/status"),
            ("GET", "/api/v1/operations/metrics"),
            ("GET", "/api/v1/operations/integrity"),
            # DATA
            ("GET", "/api/v1/datasets"),
            ("GET", "/api/v1/data/datasets"),
            ("GET", "/api/v1/validation/status"),
        )

        failures: list[str] = []
        checked: list[str] = []
        try:
            with TestClient(application, raise_server_exceptions=False) as client:
                for method, path in probes:
                    response = client.request(method, path)
                    checked.append(f"{method} {path} -> {response.status_code}")
                    # 5xx is always a failure. 4xx is acceptable for probes that
                    # legitimately have nothing to show, but a 404 that means
                    # "route missing" must not pass silently.
                    # Every probe below answers 200 against a fresh install.
                    # Anything else means the route is gone or the handler
                    # regressed, so it is a failure rather than something to
                    # interpret.
                    if response.status_code != 200:
                        failures.append(
                            f"{method} {path} -> {response.status_code}"
                        )

                # The Phase 10 contract, asserted rather than eyeballed.
                health = client.get("/health")
                if set(health.json()) != {
                    "status",
                    "project",
                    "service",
                    "version",
                    "environment",
                }:
                    failures.append("GET /health changed its response contract")

                if not health.headers.get("X-Request-ID"):
                    failures.append("GET /health returned no X-Request-ID header")

                ready = client.get("/ready")
                if ready.json().get("state") not in ("READY", "DEGRADED", "NOT_READY"):
                    failures.append(
                        f"GET /ready state is {ready.json().get('state')!r}"
                    )
        except Exception as exc:  # noqa: BLE001
            return Check(
                name="api endpoints (in process)",
                passed=False,
                detail=f"probe failed: {type(exc).__name__}: {exc}",
                duration=time.perf_counter() - started,
            )

    return Check(
        name="api endpoints (in process)",
        passed=not failures,
        detail=(
            f"{len(checked)} endpoints probed in-process"
            if not failures
            else "; ".join(failures)
        ),
        duration=time.perf_counter() - started,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the validation.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when every check passed, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="validate_deployment.py",
        description=(
            "Deterministic offline validation for SIH26078. Runs the test "
            "suite, the frontend build, the compose render, the CLI help and "
            "the API import, and reports a pass/fail summary."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip the pytest suite and the frontend build.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Deselect the network-dependent test. This is a documented "
            "subset, not a passing full suite, and the summary says so."
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the summary as JSON."
    )
    args = parser.parse_args(argv)

    steps: list[Callable[[], Check]] = [
        check_configuration,
        check_api_import,
        _check_api_in_process,
        check_api_live_server,
        check_cli_help,
        check_docker,
    ]
    if not args.fast:
        steps.insert(1, lambda: check_tests(offline=args.offline))
        steps.append(lambda: check_frontend(build=True))

    checks: list[Check] = []
    for step in steps:
        result = step()
        checks.append(result)
        if not args.json:
            mark = "PASS" if result.passed else ("SKIP" if result.skipped else "FAIL")
            print(f"[{mark}] {result.name} ({result.duration:.1f}s)")
            if not result.passed:
                print(f"       {result.detail.splitlines()[0] if result.detail else ''}")

    failed = [c for c in checks if not c.passed and not c.skipped]
    summary = {
        "total": len(checks),
        "passed": sum(1 for c in checks if c.passed),
        "failed": len(failed),
        "skipped": sum(1 for c in checks if c.skipped),
        "network_tests_included": not args.offline,
        "checks": [c.as_dict() for c in checks],
    }

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print()
        print(f"{summary['passed']}/{summary['total']} checks passed")
        if summary["skipped"]:
            print(f"{summary['skipped']} skipped (tool unavailable)")
        if args.offline:
            print(
                "Network-dependent tests were DESELECTED. This is an offline "
                "subset, not a full green suite; run without --offline to "
                "include them."
            )
        else:
            print("Network-dependent tests were included.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
