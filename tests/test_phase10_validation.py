"""Validation-script regression tests (SIH26078, Phase 10).

The deployment validator once failed with ``ECONNRESET`` / "the socket
connection was closed unexpectedly". That was a start-up race in the *harness*,
not a backend fault: importing the application costs ~15-36 s (PyTorch), so a
client that did not wait for readiness saw a refused socket and reported it as
a crash.

These tests pin the behaviour that fixes it, so the race cannot silently
return: an OS-assigned port, output captured on both streams, a readiness
gate, an early-exit report that carries the log, and unconditional cleanup.
They exercise :class:`LiveServer` directly and never bind a fixed port, so
they are deterministic and safe to run in parallel.
"""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_deployment.py"


def _load_validator() -> Any:
    """Import the validator script, which is not an installed package."""
    spec = importlib.util.spec_from_file_location("validate_deployment", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validator() -> Any:
    """Return the loaded validation module."""
    return _load_validator()


class _FakeResponse:
    """Minimal stand-in for ``httpx.Response``."""

    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


def _break_the_app_target(validator: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make uvicorn exit at once by pointing it at an unimportable module."""
    original_popen = validator.subprocess.Popen

    def failing_popen(command: list[str], **kwargs: Any) -> Any:
        broken = list(command)
        broken[broken.index("extreme_weather.api.main:app")] = "no_such_module:app"
        return original_popen(broken, **kwargs)

    monkeypatch.setattr(validator.subprocess, "Popen", failing_popen)


# --------------------------------------------------------------------------- #
# Port allocation
# --------------------------------------------------------------------------- #


def test_the_port_is_assigned_by_the_os(validator: Any) -> None:
    """A hard-coded port collides with whatever else is listening."""
    port = validator._free_port()

    assert isinstance(port, int)
    assert 1024 < port <= 65535


def test_two_allocations_do_not_collide(validator: Any) -> None:
    """Consecutive calls must not hand back the same number."""
    assert validator._free_port() != validator._free_port()


# --------------------------------------------------------------------------- #
# Start-up failure must be explained, not merely timed out
# --------------------------------------------------------------------------- #


def test_a_start_up_failure_reports_the_server_log(
    validator: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead server must be diagnosed from its log, not reported as a timeout.

    This is the difference between a useful CI failure and a mysterious
    ECONNRESET. Without the early-exit branch the caller would wait out the
    full deadline and learn nothing about why the server never answered.
    """
    _break_the_app_target(validator, monkeypatch)
    server = validator.LiveServer(validator._free_port(), tmp_path / "uvicorn.log", {})

    with server:
        ready, detail = server.wait_until_ready(timeout=90.0)

    assert ready is False
    assert "exited during start-up" in detail
    # The traceback the operator actually needs.
    assert "no_such_module" in detail
    # A bare timeout would have been the unhelpful outcome.
    assert "not ready after" not in detail

# --------------------------------------------------------------------------- #
# The readiness gate itself
# --------------------------------------------------------------------------- #


def test_readiness_waits_instead_of_failing_fast(
    validator: Any, tmp_path: Path
) -> None:
    """A server slow to start must still be waited for.

    This is the regression that produced ECONNRESET. The stub transport refuses
    every connection for the first few polls, exactly as uvicorn does while
    importing the app; a harness that gave up would report a crash.
    """
    calls: list[int] = []

    class SlowClient:
        """Refuses connections, then starts answering."""
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> "SlowClient":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def get(self, url: str) -> Any:
            calls.append(1)
            if url.endswith("/health") and len(calls) < 4:
                raise ConnectionRefusedError("not up yet")
            payload = {"state": "DEGRADED"} if url.endswith("/ready") else {}
            return _FakeResponse(200, payload)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(validator.httpx, "Client", SlowClient)
    try:
        server = validator.LiveServer(validator._free_port(), tmp_path / "u.log", {})
        # No process is running, so the early-exit branch must not fire; the
        # stub transport stands in for the real one.
        server.process = None
        ready, state = server.wait_until_ready(timeout=30.0)
    finally:
        monkey.undo()

    assert ready is True
    assert state == "DEGRADED"
    assert len(calls) >= 4, "the gate did not retry before succeeding"


def test_a_not_ready_server_is_rejected(validator: Any, tmp_path: Path) -> None:
    """A NOT_READY server must fail the gate, not pass as serving.

    ``DEGRADED`` is accepted because this project ships without checkpoints;
    ``NOT_READY`` means the process cannot serve and must not be probed.
    """

    class NotReadyClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> "NotReadyClient":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def get(self, url: str) -> Any:
            state = "DEGRADED" if url.endswith("/health") else "NOT_READY"
            return _FakeResponse(200, {"state": state})

    monkey = pytest.MonkeyPatch()
    monkey.setattr(validator.httpx, "Client", NotReadyClient)
    try:
        server = validator.LiveServer(validator._free_port(), tmp_path / "u.log", {})
        server.process = None
        ready, detail = server.wait_until_ready(timeout=10.0)
    finally:
        monkey.undo()

    assert ready is False
    assert "NOT_READY" in detail


# --------------------------------------------------------------------------- #
# The script's own wiring
# --------------------------------------------------------------------------- #


def test_both_api_probes_are_registered(validator: Any) -> None:
    """The in-process *and* live-server probes must both run.

    Dropping either would silently weaken the validation: the in-process probe
    cannot see a port or a uvicorn fault, and the live probe is the only thing
    that exercises the deployment.
    """
    source = inspect.getsource(validator.main)

    assert "_check_api_in_process" in source
    assert "check_api_live_server" in source


def test_the_live_check_probes_the_operations_surface(validator: Any) -> None:
    """The Phase 10 endpoints must be exercised over the real socket."""
    source = inspect.getsource(validator.check_api_live_server)

    for path in (
        "/health",
        "/ready",
        "/api/v1/operations/metrics",
        "/api/v1/operations/integrity",
    ):
        assert path in source


def test_the_live_check_captures_both_output_streams(validator: Any) -> None:
    """A start-up traceback goes to stderr, so it must reach the log file."""
    source = inspect.getsource(validator.LiveServer)

    assert "stderr=subprocess.STDOUT" in source


def test_cleanup_escalates_to_kill(validator: Any) -> None:
    """A server that ignores terminate must still not survive validation."""
    source = inspect.getsource(validator.LiveServer.__exit__)

    assert "process.terminate()" in source
    assert "process.kill()" in source



def test_a_failed_start_leaves_no_running_server(
    validator: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The context manager must not leak the process on the failure path."""
    _break_the_app_target(validator, monkeypatch)
    server = validator.LiveServer(validator._free_port(), tmp_path / "u.log", {})

    with server:
        server.wait_until_ready(timeout=90.0)

    assert server.process is not None
    assert server.process.poll() is not None, "server process was left running"
    # The log handle is released, so the scratch directory can be cleaned up.
    assert server._log_handle is None
