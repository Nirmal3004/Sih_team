#!/usr/bin/env python3
"""Record the final validation matrix (SIH26078, Phase 11).

Runs every check the submission is judged on and writes
``docs/FINAL_VALIDATION.md`` from the **actual** results. No number in that
document is typed by hand, because a hand-typed validation record is a
liability: it drifts from reality the moment anything changes.

    python scripts/generate_final_validation.py            # run everything
    python scripts/generate_final_validation.py --fast    # skip tests + build

Exit codes: ``0`` every check passed, ``1`` at least one failed.

The counts are reported exactly as pytest produced them. Failures, skips and
xfails are never filtered out or reinterpreted; if the suite is not clean,
this says so.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DOC = ROOT / "docs" / "FINAL_VALIDATION.md"

#: Matches a pytest status count, e.g. "1816 passed" or "2 skipped".
_STATUS = re.compile(r"(\d+)\s+(passed|failed|skipped|xfailed|xpassed|error[s]?)")


@dataclass
class Result:
    """One recorded check."""

    name: str
    passed: bool
    detail: str = ""
    values: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Return the markdown table row for this result."""
        mark = "PASS" if self.passed else "FAIL"
        return f"| {self.name} | **{mark}** | {self.detail} |"


def _which(name: str) -> str | None:
    """Return the path of an executable, or ``None``."""
    import shutil

    return shutil.which(name)


def _tail(output: str, lines: int = 20) -> str:
    """Return the last lines of an output, condensed for a failure message."""
    stripped = output.strip().splitlines()
    if not stripped:
        return "(no output)"
    return " ".join(stripped[-lines:])[:400]


def _run(
    command: list[str], *, cwd: Path | None = None, timeout: float = 3600.0
) -> tuple[int, str]:
    """Run a command and return its exit code and combined output."""
    import os

    try:
        completed = subprocess.run(  # noqa: S603 - argument list, shell=False
            list(command),
            cwd=str(cwd or ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
            shell=False,
            check=False,
        )
    except FileNotFoundError as exc:
        return 127, f"command not found: {exc}"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f}s"
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def _parse_pytest(output: str) -> dict[str, int]:
    """Extract the per-status counts from a pytest run.

    Every status is reported, including the ones we would rather not see. A
    summary that omits "skipped" is a summary that hides a decision.
    """
    counts: dict[str, int] = {}
    for line in reversed(output.splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            for match in _STATUS.finditer(line):
                label = match.group(2).rstrip("s") if match.group(2) != "errors" else "error"
                counts[label] = counts.get(label, 0) + int(match.group(1))
            if counts:
                break
    return counts


def check_tests() -> Result:
    """Run the full pytest suite and record the exact counts."""
    started = time.perf_counter()
    code, output = _run([sys.executable, "-m", "pytest", "-q", "--tb=line"])
    counts = _parse_pytest(output)
    total = sum(
        counts.get(key, 0)
        for key in ("passed", "failed", "skipped", "xfailed", "xpassed", "error")
    )
    duration = time.perf_counter() - started

    parts = [f"{counts.get('passed', 0)} passed"]
    for label in ("failed", "skipped", "xfailed", "xpassed", "error"):
        if counts.get(label):
            parts.append(f"{counts[label]} {label}")
    detail = f"{', '.join(parts)} (total {total}) in {duration:.0f}s"

    return Result(
        name="pytest suite",
        passed=(
            code == 0 and counts.get("failed", 0) == 0 and counts.get("error", 0) == 0
        ),
        detail=detail,
        values={"total": total, **counts},
    )


def check_phase11_tests() -> Result:
    """Count the Phase 11 tests specifically."""
    code, output = _run(
        [sys.executable, "-m", "pytest", "tests/test_phase11_system.py", "--collect-only", "-q"]
    )
    match = re.search(r"(\d+)\s+tests? collected", output)
    count = int(match.group(1)) if match else 0
    return Result(
        name="Phase 11 tests",
        passed=code == 0 and count > 0,
        detail=f"{count} test(s) in tests/test_phase11_system.py",
        values={"count": count},
    )


def check_frontend() -> Result:
    """Build the dashboard."""
    npm = _which("npm")
    if npm is None:
        return Result(name="frontend build", passed=False, detail="npm not found")
    code, output = _run([npm, "run", "build"], cwd=ROOT / "frontend", timeout=900.0)
    return Result(
        name="frontend build",
        passed=code == 0,
        detail="tsc --noEmit + vite build completed" if code == 0 else _tail(output),
    )


def check_docker() -> Result:
    """Render the compose file."""
    docker = _which("docker")
    if docker is None:
        return Result(
            name="docker compose config", passed=False, detail="docker not found"
        )
    code, output = _run([docker, "compose", "config"], timeout=180.0)
    services = len(re.findall(r"^  [a-z][\w-]*:$", output, re.MULTILINE))
    return Result(
        name="docker compose config",
        passed=code == 0,
        detail=f"rendered successfully ({services} service(s))"
        if code == 0
        else _tail(output),
        values={"services": services},
    )


def check_deployment() -> Result:
    """Run the Phase 10 deployment validation."""
    code, output = _run(
        [sys.executable, "-u", "scripts/validate_deployment.py"], timeout=3600.0
    )
    passed = re.findall(r"^\[PASS\]\s+(.+?)\s+\(", output, re.MULTILINE)
    failed = re.findall(r"^\[FAIL\]\s+(.+?)\s+\(", output, re.MULTILINE)
    if code != 0 and not failed:
        failed = ["the validator exited non-zero; see its output"]
    return Result(
        name="deployment validation",
        passed=code == 0,
        detail=f"{len(passed)} check(s) passed" + (f"; failed: {', '.join(failed)}" if failed else ""),
        values={"passed": len(passed), "failed": len(failed)},
    )


def check_reproducibility() -> Result:
    """Verify the deterministic components."""
    code, output = _run(
        [sys.executable, "-u", "scripts/verify_reproducibility.py"], timeout=1800.0
    )
    # The verdicts are right-aligned inside brackets ("[           PASS]"), so
    # the status must be matched with its surrounding whitespace, not as the
    # literal "[PASS]". Getting this wrong silently reports 0 checks, which
    # looks like a pass.
    passed = len(re.findall(r"\[\s*PASS\s*\]", output))
    failed = len(re.findall(r"\[\s*FAIL\s*\]", output))
    na = len(re.findall(r"\[\s*NOT_APPLICABLE\s*\]", output))
    return Result(
        name="reproducibility",
        passed=code == 0 and failed == 0 and passed > 0,
        detail=f"{passed} passed, {failed} failed, {na} not applicable",
        values={"passed": passed, "failed": failed, "not_applicable": na},
    )


def check_demo(*, dry_run: bool) -> Result:
    """Run the SIH demo pipeline."""
    args = [sys.executable, "-u", "scripts/run_sih_demo.py"]
    if dry_run:
        args.append("--dry-run")
    else:
        args.append("--reuse-existing")
    code, output = _run(args, timeout=1800.0)
    ok = re.findall(r"^\s+\[\s*OK\]", output, re.MULTILINE)
    unavailable = re.findall(r"^\s+\[UNAVAILABLE\]", output, re.MULTILINE)
    mode = "dry run" if dry_run else "live"
    return Result(
        name=f"demo pipeline ({mode})",
        passed=code == 0,
        detail=(
            f"exit 0; {len(ok)} stage(s) OK, {len(unavailable)} unavailable "
            "(documented, not failures)"
            if code == 0
            else _tail(output)
        ),
        values={"ok": len(ok), "unavailable": len(unavailable)},
    )


def check_api() -> Result:
    """Probe the API surface in process, for health and readiness."""
    code, output = _run(
        [sys.executable, "-c", _API_PROBE], timeout=600.0
    )
    return Result(
        name="API health / readiness",
        passed=code == 0,
        detail=output.strip().splitlines()[-1] if output.strip() else _tail(output),
    )


#: Probes the endpoints a judge will actually open, in process over ASGI so no
#: port is bound. Prints a one-line summary and exits non-zero on any failure.
_API_PROBE = r"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, r"{src}")
from fastapi.testclient import TestClient
from extreme_weather.api.main import create_app
from extreme_weather.config import Settings, get_settings

with tempfile.TemporaryDirectory() as tmp:
    settings = Settings(data_dir=Path(tmp) / "data", model_dir=Path(tmp) / "models")
    settings.ensure_directories()
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app, raise_server_exceptions=False) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        status = client.get("/api/v1/system/status")
        state = ready.json().get("state") if ready.status_code == 200 else "?"
        ok = health.status_code == 200 and ready.status_code == 200 and status.status_code == 200
        print(
            f"/health {{health.status_code}}, /ready {{ready.status_code}} "
            f"({{state}}), /api/v1/system/status {{status.status_code}}"
        )
        sys.exit(0 if ok else 1)
""".format(src=str(SRC).replace("\\", "\\\\"))


def render(results: list[Result], *, started: str) -> str:
    """Render the validation document from the recorded results."""
    failed = [r for r in results if not r.passed]
    lines: list[str] = []
    add = lines.append

    add("# Final validation (SIH26078, Phase 11)")
    add("")
    add("> **Generated file.** Every number below was produced by running the")
    add("> command shown against this repository. Nothing here is typed by hand.")
    add(">")
    add("> Regenerate with `python scripts/generate_final_validation.py`.")
    add("")
    add("**This project is a research prototype.** It demonstrates no forecast")
    add("skill, and no accuracy, lives-saved or economic-benefit figure has been")
    add("measured. Where a quantity could not be computed it is reported as")
    add("unavailable, never as zero.")
    add("")
    add(f"Generated: {started}")
    add("")

    add("## Summary")
    add("")
    add(f"**{len(results) - len(failed)}/{len(results)} checks passed.**")
    add("")
    add("| Check | Result | Detail |")
    add("| --- | --- | --- |")
    for result in results:
        add(result.render())
    add("")

    add("## Test counts")
    add("")
    for result in results:
        if result.name == "pytest suite":
            values = result.values
            add("| Metric | Value |")
            add("| --- | --- |")
            add(f"| Total tests | {values.get('total', 0)} |")
            add(f"| Passed | {values.get('passed', 0)} |")
            add(f"| Failed | {values.get('failed', 0)} |")
            add(f"| Skipped | {values.get('skipped', 0)} |")
            add(f"| Xfailed | {values.get('xfailed', 0)} |")
            add(f"| Errors | {values.get('error', 0)} |")
            add("")
            add(
                "Skipped and xfailed are reported as they are. A non-zero count "
                "in either column would mean a test had been suppressed, which "
                "this project does not do."
            )
    add("")

    add("## Reproducibility")
    add("")
    for result in results:
        if result.name == "reproducibility":
            add(
                f"{result.values.get('passed', 0)} passed, "
                f"{result.values.get('failed', 0)} failed, "
                f"{result.values.get('not_applicable', 0)} not applicable."
            )
            add("")
            add(
                "Two independent pipeline runs with identical seeded inputs "
                "produce identical event ids, track ids, alert ids, graph "
                "structure, provenance and run ids. Timestamps and durations "
                "are excluded because they are intentionally volatile."
            )
    add("")

    if failed:
        add("## Failures")
        add("")
        for result in failed:
            add(f"- **{result.name}** — {result.detail}")
        add("")
    else:
        add("## Notes")
        add("")
        add(
            "No check failed. A `UNAVAILABLE` demo stage (for example the "
            "evaluation, when no ground-truth reference exists) is a documented "
            "outcome, not a failure, and is reported as such rather than being "
            "converted into a zero."
        )
        add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the checks and write the validation document.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when every check passed, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="generate_final_validation.py",
        description=(
            "Run every final check and regenerate docs/FINAL_VALIDATION.md "
            "from the actual results."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip the pytest suite and the frontend build (the slow checks).",
    )
    args = parser.parse_args(argv)

    from datetime import datetime, timezone

    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    results: list[Result] = []
    steps: list[tuple[str, Any]] = []
    if not args.fast:
        steps.append(("pytest suite", check_tests))
        steps.append(("Phase 11 tests", check_phase11_tests))
    steps.extend(
        [
            ("frontend build", check_frontend),
            ("docker compose config", check_docker),
            ("API health / readiness", check_api),
            ("reproducibility", check_reproducibility),
            ("demo pipeline (dry run)", lambda: check_demo(dry_run=True)),
            ("demo pipeline (live)", lambda: check_demo(dry_run=False)),
            ("deployment validation", check_deployment),
        ]
    )

    for name, step in steps:
        print(f"... {name}", flush=True)
        try:
            result = step()
        except Exception as exc:  # noqa: BLE001 - a crashed check is a failed check
            result = Result(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}")
        print(f"    {'PASS' if result.passed else 'FAIL'}: {result.detail}", flush=True)
        results.append(result)

    document = render(results, started=started)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(document, encoding="utf-8")

    failed = [r for r in results if not r.passed]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    print(f"written: {DOC.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())



