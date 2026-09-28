"""Structured pipeline logging (SIH26078, Phase 8, section 14).

A run should be readable as a narrative. Each stage logs a bracketed tag
matching the pipeline diagram, so a log file can be scanned by eye:

.. code-block:: text

    [INGEST]     provider=open-meteo steps=24 checksum=ab12cd
    [VALIDATE]   status=VALID 9/9 checks passed
    [PREPROCESS] wrote data/processed/realtime.nc
    [ANOMALY]    3 anomalies over 24 steps
    [GNN]        SKIPPED: no trained GNN checkpoint is available

Credential safety
-----------------
:meth:`PipelineLogger.redact` is applied to every message before it is emitted.
It masks anything that looks like a token, which is a backstop rather than the
primary defence: the providers already avoid putting credentials in URLs. The
backstop exists because a third-party error string can contain anything.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

__all__ = ["STAGE_LABELS", "PipelineLogger", "get_logger", "log_stage"]

#: Display labels for each pipeline stage, in execution order.
STAGE_LABELS: dict[str, str] = {
    "ingest": "INGEST",
    "validate": "VALIDATE",
    "preprocess": "PREPROCESS",
    "anomaly": "ANOMALY",
    "track": "TRACK",
    "gnn": "GNN",
    "downscale": "DOWNSCALE",
    "alert": "ALERT",
}

#: Patterns whose values are masked anywhere they appear in a log line.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(api[_-]?key|apikey|token|password|passwd|secret)\b\s*[:=]\s*\S+"),
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)([?&](?:api[_-]?key|apikey|key|token|access_token)=)[^&\s]+"),
)


def redact(text: str) -> str:
    """Mask anything credential-shaped in a string.

    Args:
        text: The text to scrub.

    Returns:
        The text with credential values replaced by ``***``. Non-string input is
        coerced rather than raising, since this is applied to arbitrary error
        messages on the logging path.
    """
    if not isinstance(text, str):
        text = str(text)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_mask, text)
    return text


def _mask(match: re.Match[str]) -> str:
    """Replace a secret match with a masked equivalent, keeping the key name."""
    original = match.group(0)
    for separator in ("=", ":"):
        if separator in original:
            key, _, _ = original.partition(separator)
            return f"{key}{separator}***"
    return "***"


def get_logger(name: str = "extreme_weather.realtime") -> logging.Logger:
    """Return the realtime logger.

    Args:
        name: Logger name.

    Returns:
        A configured :class:`logging.Logger`. Propagation is left on so the
        application logging config in ``main.py`` still controls the handler.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class PipelineLogger:
    """Emits the bracketed stage lines that make a run readable.

    Args:
        logger: The underlying logger; defaults to the realtime logger.
        run_id: Included in every line so concurrent runs can be told apart.
    """

    def __init__(
        self, logger: logging.Logger | None = None, run_id: str = ""
    ) -> None:
        self._logger = logger or get_logger()
        self.run_id = run_id

    def redact(self, text: Any) -> str:
        """Mask credential-shaped content in ``text``."""
        return redact(str(text))

    def stage(self, stage: str, message: str, *, level: int = logging.INFO) -> str:
        """Log one stage line and return the formatted text.

        Args:
            stage: Stage key, e.g. ``anomaly``. An unknown key is used as-is
                rather than raising, so a new stage can be logged before its
                label is registered.
            message: The message, which is redacted before emission.
            level: Log level.

        Returns:
            The formatted line, which is also what tests assert on.
        """
        label = STAGE_LABELS.get(stage, stage.upper())
        prefix = f"{label:<11}" if self.run_id else f"{label:<11}"
        line = f"[{prefix}] {self.redact(message)}"
        self._logger.log(level, line)
        return line

    def info(self, message: str, *, level: int = logging.INFO) -> str:
        """Log an untagged informational line."""
        line = f"[PIPELINE  ] {self.redact(message)}"
        self._logger.log(level, line)
        return line

    def error(self, message: str) -> str:
        """Log an untagged error line."""
        return self.info(message, level=logging.ERROR)


def log_stage(
    stage: str,
    message: str,
    *,
    logger: logging.Logger | None = None,
    run_id: str = "",
    level: int = logging.INFO,
) -> str:
    """Log a single stage line without constructing a :class:`PipelineLogger`.

    Args:
        stage: Stage key.
        message: The message.
        logger: Underlying logger.
        run_id: Run identifier for the line.
        level: Log level.

    Returns:
        The formatted line.
    """
    return PipelineLogger(logger, run_id).stage(stage, message, level=level)
