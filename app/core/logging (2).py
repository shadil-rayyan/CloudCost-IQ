"""Logging configuration with safe, non-sensitive formatting."""

from __future__ import annotations

import logging
import sys
from typing import Any

_CONFIGURED = False


class _SafeFormatter(logging.Formatter):
    """Formatter that never prints billing payloads by accident."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        record.message = record.getMessage() if isinstance(record.msg, str) else str(record.msg)
        if record.exc_info:
            record.exc_text = self.formatException(record.exc_info)
        message = f"{self.formatTime(record, self.datefmt)} | {record.levelname:<8} | {record.name} | {record.message}"
        if record.exc_text:
            message = f"{message}\n{record.exc_text}"
        return message


def configure_logging(level: str = "INFO", stream: Any | None = None) -> None:
    """Configure a single stderr handler for the whole application."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(_SafeFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # Keep noisy third-party libraries quieter than the application.
    for noisy in ("urllib3", "matplotlib", "mlflow", "botocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
