"""FOCUS version handling."""

from __future__ import annotations

from typing import Final

FOCUS_VERSIONS: Final[tuple[str, ...]] = ("1.2", "1.4")
TARGET_VERSION: Final[str] = "1.4"


def normalize_version(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return TARGET_VERSION
    if text.startswith("1.2"):
        return "1.2"
    if text.startswith("1.4"):
        return "1.4"
    return text


def is_supported(version: str | None) -> bool:
    return normalize_version(version) in FOCUS_VERSIONS
