"""Display formatting helpers used by the dashboard and reports."""

from __future__ import annotations

import json
import math
from typing import Any


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def format_number(value: Any, decimals: int = 0) -> str:
    if _is_missing(value):
        return "-"
    return f"{float(value):,.{decimals}f}"


def format_currency(value: Any, currency: str | None = None, decimals: int = 2) -> str:
    """Format money with the record currency. Never relabels or converts."""
    if _is_missing(value):
        return "-"
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{float(value):,.{decimals}f}"


def format_compact_currency(value: Any, currency: str | None = None) -> str:
    if _is_missing(value):
        return "-"
    amount = float(value)
    prefix = f"{currency} " if currency else ""
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(amount) >= threshold:
            return f"{prefix}{amount / threshold:,.1f}{suffix}"
    return f"{prefix}{amount:,.2f}"


def format_percent(value: Any, decimals: int = 1) -> str:
    if _is_missing(value):
        return "-"
    return f"{float(value) * 100:,.{decimals}f}%"


def format_ratio(value: Any, decimals: int = 2) -> str:
    if _is_missing(value):
        return "-"
    return f"{float(value):,.{decimals}f}x"


def safe_json_dumps(value: Any) -> str:
    """Serialize arbitrary metadata to JSON without raising on odd types."""
    def default(obj: Any) -> str:
        return str(obj)

    try:
        return json.dumps(value, default=default, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def truncate(text: Any, length: int = 80) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= length else text[: length - 1] + "\u2026"
