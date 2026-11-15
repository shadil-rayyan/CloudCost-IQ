"""Date and time helpers.

Billing timestamps are normalized to **timezone-naive UTC**. Keeping a single,
documented timezone (UTC) avoids off-by-one-day errors when aggregating daily
costs across providers. See ``docs/data_dictionary.md``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import pandas as pd

#: Timezone used for every normalized timestamp and daily aggregation.
BILLING_TIMEZONE = "UTC"


def parse_date(value: object) -> dt.date | None:
    """Parse a single date-like value into a ``date`` or ``None``."""
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return pd.Timestamp(value).date()
    except (ValueError, TypeError):
        return None


def to_datetime_utc(series: pd.Series) -> pd.Series:
    """Convert a column to tz-naive UTC timestamps, coercing invalid values.

    Invalid values become ``NaT`` and are surfaced by validation rather than
    being silently dropped here.
    """
    if series.empty:
        return pd.Series([], dtype="datetime64[ns]")
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    else:
        cleaned = series.astype("string").str.strip()
        cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "NaT": pd.NA})
        try:
            parsed = pd.to_datetime(cleaned, errors="coerce", utc=True, format="mixed")
        except (ValueError, TypeError):
            parsed = pd.to_datetime(cleaned, errors="coerce", utc=True)
    return parsed.dt.tz_convert(BILLING_TIMEZONE).dt.tz_localize(None)


def to_date(series: pd.Series) -> pd.Series:
    """Normalize a datetime column to calendar dates (UTC)."""
    return to_datetime_utc(series).dt.normalize()


def month_key(series: pd.Series) -> pd.Series:
    """Return ``YYYY-MM`` invoice month keys for a timestamp column."""
    return to_datetime_utc(series).dt.strftime("%Y-%m")


def month_range(start: dt.date, end: dt.date) -> list[str]:
    """Inclusive list of ``YYYY-MM`` keys between two dates."""
    current = dt.date(start.year, start.month, 1)
    last = dt.date(end.year, end.month, 1)
    months: list[str] = []
    while current <= last:
        months.append(current.strftime("%Y-%m"))
        current = dt.date(
            current.year + (1 if current.month == 12 else 0),
            1 if current.month == 12 else current.month + 1,
            1,
        )
    return months


def missing_dates(observed: Iterable[dt.date], start: dt.date, end: dt.date) -> list[dt.date]:
    """Calendar dates inside ``[start, end]`` that are absent from ``observed``."""
    present = set(observed)
    total = (end - start).days + 1
    if total <= 0:
        return []
    return [start + dt.timedelta(days=offset) for offset in range(total) if start + dt.timedelta(days=offset) not in present]


def date_span_days(start: dt.date | None, end: dt.date | None) -> int:
    if start is None or end is None:
        return 0
    return (end - start).days + 1


def utc_now() -> dt.datetime:
    """Current UTC time as a tz-naive datetime (consistent with storage)."""
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)
