"""Shared API helpers: data access, filters and JSON-safe serialization."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable
from typing import Any

import pandas as pd

from app.analytics.store import CostFilter, CostStore


def cost_filter(
    *,
    providers: Iterable[str] | None = None,
    billing_accounts: Iterable[str] | None = None,
    services: Iterable[str] | None = None,
    regions: Iterable[str] | None = None,
    currencies: Iterable[str] | None = None,
    cost_basis: str | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> CostFilter:
    return CostFilter(
        providers=tuple(value for value in (providers or ()) if value),
        billing_accounts=tuple(value for value in (billing_accounts or ()) if value),
        services=tuple(value for value in (services or ()) if value),
        regions=tuple(value for value in (regions or ()) if value),
        currencies=tuple(value.upper() for value in (currencies or ()) if value),
        cost_basis=cost_basis,
        start_date=start_date,
        end_date=end_date,
    )


def load_costs(filters: CostFilter | None = None) -> tuple[CostStore, pd.DataFrame]:
    store = CostStore()
    return store, store.load(filters)


def records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe records (NaN becomes null)."""
    if frame is None or frame.empty:
        return []
    working = frame.copy()
    for column in working.columns:
        if pd.api.types.is_datetime64_any_dtype(working[column]):
            working[column] = working[column].map(
                lambda value: None if pd.isna(value) else pd.Timestamp(value).isoformat()
            )
    return [_clean(row) for row in working.to_dict(orient="records")]


def _clean(row: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if value is None or isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            cleaned[str(key)] = None
        elif isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
            cleaned[str(key)] = pd.Timestamp(value).isoformat()
        elif hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                cleaned[str(key)] = value.item()
            except (ValueError, AttributeError):
                cleaned[str(key)] = str(value)
        else:
            try:
                if pd.isna(value):
                    cleaned[str(key)] = None
                    continue
            except (TypeError, ValueError):
                pass
            cleaned[str(key)] = value
    return cleaned
