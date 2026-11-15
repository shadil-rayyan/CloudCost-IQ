"""Billing data-quality analysis.

A missing billing day is **not** automatically a zero-cost day. This module
distinguishes:

* days that contain billing records (positive or zero cost),
* days inside a series' observed date span with no records (candidate missing),
* days outside the observed span (unknown / not covered).

Only the ML dataset builder may create zero-cost observations, and only when a
day is explicitly marked as complete.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from typing import Any

import pandas as pd

from app.core.constants import NORMALIZED_SCHEMA_VERSION
from app.database.models import ImportRecord
from app.utils.dates import missing_dates

DEFAULT_SERIES_KEYS = ("ServiceProviderName", "BillingAccountId", "ServiceName", "BillingCurrency", "cost_basis")


def _billing_date(frame: pd.DataFrame) -> pd.Series:
    if "ChargePeriodStart" in frame.columns and frame["ChargePeriodStart"].notna().any():
        return pd.to_datetime(frame["ChargePeriodStart"], errors="coerce").dt.normalize()
    return pd.to_datetime(frame["usage_start"], errors="coerce").dt.normalize()


def coverage_by_series(
    frame: pd.DataFrame,
    group_keys: Iterable[str] = DEFAULT_SERIES_KEYS,
) -> pd.DataFrame:
    """Per-series date coverage, record counts and completeness indicators."""
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                *group_keys,
                "first_date",
                "last_date",
                "calendar_days",
                "days_with_records",
                "coverage_ratio",
                "zero_cost_days",
                "missing_days",
                "records",
                "cost",
            ]
        )
    working = frame.copy()
    working["_billing_date"] = _billing_date(working)
    keys = [key for key in group_keys if key in working.columns]
    working = working.loc[working["_billing_date"].notna()]

    rows: list[dict[str, Any]] = []
    for key_values, group in working.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        daily = group.groupby("_billing_date")["BilledCost"].sum()
        observed = [timestamp.date() for timestamp in daily.index]
        first = min(observed)
        last = max(observed)
        calendar_days = (last - first).days + 1
        missing = missing_dates(observed, first, last)
        row: dict[str, Any] = dict(zip(keys, key_values, strict=False))
        row.update(
            {
                "first_date": first.isoformat(),
                "last_date": last.isoformat(),
                "calendar_days": calendar_days,
                "days_with_records": len(observed),
                "coverage_ratio": (len(observed) / calendar_days) if calendar_days else 0.0,
                "zero_cost_days": int((daily.fillna(0) == 0).sum()),
                "missing_days": len(missing),
                "records": len(group),
                "cost": float(pd.to_numeric(group["BilledCost"], errors="coerce").sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def missing_period_details(
    frame: pd.DataFrame,
    *,
    group_keys: Iterable[str] = DEFAULT_SERIES_KEYS,
    max_examples: int = 25,
) -> dict[str, Any]:
    """Summarize candidate missing dates without claiming they are zero cost."""
    coverage = coverage_by_series(frame, group_keys)
    if coverage.empty:
        return {"series_with_gaps": 0, "candidate_missing_days": 0, "examples": []}
    gaps = coverage.loc[coverage["missing_days"] > 0].sort_values("missing_days", ascending=False)
    total_missing = int(coverage["missing_days"].sum())
    examples: list[dict[str, Any]] = []
    if not gaps.empty:
        working = frame.copy()
        working["_billing_date"] = _billing_date(working)
        keys = [key for key in group_keys if key in working.columns]
        for _, row in gaps.head(max_examples).iterrows():
            mask = pd.Series(True, index=working.index)
            for key in keys:
                mask &= working[key].astype("string").fillna("(null)") == str(row.get(key, "(null)"))
            series_dates = sorted(
                pd.to_datetime(working.loc[mask, "_billing_date"]).dropna().dt.date.unique()
            )
            if not series_dates:
                continue
            missing = missing_dates(
                series_dates,
                dt.date.fromisoformat(row["first_date"]),
                dt.date.fromisoformat(row["last_date"]),
            )
            examples.append(
                {
                    "series": {key: row.get(key) for key in keys},
                    "missing_days": int(row["missing_days"]),
                    "first_missing": [value.isoformat() for value in missing[:5]],
                }
            )
    return {
        "series_with_gaps": len(gaps),
        "candidate_missing_days": total_missing,
        "examples": examples,
        "note": (
            "Candidate gaps are days inside a series' observed span without billing "
            "records. They are not converted to zero cost automatically."
        ),
    }


def freshness_report(
    frame: pd.DataFrame,
    imports: list[ImportRecord] | None = None,
) -> dict[str, Any]:
    """Billing data freshness: coverage, latest usage and import timestamps."""
    imports = imports or []
    result: dict[str, Any] = {
        "latest_usage_date": None,
        "earliest_usage_date": None,
        "latest_invoice_month": None,
        "latest_upload_at": None,
        "latest_import_completed_at": None,
        "imports_completed": sum(1 for record in imports if record.status == "completed"),
        "imports_failed": sum(1 for record in imports if record.status == "failed"),
    }
    if frame is not None and not frame.empty:
        dates = _billing_date(frame).dropna()
        if not dates.empty:
            result["latest_usage_date"] = dates.max().date().isoformat()
            result["earliest_usage_date"] = dates.min().date().isoformat()
        if "invoice_month" in frame.columns:
            months = frame["invoice_month"].dropna().astype(str)
            if not months.empty:
                result["latest_invoice_month"] = months.max()
    uploads = sorted(
        (record.uploaded_at for record in imports if record.uploaded_at is not None), reverse=True
    )
    completions = sorted(
        (record.completed_at for record in imports if record.completed_at is not None), reverse=True
    )
    if uploads:
        result["latest_upload_at"] = uploads[0].isoformat()
    if completions:
        result["latest_import_completed_at"] = completions[0].isoformat()
    return result


def quality_summary(
    frame: pd.DataFrame,
    imports: list[ImportRecord] | None = None,
) -> dict[str, Any]:
    """Full, JSON-serializable data-quality summary for dashboards and ML."""
    imports = imports or []
    if frame is None or frame.empty:
        return {
            "schema_version": NORMALIZED_SCHEMA_VERSION,
            "records": 0,
            "providers": [],
            "currencies": [],
            "cost_bases": [],
            "records_by_provider": {},
            "rejected_records": sum(record.rejected_records for record in imports),
            "duplicate_rows_reported": sum(record.duplicate_records for record in imports),
            "coverage": {
                "series": 0,
                "calendar_days": 0,
                "days_with_records": 0,
                "coverage_ratio": 0.0,
            },
            "missing_periods": {"series_with_gaps": 0, "candidate_missing_days": 0, "examples": []},
            "freshness": freshness_report(frame, imports),
            "imports": len(imports),
        }

    coverage = coverage_by_series(frame)
    providers = sorted(frame["ServiceProviderName"].dropna().astype(str).unique().tolist())
    currencies = sorted(frame["BillingCurrency"].dropna().astype(str).unique().tolist())
    bases = sorted(frame["cost_basis"].dropna().astype(str).unique().tolist())
    by_provider = frame.groupby("ServiceProviderName").size().astype(int).to_dict()

    calendar_days = int(coverage["calendar_days"].sum()) if not coverage.empty else 0
    days_with_records = int(coverage["days_with_records"].sum()) if not coverage.empty else 0
    summary = {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "records": len(frame),
        "providers": providers,
        "currencies": currencies,
        "cost_bases": bases,
        "records_by_provider": by_provider,
        "rejected_records": int(sum(record.rejected_records for record in imports)),
        "duplicate_rows_reported": int(sum(record.duplicate_records for record in imports)),
        "coverage": {
            "series": len(coverage),
            "calendar_days": calendar_days,
            "days_with_records": days_with_records,
            "coverage_ratio": (days_with_records / calendar_days) if calendar_days else 0.0,
        },
        "missing_periods": missing_period_details(frame),
        "freshness": freshness_report(frame, imports),
        "imports": len(imports),
    }
    return summary


def import_warning_summary(imports: list[ImportRecord]) -> pd.DataFrame:
    """Flatten recorded import warnings/errors into a table for the UI."""
    rows: list[dict[str, Any]] = []
    for record in imports:
        for warning in record.warnings or []:
            rows.append(
                {
                    "import_id": record.import_id,
                    "ServiceProviderName": record.provider,
                    "filename": record.filename,
                    "severity": warning.get("severity", "warning"),
                    "code": warning.get("code"),
                    "message": warning.get("message"),
                    "count": warning.get("count", 0),
                }
            )
        if record.error_message:
            rows.append(
                {
                    "import_id": record.import_id,
                    "ServiceProviderName": record.provider,
                    "filename": record.filename,
                    "severity": "error",
                    "code": "import_failed",
                    "message": record.error_message,
                    "count": 1,
                }
            )
    return pd.DataFrame(rows)
