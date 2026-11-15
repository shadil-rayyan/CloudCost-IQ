"""Validation of normalized billing records.

Validation never silently drops records: invalid rows are returned separately
with an explicit issue report so the user can download and inspect them. Some
conditions are warnings/informational by design (missing resource IDs, zero
cost, negative credits) and must not be treated as errors.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.core.constants import COST_BASES, CURRENCY_PATTERN, NORMALIZED_COLUMNS
from app.utils.dates import utc_now

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
MAX_SAMPLES = 5


@dataclass(slots=True)
class ValidationIssue:
    code: str
    severity: str
    message: str
    count: int
    column: str | None = None
    samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "count": int(self.count),
            "column": self.column,
            "samples": self.samples,
        }


@dataclass(slots=True)
class ValidationReport:
    records_read: int = 0
    valid_records: int = 0
    rejected_records: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    valid_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    rejected_frame: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def error_issues(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warning_issues(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def info_issues(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "info"]

    @property
    def error_count(self) -> int:
        """Number of *records* that failed error-level checks."""
        return self.rejected_records

    @property
    def warning_count(self) -> int:
        return sum(issue.count for issue in self.warning_issues)

    @property
    def is_importable(self) -> bool:
        return self.valid_records > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_read": int(self.records_read),
            "valid_records": int(self.valid_records),
            "rejected_records": int(self.rejected_records),
            "error_count": int(self.error_count),
            "warning_count": int(self.warning_count),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _samples(frame: pd.DataFrame, mask: pd.Series, column: str | None = None) -> list[str]:
    subset = frame.loc[mask]
    if subset.empty:
        return []
    if column and column in subset.columns:
        values = subset[column].head(MAX_SAMPLES).astype("string").fillna("<null>")
        return [str(value) for value in values]
    return [f"row_index={index}" for index in subset.index[:MAX_SAMPLES]]


def validate_batch(
    frame: pd.DataFrame,
    *,
    allow_zero_cost: bool = True,
    now: dt.datetime | None = None,
) -> ValidationReport:
    """Run record-level validation against the normalized schema."""
    report = ValidationReport()
    if frame is None or frame.empty:
        report.issues.append(
            ValidationIssue(
                code="empty_file",
                severity="error",
                message="The file contains no billing records.",
                count=1,
            )
        )
        return report

    working = frame.copy()
    for column in NORMALIZED_COLUMNS:
        if column not in working.columns:
            working[column] = None

    report.records_read = len(working)
    now = now or utc_now()
    issue_rows: list[ValidationIssue] = []

    def add_issue(
        code: str,
        severity: str,
        message: str,
        mask: pd.Series,
        column: str | None = None,
    ) -> None:
        count = int(mask.fillna(False).sum())
        if count <= 0:
            return
        issue_rows.append(
            ValidationIssue(
                code=code,
                severity=severity,
                message=message,
                count=count,
                column=column,
                samples=_samples(working, mask.fillna(False), column),
            )
        )

    error_mask = pd.Series(False, index=working.index)

    def add_error(*args, **kwargs) -> None:
        nonlocal error_mask
        mask = args[3] if len(args) > 3 else kwargs.get("mask")
        if mask is not None:
            error_mask = error_mask | mask.fillna(False).astype(bool)
        add_issue(*args, **kwargs)

    usage_start = pd.to_datetime(working["usage_start"], errors="coerce")
    usage_end = pd.to_datetime(working["usage_end"], errors="coerce")
    charge_start = pd.to_datetime(working["ChargePeriodStart"], errors="coerce")
    charge_end = pd.to_datetime(working["ChargePeriodEnd"], errors="coerce")
    cost = pd.to_numeric(working["BilledCost"], errors="coerce")

    add_error(
        "missing_usage_start",
        "error",
        "Records are missing a usage start timestamp.",
        usage_start.isna(),
        "usage_start",
    )
    add_error(
        "missing_cost",
        "error",
        "Records are missing a numeric cost amount.",
        cost.isna(),
        "BilledCost",
    )
    add_error(
        "missing_currency",
        "error",
        "Records are missing a billing currency.",
        working["BillingCurrency"].isna(),
        "BillingCurrency",
    )
    currency = working["BillingCurrency"].astype("string")
    invalid_currency = currency.notna() & ~currency.str.match(CURRENCY_PATTERN, na=False)
    add_error(
        "invalid_currency",
        "error",
        "Records use a currency code that is not a 3-letter ISO code.",
        invalid_currency,
        "BillingCurrency",
    )
    add_error(
        "unsupported_cost_basis",
        "error",
        "Records use an unsupported cost basis.",
        ~working["cost_basis"].isin(COST_BASES),
        "cost_basis",
    )
    add_error(
        "missing_provider",
        "error",
        "Records are missing a provider.",
        working["ServiceProviderName"].isna(),
        "ServiceProviderName",
    )
    add_error(
        "usage_end_before_start",
        "error",
        "Usage end precedes usage start.",
        usage_end.notna() & usage_start.notna() & (usage_end < usage_start),
        "usage_end",
    )
    add_error(
        "charge_period_end_before_start",
        "error",
        "Charge period end precedes charge period start.",
        charge_end.notna() & charge_start.notna() & (charge_end < charge_start),
        "ChargePeriodEnd",
    )

    add_issue(
        "future_usage_timestamp",
        "warning",
        "Usage timestamps are in the future (billing delay or timezone issue).",
        usage_start.notna() & (usage_start > pd.Timestamp(now + dt.timedelta(days=2))),
        "usage_start",
    )
    add_issue(
        "missing_charge_period",
        "warning",
        "Records have no charge period start; usage timestamps will be used for daily aggregation.",
        charge_start.isna(),
        "ChargePeriodStart",
    )
    add_issue(
        "missing_service_name",
        "warning",
        "Records have no service name and cannot be fully attributed.",
        working["ServiceName"].isna(),
        "ServiceName",
    )
    add_issue(
        "missing_invoice_month",
        "warning",
        "Records have no invoice month; it will be derived from the charge period.",
        working["invoice_month"].isna(),
        "invoice_month",
    )
    add_issue(
        "implausible_usage_date",
        "warning",
        "Usage timestamps are implausibly old (before 2000).",
        usage_start.notna() & (usage_start < pd.Timestamp("2000-01-01")),
        "usage_start",
    )
    add_issue(
        "negative_quantity",
        "warning",
        "Records contain negative usage quantities (refunds or corrections).",
        pd.to_numeric(working["ConsumedQuantity"], errors="coerce").fillna(0) < 0,
        "ConsumedQuantity",
    )
    if allow_zero_cost:
        add_issue(
            "zero_cost_records",
            "info",
            "Records have a zero cost (free tier, credits or zero-rated usage).",
            cost.fillna(1) == 0,
            "BilledCost",
        )
    add_issue(
        "negative_cost_records",
        "info",
        "Records have negative cost (credits, refunds or adjustments); they are preserved.",
        cost.fillna(0) < 0,
        "BilledCost",
    )
    add_issue(
        "missing_resource_id",
        "info",
        "Records have no resource ID. This is normal for aggregate billing lines.",
        working["ResourceId"].isna(),
        "ResourceId",
    )
    add_issue(
        "missing_region",
        "info",
        "Records have no region.",
        working["RegionName"].isna(),
        "RegionName",
    )

    # Exact duplicate detection is informational: identical-looking rows can be
    # legitimate separate billing lines, so they are never dropped automatically.
    comparable = [c for c in NORMALIZED_COLUMNS if c not in {"record_id", "raw_fields_json"}]
    duplicate_mask = working.duplicated(subset=comparable, keep="first")
    add_issue(
        "exact_duplicate_rows",
        "warning",
        "Rows are identical across all normalized columns. Review before relying on totals; "
        "CloudCost IQ does not delete them automatically.",
        duplicate_mask,
        None,
    )

    valid_mask = ~error_mask.fillna(False)
    report.valid_frame = working.loc[valid_mask].reset_index(drop=True)
    report.rejected_frame = working.loc[~valid_mask].reset_index(drop=True)
    report.valid_records = len(report.valid_frame)
    report.rejected_records = len(report.rejected_frame)
    report.issues = sorted(
        issue_rows, key=lambda issue: (SEVERITY_ORDER.get(issue.severity, 9), issue.code)
    )
    return report
