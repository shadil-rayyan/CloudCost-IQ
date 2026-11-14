"""Dataclasses and enums describing stored metadata records."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ImportStatus(StrEnum):
    PENDING = "pending"
    VALIDATED = "validated"
    COMPLETED = "completed"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class DatasetKind(StrEnum):
    REAL = "real"
    DEMO = "demo"
    TEST = "test"


class RecommendationStatus(StrEnum):
    NEW = "new"
    UNDER_REVIEW = "under_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMPLETED = "completed"


class ForecastStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class ImportRecord:
    import_id: str
    provider: str
    filename: str
    status: str
    stored_filename: str | None = None
    file_size_bytes: int | None = None
    file_fingerprint: str | None = None
    file_format: str | None = None
    schema_name: str | None = None
    schema_version: str | None = None
    dataset_kind: str = DatasetKind.REAL
    cost_basis: str | None = None
    currency: str | None = None
    date_coverage_start: dt.date | None = None
    date_coverage_end: dt.date | None = None
    invoice_month_min: str | None = None
    invoice_month_max: str | None = None
    records_read: int = 0
    valid_records: int = 0
    rejected_records: int = 0
    duplicate_records: int = 0
    warning_count: int = 0
    error_count: int = 0
    warnings: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None
    import_config: dict[str, Any] = field(default_factory=dict)
    uploaded_at: dt.datetime | None = None
    completed_at: dt.datetime | None = None
    processing_seconds: float | None = None
    replaced_import_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ForecastRecord:
    forecast_id: str
    model_version: str
    currency: str
    cost_basis: str
    horizon_days: int
    generated_at: dt.datetime
    as_of_date: dt.date | None
    status: str = ForecastStatus.COMPLETED
    provider: str | None = None
    billing_account_id: str | None = None
    service_name: str | None = None
    history_start: dt.date | None = None
    history_end: dt.date | None = None
    results_path: str | None = None
    notes: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RecommendationRecord:
    recommendation_id: str
    rule_id: str
    title: str
    status: str = RecommendationStatus.NEW
    provider: str | None = None
    account: str | None = None
    service: str | None = None
    observation: str | None = None
    evidence: str | None = None
    suggested_action: str | None = None
    risk: str | None = None
    estimated_savings: float | None = None
    currency: str | None = None
    priority: str = "medium"
    period_start: dt.date | None = None
    period_end: dt.date | None = None
    created_at: dt.datetime | None = None
    updated_at: dt.datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
