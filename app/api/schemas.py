"""Pydantic request/response schemas for the CloudCost IQ API."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    database: bool
    data_available: bool
    forecast_available: bool
    normalized_schema_version: str


class ImportRequest(BaseModel):
    provider: Literal["Auto", "AWS", "Azure", "GCP", "FOCUS"] = "Auto"
    cost_basis: str = "billed"
    dataset_kind: Literal["real", "demo", "test"] = "real"
    replace_overlapping: bool = False


class ImportResponse(BaseModel):
    import_id: str
    status: str
    provider: str
    filename: str
    records_read: int
    valid_records: int
    rejected_records: int
    duplicate_records: int
    currency: str | None = None
    cost_basis: str | None = None
    coverage_start: str | None = None
    coverage_end: str | None = None
    schema_name: str | None = None
    schema_version: str | None = None
    warning_count: int = 0
    error_message: str | None = None
    processing_seconds: float = 0.0


class ImportDetail(ImportResponse):
    file_format: str | None = None
    dataset_kind: str = "real"
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    uploaded_at: str | None = None
    completed_at: str | None = None


class CostSummaryResponse(BaseModel):
    summaries: list[dict[str, Any]]
    records: int
    currencies: list[str]
    cost_bases: list[str]
    combined_total: float | None
    is_multi_currency: bool
    is_multi_basis: bool
    mixed_totals_blocked: bool


class RecordListResponse(BaseModel):
    rows: int
    records: list[dict[str, Any]]


class ForecastRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str | None = None
    billing_account_id: str | None = None
    service_name: str | None = None
    currency: str | None = None
    cost_basis: str | None = None
    horizon_days: int | None = Field(default=None, ge=1, le=365)
    origin_date: dt.date | None = None
    persist: bool = True


class ForecastResponse(BaseModel):
    forecast_id: str
    model_version: str
    model_stage: str
    generated_at: str
    origin_date: str | None
    horizon_days: int
    currency: str
    cost_basis: str
    summary: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)


class RecommendationPatch(BaseModel):
    status: Literal["new", "under_review", "accepted", "rejected", "completed"]
