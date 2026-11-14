"""Anomaly detection endpoints (transparent statistical detection)."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Query

from app.analytics.anomaly_detection import anomaly_summary, detect_statistical_anomalies
from app.api.deps import cost_filter, load_costs, records
from app.api.schemas import RecordListResponse

router = APIRouter()


@router.get("/anomalies", response_model=RecordListResponse)
def list_anomalies(
    providers: list[str] | None = Query(default=None),
    currencies: list[str] | None = Query(default=None),
    cost_basis: str | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    threshold: float = Query(default=3.0, gt=0.5, le=10.0),
    lookback_days: int = Query(default=28, ge=7, le=120),
) -> RecordListResponse:
    filters = cost_filter(
        providers=providers,
        currencies=currencies,
        cost_basis=cost_basis,
        start_date=start_date,
        end_date=end_date,
    )
    _, frame = load_costs(filters)
    anomalies = detect_statistical_anomalies(
        frame, lookback_days=lookback_days, threshold=threshold
    )
    items = records(anomalies)
    return RecordListResponse(rows=len(items), records=items)


@router.get("/anomalies/summary")
def summary(
    providers: list[str] | None = Query(default=None),
    threshold: float = Query(default=3.0, gt=0.5, le=10.0),
) -> dict:
    _, frame = load_costs(cost_filter(providers=providers))
    anomalies = detect_statistical_anomalies(frame, threshold=threshold)
    return anomaly_summary(anomalies)
