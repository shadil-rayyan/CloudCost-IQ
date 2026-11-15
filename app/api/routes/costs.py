"""Cost summary, time-series and breakdown endpoints."""

from __future__ import annotations

import datetime as dt

import pandas as pd
from fastapi import APIRouter, Query

from app.analytics.cost_breakdown import breakdown
from app.analytics.cost_summary import summarize_costs
from app.analytics.cost_trends import cost_trend
from app.api.deps import cost_filter, load_costs, records
from app.api.schemas import CostSummaryResponse, RecordListResponse

router = APIRouter()


def _filter(
    providers: list[str] | None,
    currencies: list[str] | None,
    cost_basis: str | None,
    start_date: dt.date | None,
    end_date: dt.date | None,
) -> tuple[pd.DataFrame, object]:
    filters = cost_filter(
        providers=providers,
        currencies=currencies,
        cost_basis=cost_basis,
        start_date=start_date,
        end_date=end_date,
    )
    _, frame = load_costs(filters)
    return frame, filters


@router.get("/costs/summary", response_model=CostSummaryResponse)
def cost_summary(
    providers: list[str] | None = Query(default=None),
    currencies: list[str] | None = Query(default=None),
    cost_basis: str | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> CostSummaryResponse:
    frame, _ = _filter(providers, currencies, cost_basis, start_date, end_date)
    return CostSummaryResponse(**summarize_costs(frame))


@router.get("/costs/timeseries", response_model=RecordListResponse)
def cost_timeseries(
    providers: list[str] | None = Query(default=None),
    currencies: list[str] | None = Query(default=None),
    cost_basis: str | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    granularity: str = Query(default="D", pattern="^(D|W|M)$"),
    group_by: list[str] | None = Query(default=None),
) -> RecordListResponse:
    frame, _ = _filter(providers, currencies, cost_basis, start_date, end_date)
    keys = tuple(group_by or ("ServiceProviderName",))
    trend = cost_trend(frame, freq=granularity, group_by=keys)
    items = records(trend)
    return RecordListResponse(rows=len(items), records=items)


@router.get("/costs/breakdown", response_model=RecordListResponse)
def cost_breakdown(
    dimension: str = "service",
    top_n: int = Query(default=15, ge=1, le=100),
    providers: list[str] | None = Query(default=None),
    currencies: list[str] | None = Query(default=None),
    cost_basis: str | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> RecordListResponse:
    frame, _ = _filter(providers, currencies, cost_basis, start_date, end_date)
    result = breakdown(frame, dimension, top_n=top_n)
    items = records(result)
    return RecordListResponse(rows=len(items), records=items)
