"""Forecast endpoints. Predictions are only produced by a validated model."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.deps import records
from app.api.schemas import ForecastRequestSchema, ForecastResponse
from app.database.repositories import ForecastRepository
from app.ml.predict import ForecastRequest, ForecastService

router = APIRouter()


@router.post("/forecasts", response_model=ForecastResponse, status_code=201)
def create_forecast(request: ForecastRequestSchema) -> ForecastResponse:
    service = ForecastService()
    availability = service.availability()
    if not availability.available:
        raise HTTPException(status_code=409, detail=availability.reason)
    result = service.generate(
        ForecastRequest(
            provider=request.provider,
            billing_account_id=request.billing_account_id,
            service_name=request.service_name,
            currency=request.currency,
            cost_basis=request.cost_basis,
            horizon_days=request.horizon_days,
            origin_date=request.origin_date,
        ),
        persist=request.persist,
    )
    return ForecastResponse(
        forecast_id=result.forecast_id,
        model_version=result.model_version,
        model_stage=result.model_stage,
        generated_at=result.generated_at.isoformat(),
        origin_date=result.origin_date.isoformat() if result.origin_date else None,
        horizon_days=result.horizon_days,
        currency=result.currency,
        cost_basis=result.cost_basis,
        summary=result.summary,
        warnings=result.warnings,
        rows=records(result.rows),
    )


@router.get("/forecasts/{forecast_id}")
def get_forecast(forecast_id: str) -> dict:
    record = ForecastRepository().get(forecast_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Forecast not found")
    return {
        "forecast_id": record.forecast_id,
        "model_version": record.model_version,
        "provider": record.provider,
        "billing_account_id": record.billing_account_id,
        "service_name": record.service_name,
        "currency": record.currency,
        "cost_basis": record.cost_basis,
        "horizon_days": record.horizon_days,
        "generated_at": record.generated_at.isoformat(),
        "origin_date": record.as_of_date.isoformat() if record.as_of_date else None,
        "status": str(record.status),
        "notes": record.notes,
        "metrics": record.metrics,
    }
