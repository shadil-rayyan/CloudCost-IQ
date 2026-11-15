"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.analytics.store import CostStore
from app.api.schemas import HealthResponse
from app.core.config import get_app_config
from app.core.constants import NORMALIZED_SCHEMA_VERSION
from app.database.repositories import ImportRepository
from app.ml import predict as forecast_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    config = get_app_config()
    try:
        ImportRepository(config.storage.database_path).list(limit=1)
        database_ok = True
    except Exception:
        database_ok = False
    store = CostStore()
    availability = forecast_service.ForecastService().availability()
    return HealthResponse(
        status="ok" if database_ok else "degraded",
        version=config.app_version,
        database=database_ok,
        data_available=store.has_data(),
        forecast_available=availability.available,
        normalized_schema_version=NORMALIZED_SCHEMA_VERSION,
    )
