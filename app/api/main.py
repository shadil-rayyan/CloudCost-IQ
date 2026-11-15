"""FastAPI application factory and error handling."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import anomalies, costs, forecasts, health, imports, recommendations
from app.core.config import get_app_config
from app.core.exceptions import (
    CloudCostError,
    ModelContractError,
    ModelNotAvailableError,
)
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    config = get_app_config()
    application = FastAPI(
        title=config.app_name,
        version=config.app_version,
        description=(
            "Multi-cloud billing analytics and validated forecasting API. "
            "Predictions are only served from an approved production model artifact."
        ),
    )
    application.include_router(health.router, prefix="/api/v1", tags=["system"])
    application.include_router(imports.router, prefix="/api/v1", tags=["imports"])
    application.include_router(costs.router, prefix="/api/v1", tags=["costs"])
    application.include_router(forecasts.router, prefix="/api/v1", tags=["forecasts"])
    application.include_router(anomalies.router, prefix="/api/v1", tags=["anomalies"])
    application.include_router(recommendations.router, prefix="/api/v1", tags=["recommendations"])

    @application.exception_handler(ModelNotAvailableError)
    async def handle_model_unavailable(request: Request, exc: ModelNotAvailableError):
        return JSONResponse(status_code=409, content={"error": "model_unavailable", "detail": str(exc)})

    @application.exception_handler(ModelContractError)
    async def handle_model_contract(request: Request, exc: ModelContractError):
        return JSONResponse(status_code=409, content={"error": "model_contract", "detail": str(exc)})

    @application.exception_handler(CloudCostError)
    async def handle_cloudcost(request: Request, exc: CloudCostError):
        return JSONResponse(status_code=400, content={"error": type(exc).__name__, "detail": str(exc)})

    @application.exception_handler(ValueError)
    async def handle_value_error(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"error": "invalid_request", "detail": str(exc)})

    @application.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        logger.exception("Unhandled API error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": "An unexpected error occurred."},
        )

    return application


app = create_app()
