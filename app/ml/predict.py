"""Forecast inference service.

The service refuses to serve anything except a validated production bundle.
Inference is direct (one feature row per horizon), so predictions are never
fed back as lag values. Quantile ordering is checked and corrected per row and
the correction rate is recorded with every forecast.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.analytics.store import CostFilter, CostStore
from app.core.config import get_app_config, get_ml_config
from app.core.exceptions import (
    CostBasisMismatchError,
    CurrencyMismatchError,
    InsufficientHistoryError,
    ModelContractError,
    ModelNotAvailableError,
)
from app.database.models import ForecastRecord, ForecastStatus
from app.database.repositories import ForecastRepository
from app.ml.artifact_manager import ArtifactManager, ModelBundle
from app.ml.dataset_builder import DatasetBuildConfig, build_daily_panel
from app.ml.evaluate import enforce_monotonic
from app.ml.feature_engineering import build_inference_features
from app.ml.model_contract import validate_forecast_frame
from app.utils.dates import utc_now
from app.utils.files import write_parquet_atomic

UNAVAILABLE_MESSAGE = (
    "Forecasting is not yet available. A validated production model has not been registered."
)


@dataclass(slots=True)
class ForecastRequest:
    provider: str | None = None
    billing_account_id: str | None = None
    service_name: str | None = None
    currency: str | None = None
    cost_basis: str | None = None
    horizon_days: int | None = None
    origin_date: dt.date | None = None

    def scope_label(self) -> str:
        parts = [
            self.provider or "All providers",
            self.billing_account_id or "all accounts",
            self.service_name or "all services",
        ]
        return " / ".join(parts)


@dataclass(slots=True)
class ForecastAvailability:
    available: bool
    reason: str
    version: str | None = None
    stage: str | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    model_card: str = ""
    validated: bool = False


@dataclass(slots=True)
class ForecastResult:
    forecast_id: str
    model_version: str
    model_stage: str
    generated_at: dt.datetime
    origin_date: dt.date | None
    horizon_days: int
    currency: str
    cost_basis: str
    rows: pd.DataFrame
    history: pd.DataFrame
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    results_path: str | None = None
    model_config: dict[str, Any] = field(default_factory=dict)

    def totals(self) -> dict[str, float]:
        return self.summary.get("totals", {})


class ForecastService:
    """Load a production bundle and generate interval forecasts."""

    def __init__(
        self,
        *,
        processed_dir: Path | str | None = None,
        artifacts_dir: Path | str | None = None,
        database_path: Path | str | None = None,
        store: CostStore | None = None,
        artifacts: ArtifactManager | None = None,
        forecasts: ForecastRepository | None = None,
    ) -> None:
        self.store = store or CostStore(processed_dir=processed_dir)
        self.artifacts = artifacts or ArtifactManager(artifacts_dir=artifacts_dir)
        self.forecasts = forecasts or ForecastRepository(database_path=database_path)

    # ------------------------------------------------------------------ status
    def availability(self) -> ForecastAvailability:
        try:
            bundle = self.artifacts.production_bundle()
        except (ModelNotAvailableError, ModelContractError) as exc:
            return ForecastAvailability(available=False, reason=f"{UNAVAILABLE_MESSAGE} ({exc})")
        if bundle is None:
            return ForecastAvailability(available=False, reason=UNAVAILABLE_MESSAGE)
        return ForecastAvailability(
            available=True,
            reason="A validated production model is available.",
            version=bundle.version,
            stage=bundle.stage,
            model_config=bundle.model_config.model_dump(),
            manifest=bundle.manifest,
            metrics=bundle.metrics,
            model_card=bundle.model_card,
            validated=True,
        )

    def model_options(self) -> list[dict[str, Any]]:
        return [summary.to_dict() for summary in self.artifacts.list_models()]

    # ----------------------------------------------------------------- generate
    def generate(
        self,
        request: ForecastRequest | None = None,
        *,
        persist: bool = True,
    ) -> ForecastResult:
        request = request or ForecastRequest()
        bundle = self.artifacts.production_bundle()
        if bundle is None:
            raise ModelNotAvailableError(UNAVAILABLE_MESSAGE)

        currency = (request.currency or bundle.model_config.currency).upper()
        cost_basis = request.cost_basis or bundle.model_config.cost_basis
        if currency != bundle.model_config.currency.upper():
            raise CurrencyMismatchError(
                f"The production model was trained for {bundle.model_config.currency}; "
                f"a {currency} forecast cannot be produced with it."
            )
        if cost_basis != bundle.model_config.cost_basis:
            raise CostBasisMismatchError(
                f"The production model was trained on the '{bundle.model_config.cost_basis}' "
                f"cost basis; '{cost_basis}' cannot be used."
            )

        horizon = int(request.horizon_days or bundle.model_config.forecast_horizon_days)
        if horizon <= 0:
            raise ValueError("Forecast horizon must be a positive number of days.")
        if horizon > bundle.model_config.forecast_horizon_days:
            raise InsufficientHistoryError(
                f"The production model supports at most {bundle.model_config.forecast_horizon_days} "
                f"forecast days; {horizon} were requested."
            )

        frame = self.store.load(
            CostFilter(
                providers=(request.provider,) if request.provider else (),
                billing_accounts=(request.billing_account_id,) if request.billing_account_id else (),
                services=(request.service_name,) if request.service_name else (),
                currencies=(currency,),
                cost_basis=cost_basis,
            ),
            columns=[
                "ServiceProviderName",
                "BillingAccountId",
                "ServiceName",
                "BillingCurrency",
                "cost_basis",
                "usage_start",
                "ChargePeriodStart",
                "BilledCost",
            ],
        )
        if frame.empty:
            raise InsufficientHistoryError(
                "No normalized billing records match the selected forecast scope."
            )

        panel_config = DatasetBuildConfig(
            currency=currency,
            cost_basis=cost_basis,
            horizon_days=horizon,
            min_history_days=max(28 + horizon, 60),
            min_observations=max(28, horizon),
            min_coverage_ratio=0.2,
            validation_folds=0,
            min_train_days=0,
        )
        built = build_daily_panel(frame, panel_config)
        if built.panel.empty or not built.eligible_series:
            raise InsufficientHistoryError(
                "Insufficient history for this scope. "
                + (built.insufficient_reason() or "At least 60 calendar days are required.")
            )

        panel = built.eligible_panel
        schema = bundle.feature_schema
        feature_frame = build_inference_features(
            panel,
            schema=schema,
            origin_date=request.origin_date,
            horizon_days=horizon,
        )
        if feature_frame.empty:
            raise InsufficientHistoryError(
                "The selected scope has no series with enough history to build forecast features."
            )
        validate_forecast_frame(feature_frame, schema)

        raw = np.asarray(bundle.model.predict(feature_frame.loc[:, list(schema.all_features)]), dtype="float64")
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        if raw.shape[1] != len(bundle.model_config.quantile_alpha):
            raise ModelContractError(
                f"The model returned {raw.shape[1]} outputs but the bundle declares "
                f"{len(bundle.model_config.quantile_alpha)} quantiles."
            )
        if not np.isfinite(raw).all():
            raise ModelContractError("The model returned non-finite forecast values.")

        rows = feature_frame.loc[:, [
            "series_id",
            "ServiceProviderName",
            "BillingAccountId",
            "ServiceName",
            "BillingCurrency",
            "cost_basis",
            "origin_date",
            "target_date",
            "horizon",
        ]].copy()
        rows["p10"] = raw[:, 0]
        rows["p50"] = raw[:, 1]
        rows["p90"] = raw[:, 2]
        rows, crossings = enforce_monotonic(rows)
        crossing_rate = float(crossings / len(rows)) if len(rows) else 0.0
        warnings: list[str] = []
        threshold = float(get_ml_config().get("evaluation", {}).get("max_quantile_crossing_rate", 0.02))
        if crossing_rate > threshold:
            warnings.append(
                f"Quantile crossing occurred for {crossings} of {len(rows)} rows "
                f"({crossing_rate:.1%}); intervals were corrected to P10 <= P50 <= P90."
            )

        origin = rows["origin_date"].min()
        origin_date = pd.Timestamp(origin).date() if pd.notna(origin) else None
        history = (
            built.panel.loc[built.panel["observed"]]
            .groupby("date", as_index=False)["cost"]
            .sum()
            .sort_values("date")
        )
        history = history.rename(columns={"date": "date", "cost": "cost"}).reset_index(drop=True)

        summary = _summarise(rows, history)
        summary["quantile_crossing"] = {
            "rows": int(crossings),
            "rate": crossing_rate,
            "corrected": True,
        }
        forecast_id = "fc-" + uuid.uuid4().hex[:16]
        generated_at = utc_now()

        result = ForecastResult(
            forecast_id=forecast_id,
            model_version=bundle.version,
            model_stage=bundle.stage,
            generated_at=generated_at,
            origin_date=origin_date,
            horizon_days=horizon,
            currency=currency,
            cost_basis=cost_basis,
            rows=rows,
            history=history,
            summary=summary,
            warnings=warnings,
            model_config=bundle.model_config.model_dump(),
        )

        if persist:
            results_path = self._persist(result, bundle, request)
            result.results_path = str(results_path)
            self._save_metadata(result, bundle, request)
        return result

    # --------------------------------------------------------------- persistence
    def _persist(self, result: ForecastResult, bundle: ModelBundle, request: ForecastRequest) -> Path:
        metadata_dir = Path(get_app_config().storage.metadata_dir) / "forecasts"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        path = metadata_dir / f"{result.forecast_id}.parquet"
        write_parquet_atomic(result.rows, path)
        return path

    def _save_metadata(self, result: ForecastResult, bundle: ModelBundle, request: ForecastRequest) -> None:
        history_start = None
        history_end = None
        if not result.history.empty:
            history_start = pd.Timestamp(result.history["date"].min()).date()
            history_end = pd.Timestamp(result.history["date"].max()).date()
        record = ForecastRecord(
            forecast_id=result.forecast_id,
            model_version=bundle.version,
            currency=result.currency,
            cost_basis=result.cost_basis,
            horizon_days=result.horizon_days,
            generated_at=result.generated_at,
            as_of_date=result.origin_date,
            provider=request.provider,
            billing_account_id=request.billing_account_id,
            service_name=request.service_name,
            history_start=history_start,
            history_end=history_end,
            status=ForecastStatus.COMPLETED,
            results_path=result.results_path,
            notes=" ".join(result.warnings) or None,
            metrics={
                "totals": result.summary.get("totals", {}),
                "series": result.summary.get("series_count", 0),
                "quantile_crossing_rate": result.summary.get("quantile_crossing", {}).get("rate", 0.0),
                "model_version": bundle.version,
                "scope": request.scope_label(),
            },
        )
        self.forecasts.save(record)


def _summarise(rows: pd.DataFrame, history: pd.DataFrame) -> dict[str, Any]:
    totals = {
        "p10": float(rows["p10"].sum()),
        "p50": float(rows["p50"].sum()),
        "p90": float(rows["p90"].sum()),
    }

    def group(column: str) -> list[dict[str, Any]]:
        if column not in rows.columns:
            return []
        grouped = (
            rows.groupby(column, dropna=False)[["p10", "p50", "p90"]]
            .sum()
            .reset_index()
            .sort_values("p50", ascending=False)
        )
        return [
            {
                column: None if pd.isna(row[column]) else str(row[column]),
                "p10": float(row["p10"]),
                "p50": float(row["p50"]),
                "p90": float(row["p90"]),
            }
            for _, row in grouped.iterrows()
        ]

    return {
        "totals": totals,
        "series_count": int(rows["series_id"].nunique()),
        "rows": len(rows),
        "history_days": len(history),
        "by_provider": group("ServiceProviderName"),
        "by_service": group("ServiceName"),
        "by_account": group("BillingAccountId"),
        "horizon_start": pd.Timestamp(rows["target_date"].min()).date().isoformat()
        if len(rows)
        else None,
        "horizon_end": pd.Timestamp(rows["target_date"].max()).date().isoformat() if len(rows) else None,
    }
