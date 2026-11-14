from __future__ import annotations

from app.ml.artifact_manager import ArtifactManager
from app.ml.dataset_builder import DatasetBuildConfig, build_daily_panel
from app.ml.feature_engineering import build_supervised_dataset
from app.ml.model_contract import validate_bundle
from app.ml.pipeline import make_splits
from app.ml.predict import ForecastRequest, ForecastService
from app.ml.train import TrainingConfig, train_forecast_model


def _fast_config() -> TrainingConfig:
    return TrainingConfig(
        quantiles=(0.1, 0.5, 0.9),
        horizon_days=30,
        random_state=42,
        hyperparameters={
            "max_depth": 3,
            "learning_rate": 0.2,
            "n_estimators": 30,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "early_stopping_rounds": 10,
        },
        gates={
            "max_wape": 10.0,
            "max_pinball_loss_ratio_vs_baseline": 10.0,
            "min_interval_coverage": 0.0,
            "max_interval_coverage": 1.0,
            "max_quantile_crossing_rate": 1.0,
            "require_baseline_comparison": True,
        },
        mlflow_enabled=False,
    )


def test_training_produces_valid_bundle_and_prediction(sample_normalized_frame, isolated_workspace):
    from app.analytics.store import CostStore
    from app.processing.storage import ProcessedStore

    # ForecastService reads from processed storage; persist the synthetic test records.
    ProcessedStore().write(sample_normalized_frame, "test-import")

    panel = build_daily_panel(sample_normalized_frame, DatasetBuildConfig()).panel
    supervised = build_supervised_dataset(panel)
    splits = make_splits(supervised, horizon_days=30, validation_folds=2, min_train_days=60)
    assert splits.is_valid

    artifacts = ArtifactManager()
    result = train_forecast_model(
        supervised,
        splits,
        dataset_quality={"dataset_fingerprint": "test-fp", "history_start": "2025-01-01", "history_end": "2025-08-08", "history_days": 220, "series_count": 3, "eligible_series_count": 3, "sufficient": True},
        config=_fast_config(),
        artifacts=artifacts,
        save_bundle=True,
        track_mlflow=False,
    )
    assert result.bundle_path is not None
    validation = validate_bundle(result.bundle_path)
    assert validation.passed, validation.to_dict()
    assert "final_test" in result.metrics
    assert result.metrics["final_test"]["mae"] is not None

    artifacts.promote(result.version, from_stage="candidates", to_stage="production")
    service = ForecastService(store=CostStore(), artifacts=artifacts)
    availability = service.availability()
    assert availability.available
    forecast = service.generate(ForecastRequest(currency="USD", cost_basis="billed", horizon_days=30))
    assert len(forecast.rows) > 0
    assert (forecast.rows["p10"] <= forecast.rows["p50"]).all()
    assert (forecast.rows["p50"] <= forecast.rows["p90"]).all()
