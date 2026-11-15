"""Reusable training pipeline helpers.

The scripts and the Google Colab notebook both call these functions, so the
training data preparation steps are identical everywhere. They are thin
coordination helpers: the actual logic lives in the dataset, feature, split and
training modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.analytics.store import CostFilter, CostStore
from app.core.config import get_ml_config
from app.ml.dataset_builder import (
    DatasetBuildConfig,
    DatasetBuildResult,
    build_daily_panel,
)
from app.ml.feature_engineering import (
    FeatureBuildConfig,
    FeatureSchema,
    SupervisedBuildResult,
    build_supervised_dataset,
)
from app.ml.time_series_split import SplitPlan, build_rolling_origin_splits
from app.utils.files import write_parquet_atomic


def load_normalized_records(
    records_path: Path | str | None = None,
    *,
    filters: CostFilter | None = None,
) -> pd.DataFrame:
    """Load normalized billing records from Parquet or the processed store."""
    if records_path is None:
        return CostStore().load(filters)
    path = Path(records_path)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    if not path.exists():
        raise FileNotFoundError(f"Normalized records path does not exist: {path}")
    return pd.read_parquet(path)


def prepare_training_panel(
    records: pd.DataFrame,
    *,
    currency: str | None = None,
    cost_basis: str | None = None,
    config: DatasetBuildConfig | None = None,
) -> DatasetBuildResult:
    """Build the daily multi-series panel and its eligibility report."""
    config = config or DatasetBuildConfig.from_ml_config(
        currency=currency, cost_basis=cost_basis
    )
    return build_daily_panel(records, config)


def build_supervised(
    panel: pd.DataFrame,
    *,
    horizon_days: int | None = None,
    schema: FeatureSchema | dict[str, Any] | None = None,
    include_account: bool = True,
) -> SupervisedBuildResult:
    """Engineer leakage-safe direct multi-horizon training rows."""
    config = FeatureBuildConfig(
        horizon_days=int(horizon_days or get_ml_config().get("model", {}).get("forecast_horizon_days", 30)),
        include_account=include_account,
    )
    return build_supervised_dataset(panel, config=config, schema=schema)


def make_splits(
    supervised: SupervisedBuildResult,
    *,
    horizon_days: int | None = None,
    validation_folds: int | None = None,
    min_train_days: int | None = None,
) -> SplitPlan:
    """Create chronological rolling-origin folds and an untouched test window."""
    training = get_ml_config().get("training", {})
    return build_rolling_origin_splits(
        supervised.frame,
        horizon_days=int(
            horizon_days or get_ml_config().get("model", {}).get("forecast_horizon_days", 30)
        ),
        validation_folds=int(validation_folds or training.get("validation_folds", 3)),
        min_train_days=int(min_train_days or training.get("min_train_days", 90)),
    )


def save_panel(
    result: DatasetBuildResult,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Persist the panel and quality report for later training runs."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    panel_path = destination / "daily_panel.parquet"
    quality_path = destination / "dataset_quality.json"
    write_parquet_atomic(result.panel, panel_path)
    quality_path.write_text(
        json.dumps(result.quality, indent=2, default=str), encoding="utf-8"
    )
    return {"panel": panel_path, "quality": quality_path}


def load_panel(panel_path: Path | str) -> pd.DataFrame:
    path = Path(panel_path)
    if not path.exists():
        raise FileNotFoundError(f"Panel file does not exist: {path}")
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    if "observed" not in panel.columns:
        panel["observed"] = panel["cost"].notna()
    return panel
