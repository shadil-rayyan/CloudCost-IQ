from __future__ import annotations

import pandas as pd

from app.ml.dataset_builder import DatasetBuildConfig, build_daily_panel
from app.ml.feature_engineering import build_supervised_dataset
from app.ml.time_series_split import build_rolling_origin_splits


def test_build_daily_panel_keeps_missing_days(sample_normalized_frame):
    frame = sample_normalized_frame.copy()
    # Drop one day of records to create a missing day in the panel.
    drop_date = pd.Timestamp("2025-02-15")
    frame = frame.loc[frame["usage_start"] != drop_date]
    result = build_daily_panel(frame, DatasetBuildConfig(currency="USD", cost_basis="billed"))
    assert not result.panel.empty
    missing = result.panel.loc[~result.panel["observed"]]
    assert (missing["cost"].isna()).all()
    assert result.quality["missing_periods"] > 0
    assert result.quality["coverage"]["calendar_days"] >= result.quality["coverage"]["observed_days"]


def test_supervised_features_are_leakage_safe(sample_normalized_frame):
    panel = build_daily_panel(sample_normalized_frame, DatasetBuildConfig()).panel
    supervised = build_supervised_dataset(panel)
    assert not supervised.frame.empty
    assert (supervised.frame["target_date"] > supervised.frame["origin_date"]).all()
    for column in supervised.schema.numeric_features:
        assert column in supervised.frame.columns
    assert supervised.frame["lag_1"].notna().any()


def test_rolling_origin_splits_are_chronological(sample_normalized_frame):
    panel = build_daily_panel(sample_normalized_frame, DatasetBuildConfig()).panel
    supervised = build_supervised_dataset(panel)
    # Sample fixture has ~220 days; use feasible params to verify chronology.
    plan = build_rolling_origin_splits(supervised.frame, horizon_days=30, validation_folds=2, min_train_days=60)
    assert plan.is_valid, plan.messages
    assert plan.final_test is not None
    for fold in plan.folds:
        assert fold.train_end < fold.validation_start
    assert plan.final_test.train_end < plan.final_test.validation_start
