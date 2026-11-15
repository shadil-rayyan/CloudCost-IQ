"""Leakage-safe feature engineering for the global multi-horizon forecast model.

Training rows are built with a *direct* multi-horizon design: for a forecast
origin ``t`` and horizon ``h`` in ``1..H`` the label is the cost observed on
``t + h``. Historical features only ever look at data up to and including
``t``; calendar features describe the target date ``t + h`` -- which is known
at forecast time and is therefore not leakage.

Missing days are handled explicitly. A lag on a missing day is ``NaN`` (never
zero), rolling statistics are computed over observed values with a documented
minimum number of observations, and ``days_since_last_observed`` /
``observed_ratio_28`` expose data gaps to the model.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from app.core.constants import classify_service
from app.ml.model_contract import FeatureSchema

FEATURE_SCHEMA_VERSION = "1.0"

#: Numeric features, in the exact order used by training and inference.
NUMERIC_FEATURES: tuple[str, ...] = (
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_median_7",
    "rolling_std_7",
    "rolling_sum_7",
    "rolling_mean_28",
    "rolling_median_28",
    "rolling_std_28",
    "rolling_sum_28",
    "ewm_14",
    "trend_short",
    "trend_long",
    "days_since_last_observed",
    "observed_ratio_28",
    "horizon",
    "day_of_week",
    "is_weekend",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "day_of_year",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
)

#: Categorical features. High-cardinality resource identifiers are excluded.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "provider",
    "service_category",
    "service_name",
    "billing_account_id",
)

ALL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

META_COLUMNS: tuple[str, ...] = (
    "series_id",
    "origin_date",
    "target_date",
    "provider",
    "billing_account_id",
    "service_name",
    "currency",
    "cost_basis",
    "target",
    "target_observed",
)


@dataclass(slots=True)
class FeatureBuildConfig:
    horizon_days: int = 30
    min_lag_days: int = 28
    rolling_min_periods: int = 4
    include_account: bool = True
    max_rows: int | None = 2_000_000

    @property
    def categorical_features(self) -> tuple[str, ...]:
        if self.include_account:
            return CATEGORICAL_FEATURES
        return tuple(name for name in CATEGORICAL_FEATURES if name != "billing_account_id")


@dataclass(slots=True)
class SupervisedBuildResult:
    """Supervised training frame plus provenance and data-quality counts."""

    frame: pd.DataFrame
    schema: FeatureSchema
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return list(self.schema.all_features)

    def features(self) -> pd.DataFrame:
        return self.frame.loc[:, _unique(self.feature_names)]


# --------------------------------------------------------------------------- #
# Public builders
# --------------------------------------------------------------------------- #
def build_supervised_dataset(
    panel: pd.DataFrame,
    *,
    config: FeatureBuildConfig | None = None,
    schema: FeatureSchema | dict[str, Any] | None = None,
) -> SupervisedBuildResult:
    """Turn a daily series panel into direct multi-horizon training rows.

    ``panel`` must be the output of :func:`app.ml.dataset_builder.build_daily_panel`
    (columns ``series_id, date, cost, observed, provider, billing_account_id,
    service_name, currency, cost_basis``).
    """
    config = config or FeatureBuildConfig()
    panel = _validate_panel(panel)
    categories = _category_mappings(panel, config.categorical_features)
    feature_schema = _coerce_schema(schema, categories, config.categorical_features)

    frames: list[pd.DataFrame] = []
    quality = {
        "series_total": 0,
        "series_used": 0,
        "series_skipped": 0,
        "candidate_rows": 0,
        "rows_with_missing_target": 0,
        "rows_returned": 0,
        "horizon_days": config.horizon_days,
        "min_lag_days": config.min_lag_days,
        "feature_schema_version": feature_schema.schema_version,
        "skipped_series": [],
    }

    for series_id, series in panel.groupby("series_id", sort=True):
        quality["series_total"] += 1
        series = series.sort_values("date").reset_index(drop=True)
        if len(series) <= config.min_lag_days + config.horizon_days:
            quality["series_skipped"] += 1
            quality["skipped_series"].append(
                {
                    "series_id": series_id,
                    "reason": (
                        f"{len(series)} calendar days; {config.min_lag_days + config.horizon_days + 1} "
                        "are required for the configured lags and horizon"
                    ),
                }
            )
            continue
        built = _series_training_rows(series, config)
        if built.empty:
            quality["series_skipped"] += 1
            quality["skipped_series"].append(
                {"series_id": series_id, "reason": "no origin produced a usable feature row"}
            )
            continue
        quality["series_used"] += 1
        quality["candidate_rows"] += int(len(built))
        quality["rows_with_missing_target"] += int((~built["target_observed"]).sum())
        frames.append(built)

    if not frames:
        empty = pd.DataFrame(columns=_unique([*META_COLUMNS, *feature_schema.all_features]))
        quality["rows_returned"] = 0
        quality["reason"] = "No series produced a usable training row."
        return SupervisedBuildResult(empty, feature_schema, quality)

    frame = pd.concat(frames, ignore_index=True)
    frame = _apply_categories(frame, feature_schema)
    # A row is only trainable when the label was actually billed.
    missing_target = int((~frame["target_observed"]).sum())
    frame = frame.loc[frame["target_observed"]].reset_index(drop=True)
    frame = frame.sort_values(["target_date", "series_id", "horizon"]).reset_index(drop=True)
    if config.max_rows and len(frame) > config.max_rows:
        quality["reason"] = (
            f"Built {len(frame)} rows which exceeds max_rows={config.max_rows}; "
            "narrow the training scope."
        )
        quality["rows_returned"] = 0
        return SupervisedBuildResult(frame.iloc[0:0].copy(), feature_schema, quality)

    quality["rows_with_missing_target"] = missing_target
    quality["rows_returned"] = int(len(frame))
    quality["origin_start"] = frame["origin_date"].min().date().isoformat()
    quality["origin_end"] = frame["origin_date"].max().date().isoformat()
    quality["target_start"] = frame["target_date"].min().date().isoformat()
    quality["target_end"] = frame["target_date"].max().date().isoformat()
    return SupervisedBuildResult(frame, feature_schema, quality)


def build_inference_features(
    panel: pd.DataFrame,
    *,
    schema: FeatureSchema | dict[str, Any],
    origin_date: dt.date | str | pd.Timestamp | None = None,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """Build the next ``horizon_days`` feature rows for one or more series.

    The origin defaults to the last observed date of each series. Historical
    features are calculated once at that origin; the forecast is direct, so
    no recursive predictions are used as lags.
    """
    panel = _validate_panel(panel)
    feature_schema = (
        schema if isinstance(schema, FeatureSchema) else FeatureSchema.model_validate(schema)
    )
    requested_origin = pd.Timestamp(origin_date).normalize() if origin_date is not None else None
    config = FeatureBuildConfig(horizon_days=int(horizon_days))

    frames: list[pd.DataFrame] = []
    for series_id, series in panel.groupby("series_id", sort=True):
        series = series.sort_values("date").reset_index(drop=True)
        observed_positions = np.flatnonzero(series["observed"].to_numpy(dtype=bool))
        if observed_positions.size == 0:
            continue
        if requested_origin is None:
            origin_pos = int(observed_positions[-1])
        else:
            matches = np.flatnonzero(series["date"].to_numpy() <= requested_origin.to_datetime64())
            if matches.size == 0:
                continue
            origin_pos = int(matches[-1])
        if origin_pos < config.min_lag_days:
            continue
        built = _series_inference_rows(series, origin_pos, config)
        if not built.empty:
            frames.append(built)

    ordered = _unique([*META_COLUMNS, *feature_schema.all_features])
    if not frames:
        return pd.DataFrame(columns=ordered)
    frame = pd.concat(frames, ignore_index=True)
    frame = _apply_categories(frame, feature_schema)
    for feature in feature_schema.numeric_features:
        if feature not in frame.columns:
            frame[feature] = np.nan
    return frame.loc[:, ordered]


# --------------------------------------------------------------------------- #
# Per-series builders
# --------------------------------------------------------------------------- #
def _series_training_rows(series: pd.DataFrame, config: FeatureBuildConfig) -> pd.DataFrame:
    dates = pd.to_datetime(series["date"]).to_numpy(dtype="datetime64[ns]")
    cost = pd.to_numeric(series["cost"], errors="coerce").to_numpy(dtype="float64")
    observed = series["observed"].to_numpy(dtype=bool)
    n = len(series)
    last_origin = n - 1 - config.horizon_days
    if last_origin < config.min_lag_days:
        return pd.DataFrame()

    origin_positions = np.arange(config.min_lag_days, last_origin + 1)
    historical = _historical_features(cost, observed, origin_positions, config)
    historical["origin_pos"] = origin_positions
    historical["origin_date"] = dates[origin_positions]

    horizons = np.arange(1, config.horizon_days + 1)
    origins = pd.DataFrame(
        {
            "origin_pos": np.repeat(origin_positions, len(horizons)),
            "horizon": np.tile(horizons, len(origin_positions)),
        }
    )
    origins = origins.merge(historical, on="origin_pos", how="left", validate="many_to_one")
    target_pos = origins["origin_pos"].to_numpy() + origins["horizon"].to_numpy()
    origins["target_date"] = dates[target_pos]
    origins["target"] = cost[target_pos]
    origins["target_observed"] = observed[target_pos]

    origins["series_id"] = series["series_id"].iloc[0]
    for column in ("provider", "billing_account_id", "service_name", "currency", "cost_basis"):
        origins[column] = series[column].iloc[0] if column in series.columns else None
    origins["service_category"] = classify_service(series["service_name"].iloc[0])

    _add_calendar_features(origins, "target_date")
    origins["horizon"] = origins["horizon"].astype("int16")
    return origins


def _series_inference_rows(series: pd.DataFrame, origin_pos: int, config: FeatureBuildConfig) -> pd.DataFrame:
    dates = pd.to_datetime(series["date"]).to_numpy(dtype="datetime64[ns]")
    cost = pd.to_numeric(series["cost"], errors="coerce").to_numpy(dtype="float64")
    observed = series["observed"].to_numpy(dtype=bool)

    historical = _historical_features(cost, observed, np.array([origin_pos]), config)
    horizons = np.arange(1, config.horizon_days + 1)
    frame = pd.DataFrame({"horizon": horizons})
    for column in historical.columns:
        frame[column] = historical[column].iloc[0]
    target_pos = origin_pos + horizons
    in_bounds = target_pos < len(series)
    # A forecast may extend past the last calendar date; target dates are still
    # well defined and no target cost is needed at inference time.
    frame["target_date"] = pd.Timestamp(dates[origin_pos]) + pd.to_timedelta(horizons, unit="D")
    if in_bounds.any():
        known = dates[target_pos[in_bounds]]
        frame.loc[in_bounds, "target_date"] = known
    frame["origin_date"] = pd.Timestamp(dates[origin_pos])
    frame["series_id"] = series["series_id"].iloc[0]
    for column in ("provider", "billing_account_id", "service_name", "currency", "cost_basis"):
        frame[column] = series[column].iloc[0] if column in series.columns else None
    frame["service_category"] = classify_service(series["service_name"].iloc[0])
    frame["target"] = np.nan
    frame["target_observed"] = False
    _add_calendar_features(frame, "target_date")
    frame["horizon"] = frame["horizon"].astype("int16")
    return frame


def _historical_features(
    cost: np.ndarray,
    observed: np.ndarray,
    origin_positions: np.ndarray,
    config: FeatureBuildConfig,
) -> pd.DataFrame:
    """Features available at each origin (inclusive of the origin day)."""
    series = pd.Series(cost, dtype="float64")
    rolling_min = config.rolling_min_periods

    def take(values: pd.Series) -> np.ndarray:
        return values.to_numpy(dtype="float64")[origin_positions]

    def lag(k: int) -> np.ndarray:
        positions = origin_positions - (k - 1)
        result = np.full(len(origin_positions), np.nan, dtype="float64")
        valid = positions >= 0
        result[valid] = cost[positions[valid]]
        return result

    features = pd.DataFrame(
        {
            "lag_1": lag(1),
            "lag_7": lag(7),
            "lag_14": lag(14),
            "lag_28": lag(28),
            "rolling_mean_7": take(series.rolling(7, min_periods=rolling_min).mean()),
            "rolling_median_7": take(series.rolling(7, min_periods=rolling_min).median()),
            "rolling_std_7": take(series.rolling(7, min_periods=rolling_min).std()),
            "rolling_sum_7": take(series.rolling(7, min_periods=rolling_min).sum()),
            "rolling_mean_28": take(series.rolling(28, min_periods=rolling_min).mean()),
            "rolling_median_28": take(series.rolling(28, min_periods=rolling_min).median()),
            "rolling_std_28": take(series.rolling(28, min_periods=rolling_min).std()),
            "rolling_sum_28": take(series.rolling(28, min_periods=rolling_min).sum()),
            "ewm_14": take(series.ewm(span=14, min_periods=rolling_min, adjust=False).mean()),
            "observed_ratio_28": take(
                pd.Series(observed.astype("float64")).rolling(28, min_periods=1).mean()
            ),
        }
    )

    mean7 = features["rolling_mean_7"].to_numpy()
    mean28 = features["rolling_mean_28"].to_numpy()
    features["trend_short"] = _safe_ratio(features["lag_1"].to_numpy() - mean7, mean7)
    features["trend_long"] = _safe_ratio(mean7 - mean28, mean28)
    features["days_since_last_observed"] = _days_since_observed(observed, origin_positions)

    # Historical windows are calendar-day windows, so a sparse series is
    # distinguishable from a truly flat one via the observed ratio.
    features["observed_ratio_28"] = features["observed_ratio_28"].clip(0.0, 1.0)
    return features


def _add_calendar_features(frame: pd.DataFrame, date_column: str) -> None:
    dates = pd.to_datetime(frame[date_column])
    frame["day_of_week"] = dates.dt.dayofweek.astype("int8")
    frame["is_weekend"] = (dates.dt.dayofweek >= 5).astype("int8")
    frame["day_of_month"] = dates.dt.day.astype("int8")
    frame["week_of_year"] = dates.dt.isocalendar().week.astype("int16")
    frame["month"] = dates.dt.month.astype("int8")
    frame["quarter"] = dates.dt.quarter.astype("int8")
    frame["day_of_year"] = dates.dt.dayofyear.astype("int16")
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7.0)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7.0)
    frame["month_sin"] = np.sin(2 * np.pi * (frame["month"] - 1) / 12.0)
    frame["month_cos"] = np.cos(2 * np.pi * (frame["month"] - 1) / 12.0)


# --------------------------------------------------------------------------- #
# Schema helpers
# --------------------------------------------------------------------------- #
def build_feature_schema(
    panel: pd.DataFrame,
    *,
    categorical_features: Iterable[str] | None = None,
) -> FeatureSchema:
    names = tuple(categorical_features or CATEGORICAL_FEATURES)
    mappings = _category_mappings(_validate_panel(panel), names)
    return FeatureSchema(
        schema_version=FEATURE_SCHEMA_VERSION,
        numeric_features=list(NUMERIC_FEATURES),
        categorical_features=list(names),
        category_mappings=mappings,
        all_features=[*NUMERIC_FEATURES, *names],
    )


def _category_mappings(panel: pd.DataFrame, categorical_features: Iterable[str]) -> dict[str, list[str]]:
    mappings: dict[str, list[str]] = {}
    for feature in categorical_features:
        if feature == "service_category":
            values = panel["service_name"].dropna().astype(str).map(classify_service)
        elif feature in panel.columns:
            values = panel[feature].dropna().astype(str)
        else:
            values = pd.Series(dtype="object")
        mappings[feature] = sorted({value for value in values if value not in {"", "nan", "None"}})
    return mappings


def _coerce_schema(
    schema: FeatureSchema | dict[str, Any] | None,
    mappings: dict[str, list[str]],
    categorical_features: tuple[str, ...],
) -> FeatureSchema:
    if schema is None:
        return FeatureSchema(
            schema_version=FEATURE_SCHEMA_VERSION,
            numeric_features=list(NUMERIC_FEATURES),
            categorical_features=list(categorical_features),
            category_mappings=mappings,
            all_features=[*NUMERIC_FEATURES, *categorical_features],
        )
    resolved = schema if isinstance(schema, FeatureSchema) else FeatureSchema.model_validate(schema)
    if not resolved.category_mappings:
        resolved.category_mappings = mappings
    if not resolved.numeric_features:
        resolved.numeric_features = list(NUMERIC_FEATURES)
    if not resolved.categorical_features:
        resolved.categorical_features = list(categorical_features)
    if not resolved.all_features:
        resolved.all_features = [*resolved.numeric_features, *resolved.categorical_features]
    return resolved


def _apply_categories(frame: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    for feature in schema.categorical_features:
        categories = schema.category_mappings.get(feature)
        values = frame[feature] if feature in frame.columns else pd.Series([pd.NA] * len(frame))
        frame[feature] = pd.Categorical(
            values.astype("object").where(values.notna(), None),
            categories=categories or None,
        )
    for feature in schema.numeric_features:
        if feature in frame.columns:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce").astype("float64")
    return frame


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if panel is None or panel.empty:
        raise ValueError("Cannot engineer features: the daily panel is empty.")
    required = {"series_id", "date", "cost", "observed"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError("Daily panel is missing required columns: " + ", ".join(missing))
    working = panel.copy()
    working["date"] = pd.to_datetime(working["date"]).dt.normalize()
    working["observed"] = working["observed"].fillna(False).astype(bool)
    working["cost"] = pd.to_numeric(working["cost"], errors="coerce")
    return working


def _days_since_observed(observed: np.ndarray, origin_positions: np.ndarray) -> np.ndarray:
    positions = np.arange(len(observed))
    last_observed = np.maximum.accumulate(np.where(observed, positions, -1))
    result = origin_positions - last_observed[origin_positions]
    result = result.astype("float64")
    result[result < 0] = np.nan
    return result


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype="float64")
    denominator = np.asarray(denominator, dtype="float64")
    result = np.full(numerator.shape, np.nan, dtype="float64")
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-9)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _unique(names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def feature_summary(frame: pd.DataFrame, schema: FeatureSchema) -> dict[str, Any]:
    """Describe engineered features for reports and model cards."""
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "numeric_features": list(schema.numeric_features),
        "categorical_features": list(schema.categorical_features),
    }
    missing = {}
    for feature in schema.all_features:
        if feature in frame.columns:
            missing[feature] = float(frame[feature].isna().mean())
    summary["missing_ratio"] = missing
    return summary
