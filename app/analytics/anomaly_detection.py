"""Cost anomaly detection.

MVP 1 uses a transparent robust-statistics rule (rolling median + median
absolute deviation). MVP 2 adds comparison of actual spend against a validated
model's prediction interval. An anomaly is always reported as *unusual
spending*, never as a confirmed infrastructure failure.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from app.processing.aggregation import add_billing_date

ANOMALY_COLUMNS = [
    "detection_date",
    "provider",
    "billing_account_id",
    "service_name",
    "currency",
    "cost_basis",
    "expected_cost",
    "observed_cost",
    "difference",
    "difference_pct",
    "severity",
    "method",
    "explanation",
]

DEFAULT_KEYS = ("provider", "billing_account_id", "service_name")


def _severity(score: float, threshold: float) -> str:
    if score >= threshold * 1.6:
        return "high"
    if score >= threshold * 1.2:
        return "medium"
    return "low"


def detect_statistical_anomalies(
    frame: pd.DataFrame,
    *,
    lookback_days: int = 28,
    threshold: float = 3.0,
    min_history: int = 10,
    group_keys: tuple[str, ...] = DEFAULT_KEYS,
) -> pd.DataFrame:
    """Detect daily spend that deviates from each series' recent baseline.

    The baseline for a day only uses **previous** days (shifted rolling
    statistics), so a spike cannot hide itself by inflating its own baseline.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=ANOMALY_COLUMNS)

    working = add_billing_date(frame)
    keys = [key for key in group_keys if key in working.columns]
    results: list[dict[str, Any]] = []

    for key_values, group in working.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        daily = (
            group.groupby("billing_date", dropna=False)["cost_amount"]
            .sum()
            .sort_index()
            .astype("float64")
        )
        daily = daily.loc[daily.index.notna()]
        if len(daily) < min_history + 1:
            continue
        history = daily.shift(1)
        median = history.rolling(lookback_days, min_periods=min_history).median()
        mad = history.rolling(lookback_days, min_periods=min_history).apply(
            lambda values: float(np.nanmedian(np.abs(values - np.nanmedian(values)))), raw=True
        )
        scale = np.maximum.reduce(
            [
                (mad * 1.4826).fillna(0.0).to_numpy(),
                (median.abs() * 0.05).fillna(0.0).to_numpy(),
                np.full(len(daily), 1e-6),
            ]
        )
        scale_series = pd.Series(scale, index=daily.index).replace(0, 1e-6)
        scores = (daily - median).abs() / scale_series
        flagged = (scores >= threshold) & median.notna()
        if not flagged.any():
            continue
        currency = _first_value(group["currency"])
        basis = _first_value(group["cost_basis"])
        for date, observed in daily.loc[flagged].items():
            expected = float(median.loc[date])
            observed_value = float(observed)
            difference = observed_value - expected
            results.append(
                {
                    "detection_date": pd.Timestamp(date).date().isoformat(),
                    "provider": dict(zip(keys, key_values, strict=False)).get("provider"),
                    "billing_account_id": dict(zip(keys, key_values, strict=False)).get(
                        "billing_account_id"
                    ),
                    "service_name": dict(zip(keys, key_values, strict=False)).get("service_name"),
                    "currency": currency,
                    "cost_basis": basis,
                    "expected_cost": expected,
                    "observed_cost": observed_value,
                    "difference": difference,
                    "difference_pct": (difference / abs(expected)) if expected else None,
                    "severity": _severity(float(scores.loc[date]), threshold),
                    "method": "rolling_median_mad",
                    "explanation": (
                        f"Observed daily cost deviates from the previous {lookback_days}-day "
                        f"median by {float(scores.loc[date]):.1f} robust standard deviations. "
                        "This indicates unusual spending, not a confirmed outage."
                    ),
                }
            )

    if not results:
        return pd.DataFrame(columns=ANOMALY_COLUMNS)
    result = pd.DataFrame(results)
    result["_abs"] = result["difference"].abs()
    result = result.sort_values(["detection_date", "_abs"], ascending=[False, False]).drop(
        columns="_abs"
    )
    return result[ANOMALY_COLUMNS].reset_index(drop=True)


def detect_from_forecast(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    keys: tuple[str, ...] = ("provider", "billing_account_id", "service_name"),
) -> pd.DataFrame:
    """Compare realized daily cost with a validated model's P10/P90 interval.

    ``actual`` must contain ``date`` and ``cost`` (or ``cost_amount``).
    ``forecast`` must contain ``date``, ``p10``, ``p50`` and ``p90``.
    """
    if actual is None or actual.empty or forecast is None or forecast.empty:
        return pd.DataFrame(columns=ANOMALY_COLUMNS)

    actual_frame = actual.copy()
    if "cost" not in actual_frame.columns:
        actual_frame = actual_frame.rename(columns={"cost_amount": "cost"})
    actual_frame["date"] = pd.to_datetime(actual_frame["date"], errors="coerce").dt.normalize()
    forecast_frame = forecast.copy()
    forecast_frame["date"] = pd.to_datetime(forecast_frame["date"], errors="coerce").dt.normalize()

    merged = actual_frame.merge(forecast_frame, on="date", how="inner", suffixes=("", "_fc"))
    if merged.empty:
        return pd.DataFrame(columns=ANOMALY_COLUMNS)

    outside = (merged["cost"] > merged["p90"]) | (merged["cost"] < merged["p10"])
    merged = merged.loc[outside.fillna(False)]
    if merged.empty:
        return pd.DataFrame(columns=ANOMALY_COLUMNS)

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        expected = float(row["p50"])
        observed = float(row["cost"])
        upper = float(row["p90"])
        lower = float(row["p10"])
        if observed > upper:
            exceedance = (observed - upper) / abs(upper) if upper else 1.0
            direction = "above the P90 upper bound"
        else:
            exceedance = (lower - observed) / abs(lower) if lower else 1.0
            direction = "below the P10 lower bound"
        severity = "high" if exceedance >= 0.5 else "medium" if exceedance >= 0.2 else "low"
        rows.append(
            {
                "detection_date": pd.Timestamp(row["date"]).date().isoformat(),
                "provider": row.get("provider"),
                "billing_account_id": row.get("billing_account_id"),
                "service_name": row.get("service_name"),
                "currency": row.get("currency"),
                "cost_basis": row.get("cost_basis"),
                "expected_cost": expected,
                "observed_cost": observed,
                "difference": observed - expected,
                "difference_pct": ((observed - expected) / abs(expected)) if expected else None,
                "severity": severity,
                "method": "forecast_interval",
                "explanation": (
                    f"Actual daily cost is {direction} of the validated forecast interval "
                    f"(P10 {lower:,.2f} / P50 {expected:,.2f} / P90 {upper:,.2f})."
                ),
            }
        )
    return pd.DataFrame(rows, columns=ANOMALY_COLUMNS).sort_values(
        "detection_date", ascending=False
    )


def _first_value(series: pd.Series) -> Any:
    if series is None or series.dropna().empty:
        return None
    return series.dropna().iloc[0]


def anomaly_summary(anomalies: pd.DataFrame) -> dict[str, Any]:
    if anomalies is None or anomalies.empty:
        return {"count": 0, "by_severity": {}, "latest_date": None}
    return {
        "count": int(len(anomalies)),
        "by_severity": anomalies["severity"].value_counts().astype(int).to_dict(),
        "latest_date": anomalies["detection_date"].iloc[0] if len(anomalies) else None,
    }
