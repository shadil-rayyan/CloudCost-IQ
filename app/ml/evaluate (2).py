"""Forecast evaluation: point metrics, quantile metrics, calibration and baselines.

All metrics are computed from out-of-sample predictions only. Baselines are
simple reference methods (trailing 7-day average and seasonal-naive weekly);
they are benchmarks, not additional ML models.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)
QUANTILE_COLUMNS: tuple[str, str, str] = ("p10", "p50", "p90")
HORIZON_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("1-7", 1, 7),
    ("8-14", 8, 14),
    ("15-21", 15, 21),
    ("22-30", 22, 30),
)

BASELINE_COLUMNS = {
    "trailing_7d_average": ("rolling_mean_7", "rolling_mean_28"),
    "seasonal_naive_weekly": ("lag_7", "lag_14"),
}


# --------------------------------------------------------------------------- #
# Primitive metrics
# --------------------------------------------------------------------------- #
def pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    actual, predicted = _aligned(actual, predicted)
    if actual.size == 0:
        return float("nan")
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = _aligned(actual, predicted)
    return float(np.mean(np.abs(actual - predicted))) if actual.size else float("nan")


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = _aligned(actual, predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2))) if actual.size else float("nan")


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = _aligned(actual, predicted)
    if actual.size == 0:
        return float("nan")
    denominator = float(np.sum(np.abs(actual)))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.sum(np.abs(actual - predicted)) / denominator)


def mase(actual: np.ndarray, predicted: np.ndarray, naive_scale: float) -> float:
    if not np.isfinite(naive_scale) or naive_scale <= 1e-12:
        return float("nan")
    return mae(actual, predicted) / naive_scale


def enforce_monotonic(
    frame: pd.DataFrame,
    columns: tuple[str, str, str] = QUANTILE_COLUMNS,
) -> tuple[pd.DataFrame, int]:
    """Sort each row's quantiles so ``p10 <= p50 <= p90``.

    Returns the corrected frame and the number of rows that were reordered.
    The raw crossing rate is reported separately; crossing is never silently
    ignored.
    """
    working = frame.copy()
    values = working.loc[:, list(columns)].to_numpy(dtype="float64")
    if values.size == 0:
        return working, 0
    corrected = np.sort(values, axis=1)
    crossed = int(np.any(np.diff(values, axis=1) < 0, axis=1).sum())
    working.loc[:, list(columns)] = corrected
    return working, crossed


def quantile_crossing_rate(frame: pd.DataFrame, columns: tuple[str, str, str] = QUANTILE_COLUMNS) -> float:
    values = frame.loc[:, list(columns)].to_numpy(dtype="float64")
    if values.size == 0:
        return float("nan")
    crossing = np.any(np.diff(values, axis=1) < 0, axis=1)
    return float(np.mean(crossing))


# --------------------------------------------------------------------------- #
# Full evaluation
# --------------------------------------------------------------------------- #
def evaluate_predictions(
    frame: pd.DataFrame,
    *,
    group_columns: Iterable[str] = ("provider",),
    include_horizon_buckets: bool = True,
) -> dict[str, Any]:
    """Evaluate out-of-sample predictions.

    The frame must contain ``target``, ``p10``, ``p50`` and ``p90``. Optional
    metadata columns (``provider``, ``horizon``, ``series_id``) enable grouped
    reporting. ``lag_7`` / ``rolling_mean_7`` are required only for the
    baseline comparison.
    """
    required = {"target", "p10", "p50", "p90"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Prediction frame is missing columns: " + ", ".join(missing))
    if frame.empty:
        return _empty_report()

    working = frame.copy()
    for column in ("target", "p10", "p50", "p90"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    raw_crossing_rate = quantile_crossing_rate(working)
    working, _ = enforce_monotonic(working)

    actual = working["target"].to_numpy(dtype="float64")
    p50 = working["p50"].to_numpy(dtype="float64")
    report = _point_and_quantile_metrics(working, actual, p50)
    report["quantile_crossing_rate"] = raw_crossing_rate
    report["baselines"] = _baseline_metrics(working)
    report["baseline_comparison"] = _baseline_comparison(report)

    grouped: dict[str, Any] = {}
    for column in group_columns:
        if column not in working.columns:
            continue
        grouped[column] = {
            str(value): _point_and_quantile_metrics(
                subset,
                subset["target"].to_numpy(dtype="float64"),
                subset["p50"].to_numpy(dtype="float64"),
            )
            for value, subset in working.groupby(column, dropna=False)
            if len(subset)
        }
    report["by_group"] = grouped

    if include_horizon_buckets and "horizon" in working.columns:
        report["by_horizon"] = {}
        for label, start, end in HORIZON_BUCKETS:
            subset = working.loc[working["horizon"].between(start, end)]
            if len(subset):
                report["by_horizon"][label] = _point_and_quantile_metrics(
                    subset,
                    subset["target"].to_numpy(dtype="float64"),
                    subset["p50"].to_numpy(dtype="float64"),
                )
    return report


def summarise_folds(fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate fold-level metrics for the model card and validation gates."""
    if not fold_reports:
        return {"folds": 0}
    keys = ("mae", "rmse", "wape", "mase", "pinball_mean", "interval_coverage", "interval_width")
    summary: dict[str, Any] = {"folds": len(fold_reports)}
    for key in keys:
        values = [report.get(key) for report in fold_reports]
        clean = [float(value) for value in values if value is not None and np.isfinite(value)]
        summary[key] = float(np.mean(clean)) if clean else None
        summary[f"{key}_values"] = values
    return summary


def evaluate_bundle_gates(
    metrics: dict[str, Any],
    gates: dict[str, Any],
    *,
    trained_on_final_test: bool = False,
) -> list[dict[str, Any]]:
    """Run the documented promotion gates against measured metrics."""
    final = metrics.get("final_test", {})
    comparison = metrics.get("baseline_comparison", {})
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    max_wape = gates.get("max_wape")
    if max_wape is not None:
        value = final.get("wape")
        check(
            "wape_within_limit",
            value is not None and np.isfinite(value) and value <= float(max_wape),
            f"wape={value} limit={max_wape}",
        )
    ratio_limit = gates.get("max_pinball_loss_ratio_vs_baseline")
    if ratio_limit is not None:
        value = comparison.get("pinball_ratio_vs_best_baseline")
        check(
            "pinball_beats_baseline",
            value is not None and np.isfinite(value) and value <= float(ratio_limit),
            f"ratio={value} limit={ratio_limit}",
        )
    min_cov = gates.get("min_interval_coverage")
    max_cov = gates.get("max_interval_coverage")
    coverage = final.get("interval_coverage")
    if min_cov is not None and max_cov is not None:
        check(
            "interval_calibration",
            coverage is not None and float(min_cov) <= coverage <= float(max_cov),
            f"coverage={coverage} expected=[{min_cov}, {max_cov}]",
        )
    crossing_limit = gates.get("max_quantile_crossing_rate")
    if crossing_limit is not None:
        rate = final.get("quantile_crossing_rate")
        check(
            "quantile_crossing_within_limit",
            rate is not None and rate <= float(crossing_limit),
            f"rate={rate} limit={crossing_limit}",
        )
    if gates.get("require_baseline_comparison", True):
        check(
            "baseline_comparison_present",
            bool(metrics.get("baseline_comparison")),
            "baseline comparison recorded",
        )
    check(
        "final_test_untouched",
        not trained_on_final_test,
        "final test period excluded from model selection" if not trained_on_final_test else "final test was used for selection",
    )
    return checks


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _point_and_quantile_metrics(
    frame: pd.DataFrame,
    actual: np.ndarray,
    p50: np.ndarray,
) -> dict[str, Any]:
    p10 = frame["p10"].to_numpy(dtype="float64")
    p90 = frame["p90"].to_numpy(dtype="float64")
    pinball = {
        "p10": pinball_loss(actual, p10, 0.10),
        "p50": pinball_loss(actual, p50, 0.50),
        "p90": pinball_loss(actual, p90, 0.90),
    }
    pinball_values = [value for value in pinball.values() if np.isfinite(value)]
    coverage = _interval_coverage(actual, p10, p90)
    width = _interval_width(p10, p90)
    return {
        "rows": int(len(frame)),
        "mae": mae(actual, p50),
        "rmse": rmse(actual, p50),
        "wape": wape(actual, p50),
        "mase": mase(actual, p50, _naive_scale(frame, actual)),
        "pinball": pinball,
        "pinball_mean": float(np.mean(pinball_values)) if pinball_values else float("nan"),
        "interval_coverage": coverage,
        "interval_width": width,
        "mean_actual": float(np.nanmean(actual)) if len(actual) else float("nan"),
        "mean_predicted": float(np.nanmean(p50)) if len(p50) else float("nan"),
    }


def _naive_scale(frame: pd.DataFrame, actual: np.ndarray) -> float:
    if "lag_7" in frame.columns:
        naive = pd.to_numeric(frame["lag_7"], errors="coerce").to_numpy(dtype="float64")
        mask = np.isfinite(actual) & np.isfinite(naive)
        if mask.any():
            scale = float(np.mean(np.abs(actual[mask] - naive[mask])))
            if scale > 1e-12:
                return scale
    if "rolling_mean_7" in frame.columns:
        baseline = pd.to_numeric(frame["rolling_mean_7"], errors="coerce").to_numpy(dtype="float64")
        mask = np.isfinite(actual) & np.isfinite(baseline)
        if mask.any():
            return float(np.mean(np.abs(actual[mask] - baseline[mask])))
    return float("nan")


def _interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    if not mask.any():
        return float("nan")
    return float(np.mean((actual[mask] >= lower[mask]) & (actual[mask] <= upper[mask])))


def _interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    mask = np.isfinite(lower) & np.isfinite(upper)
    if not mask.any():
        return float("nan")
    return float(np.mean(upper[mask] - lower[mask]))


def _baseline_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    actual = frame["target"].to_numpy(dtype="float64")
    baselines: dict[str, Any] = {}
    for name, candidates in BASELINE_COLUMNS.items():
        prediction = _first_available(frame, candidates)
        if prediction is None:
            baselines[name] = {"available": False}
            continue
        mask = np.isfinite(actual) & np.isfinite(prediction)
        if not mask.any():
            baselines[name] = {"available": False}
            continue
        selected_actual = actual[mask]
        selected_prediction = prediction[mask]
        baselines[name] = {
            "available": True,
            "rows": int(mask.sum()),
            "mae": mae(selected_actual, selected_prediction),
            "rmse": rmse(selected_actual, selected_prediction),
            "wape": wape(selected_actual, selected_prediction),
            "pinball_p50": pinball_loss(selected_actual, selected_prediction, 0.50),
        }
    return baselines


def _first_available(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray | None:
    result: np.ndarray | None = None
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
        result = values if result is None else np.where(np.isfinite(result), result, values)
    return result


def _baseline_comparison(report: dict[str, Any]) -> dict[str, Any]:
    model_wape = report.get("wape")
    model_pinball = report.get("pinball_mean")
    baselines = report.get("baselines", {})
    comparison: dict[str, Any] = {"model_wape": model_wape}
    for name, values in baselines.items():
        if not values.get("available"):
            continue
        base_wape = values.get("wape")
        base_pinball = values.get("pinball_p50")
        comparison[f"wape_{name}"] = base_wape
        if model_wape is not None and base_wape is not None and np.isfinite(base_wape) and base_wape > 1e-12:
            comparison[f"wape_improvement_vs_{name}"] = float(1.0 - model_wape / base_wape)
    best = [
        values.get("pinball_p50")
        for values in baselines.values()
        if values.get("available") and values.get("pinball_p50") is not None
    ]
    if best and model_pinball is not None and np.isfinite(model_pinball):
        best_value = float(min(best))
        comparison["best_baseline_pinball_p50"] = best_value
        comparison["pinball_ratio_vs_best_baseline"] = (
            float(model_pinball / best_value) if best_value > 1e-12 else None
        )
    return comparison


def _aligned(actual: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    mask = np.isfinite(actual) & np.isfinite(predicted)
    return actual[mask], predicted[mask]


def _empty_report() -> dict[str, Any]:
    return {
        "rows": 0,
        "mae": float("nan"),
        "rmse": float("nan"),
        "wape": float("nan"),
        "mase": float("nan"),
        "pinball": {"p10": float("nan"), "p50": float("nan"), "p90": float("nan")},
        "pinball_mean": float("nan"),
        "interval_coverage": float("nan"),
        "interval_width": float("nan"),
        "quantile_crossing_rate": float("nan"),
        "baselines": {},
        "baseline_comparison": {},
        "by_group": {},
        "by_horizon": {},
    }
