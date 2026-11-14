"""Model explainability: native gain importance plus optional SHAP values.

Feature importance describes how the model uses inputs; it is not evidence of
causation. SHAP results are computed for the P50 output unless a specific
quantile is requested.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.ml.artifact_manager import ModelBundle


def native_importance(bundle: ModelBundle) -> list[dict[str, Any]]:
    """Return XGBoost's native split/gain importance for every feature."""
    model = bundle.model
    try:
        scores = model.get_booster().get_score(importance_type="gain")
    except Exception:  # pragma: no cover - defensive
        scores = {}
    features = list(bundle.feature_schema.all_features)
    if not scores:
        try:
            values = np.asarray(model.feature_importances_, dtype="float64")
            scores = {
                feature: float(value)
                for feature, value in zip(features, values, strict=False)
            }
        except Exception:  # pragma: no cover - defensive
            scores = {}
    total = float(sum(scores.values())) or 1.0
    rows = [
        {
            "feature": feature,
            "importance": float(scores.get(feature, 0.0)),
            "importance_ratio": float(scores.get(feature, 0.0)) / total,
        }
        for feature in features
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)


def shap_values(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    *,
    max_rows: int = 2000,
    quantile_index: int = 1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute mean absolute SHAP values for a sample of inference rows.

    Returns ``{"available": False, "reason": ...}`` when SHAP is unavailable
    rather than failing the request.
    """
    features = list(bundle.feature_schema.all_features)
    sample = frame.loc[:, features].head(max_rows).copy()
    if sample.empty:
        return {"available": False, "reason": "No rows were supplied for explainability."}
    try:
        import shap
    except Exception:  # pragma: no cover - optional dependency
        return {"available": False, "reason": "The shap package is not installed."}
    try:
        explainer = shap.TreeExplainer(bundle.model)
        values = explainer.shap_values(sample)
    except Exception as exc:  # pragma: no cover - version dependent
        return {"available": False, "reason": f"SHAP could not explain this model ({type(exc).__name__})."}

    matrix = _select_quantile(values, quantile_index, len(sample), len(features))
    if matrix is None:
        return {"available": False, "reason": "SHAP returned an unexpected output shape."}
    mean_abs = np.abs(matrix).mean(axis=0)
    rows = [
        {"feature": feature, "mean_abs_shap": float(value)}
        for feature, value in zip(features, mean_abs, strict=False)
    ]
    rows.sort(key=lambda row: row["mean_abs_shap"], reverse=True)
    return {
        "available": True,
        "quantile": bundle.model_config.quantile_alpha[quantile_index]
        if quantile_index < len(bundle.model_config.quantile_alpha)
        else None,
        "rows": int(len(sample)),
        "values": rows,
    }


def explain_forecast(bundle: ModelBundle, frame: pd.DataFrame, *, max_rows: int = 2000) -> dict[str, Any]:
    """Combine native importance and SHAP into one explainability document."""
    return {
        "model_version": bundle.version,
        "native_importance": native_importance(bundle),
        "shap": shap_values(bundle, frame, max_rows=max_rows),
        "notes": [
            "Feature importance reflects model usage, not causation.",
            "SHAP values describe marginal contributions for the sampled rows only.",
        ],
    }


def _select_quantile(values: Any, quantile_index: int, rows: int, features: int) -> np.ndarray | None:
    array = np.asarray(values, dtype="object") if isinstance(values, list) else values
    if isinstance(values, list):
        # Older SHAP versions return one array per output.
        if not values:
            return None
        selected = values[min(quantile_index, len(values) - 1)]
        selected = np.asarray(selected, dtype="float64")
        return selected if selected.shape == (rows, features) else None
    array = np.asarray(array, dtype="float64")
    if array.ndim == 2 and array.shape == (rows, features):
        return array
    if array.ndim == 3 and array.shape[0] == rows and array.shape[1] == features:
        return array[:, :, min(quantile_index, array.shape[2] - 1)]
    if array.ndim == 3 and array.shape[1] == rows and array.shape[2] == features:
        return array[min(quantile_index, array.shape[0] - 1)]
    return None
