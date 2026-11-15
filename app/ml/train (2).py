"""Global multi-horizon XGBoost quantile training.

Training follows the documented protocol:

1. Fit on each rolling-origin training window with chronological early stopping.
2. Evaluate each fold out-of-sample.
3. Fit the untouched final test model using the number of rounds selected from
   the validation folds (never early stopping on the test window).
4. Report final test metrics, run promotion gates, then refit on all approved
   rows to produce the deployable candidate artifact.

Nothing is promoted automatically; a candidate that fails a gate stays a
candidate and the dashboard refuses to serve it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_ml_config, get_settings
from app.core.constants import NORMALIZED_SCHEMA_VERSION
from app.core.exceptions import InsufficientHistoryError
from app.ml.artifact_manager import ArtifactManager
from app.ml.evaluate import evaluate_bundle_gates, evaluate_predictions, summarise_folds
from app.ml.feature_engineering import (
    FEATURE_SCHEMA_VERSION,
    SupervisedBuildResult,
    build_feature_schema,
)
from app.ml.time_series_split import SplitPlan
from app.utils.dates import utc_now


@dataclass(slots=True)
class TrainingConfig:
    model_name: str = "cloudcost_iq_global_xgboost_quantile"
    objective: str = "reg:quantileerror"
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90)
    tree_method: str = "hist"
    horizon_days: int = 30
    random_state: int = 42
    enable_categorical: bool = True
    max_cat_to_onehot: int = 4
    hyperparameters: dict[str, Any] = field(
        default_factory=lambda: {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 600,
            "min_child_weight": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "early_stopping_rounds": 50,
        }
    )
    gates: dict[str, Any] = field(default_factory=dict)
    mlflow_enabled: bool = False
    mlflow_experiment: str = "cloudcost-iq-forecasting"
    target: str = "daily_billed_cost"
    cost_basis: str = "billed"
    currency: str = "USD"

    @classmethod
    def from_ml_config(cls, *, currency: str | None = None, cost_basis: str | None = None) -> "TrainingConfig":
        config = get_ml_config()
        model = config.get("model", {})
        training = config.get("training", {})
        settings = get_settings()
        mlflow_enabled = (
            settings.mlflow_enabled
            if settings.mlflow_enabled is not None
            else bool(training.get("mlflow_enabled", False))
        )
        return cls(
            model_name=str(model.get("model_name", "cloudcost_iq_global_xgboost_quantile")),
            objective=str(model.get("objective", "reg:quantileerror")),
            quantiles=tuple(float(value) for value in model.get("quantile_alpha", [0.1, 0.5, 0.9])),
            tree_method=str(model.get("tree_method", "hist")),
            horizon_days=int(model.get("forecast_horizon_days", 30)),
            random_state=int(model.get("random_state", 42)),
            enable_categorical=bool(model.get("enable_categorical", True)),
            max_cat_to_onehot=int(model.get("max_cat_to_onehot", 4)),
            hyperparameters=dict(config.get("hyperparameters", {})),
            gates=dict(config.get("evaluation", {})),
            mlflow_enabled=mlflow_enabled,
            mlflow_experiment=str(training.get("mlflow_experiment", "cloudcost-iq-forecasting")),
            target=str(training.get("target", "daily_billed_cost")),
            cost_basis=str(cost_basis or training.get("cost_basis", "billed")),
            currency=str(currency or training.get("currency", "USD")).upper(),
        )


@dataclass(slots=True)
class FoldOutcome:
    fold: str
    kind: str
    rounds: int
    metrics: dict[str, Any]
    train_rows: int
    validation_rows: int
    validation_start: str | None
    validation_end: str | None


@dataclass(slots=True)
class TrainingResult:
    model: Any
    model_config: dict[str, Any]
    feature_schema: dict[str, Any]
    metrics: dict[str, Any]
    manifest: dict[str, Any]
    dataset_quality: dict[str, Any]
    model_card: str
    version: str
    rounds: int
    gates: list[dict[str, Any]]
    passes_gates: bool
    bundle_path: Path | None = None
    evaluation_predictions: pd.DataFrame | None = None
    messages: list[str] = field(default_factory=list)

    @property
    def failed_gates(self) -> list[str]:
        return [gate["name"] for gate in self.gates if not gate["passed"]]


def train_forecast_model(
    supervised: SupervisedBuildResult,
    splits: SplitPlan,
    *,
    dataset_quality: dict[str, Any] | None = None,
    config: TrainingConfig | None = None,
    artifacts: ArtifactManager | None = None,
    version: str | None = None,
    save_bundle: bool = True,
    track_mlflow: bool | None = None,
) -> TrainingResult:
    """Train the global model, evaluate it and optionally save a candidate bundle."""
    import xgboost as xgb  # local import keeps non-ML pages light

    if supervised.frame.empty:
        raise InsufficientHistoryError("No supervised training rows are available.")
    if not splits.is_valid or splits.final_test is None:
        raise InsufficientHistoryError(
            "The chronological split plan is incomplete. "
            + " ".join(splits.messages or ["At least one validation fold and a final test window are required."])
        )

    config = config or TrainingConfig.from_ml_config()
    dataset_quality = dict(dataset_quality or supervised.quality or {})
    feature_schema = supervised.schema
    features = list(feature_schema.all_features)
    frame = supervised.frame

    x = frame.loc[:, features]
    y = frame["target"].to_numpy(dtype="float64")

    outcomes: list[FoldOutcome] = []
    fold_predictions: list[pd.DataFrame] = []
    for fold in splits.folds:
        outcome, predictions = _fit_and_evaluate_fold(
            xgb,
            x=x,
            y=y,
            frame=frame,
            train_index=fold.train_index,
            validation_index=fold.validation_index,
            config=config,
            fold_name=fold.name,
            kind=fold.kind,
            validation_start=fold.validation_start,
            validation_end=fold.validation_end,
            early_stopping=True,
        )
        outcomes.append(outcome)
        predictions["fold"] = fold.name
        fold_predictions.append(predictions)

    if not outcomes:
        raise InsufficientHistoryError("No validation fold could be trained.")

    selected_rounds = int(
        max(
            1,
            min(
                int(np.median([outcome.rounds for outcome in outcomes])),
                int(config.hyperparameters.get("n_estimators", 600)),
            ),
        )
    )

    test_fold = splits.final_test
    test_outcome, test_predictions = _fit_and_evaluate_fold(
        xgb,
        x=x,
        y=y,
        frame=frame,
        train_index=test_fold.train_index,
        validation_index=test_fold.validation_index,
        config=config,
        fold_name=test_fold.name,
        kind="test",
        validation_start=test_fold.validation_start,
        validation_end=test_fold.validation_end,
        early_stopping=False,
        fixed_rounds=selected_rounds,
    )
    test_predictions["fold"] = test_fold.name

    fold_metrics = [outcome.metrics for outcome in outcomes]
    cv_summary = summarise_folds(fold_metrics)
    metrics = {
        "generated_at": utc_now().isoformat(),
        "final_test": test_outcome.metrics,
        "folds": [
            {
                "fold": outcome.fold,
                "train_rows": outcome.train_rows,
                "validation_rows": outcome.validation_rows,
                "validation_start": outcome.validation_start,
                "validation_end": outcome.validation_end,
                "rounds": outcome.rounds,
                **outcome.metrics,
            }
            for outcome in outcomes
        ],
        "cv_summary": cv_summary,
        "baseline_comparison": test_outcome.metrics.get("baseline_comparison", {}),
        "fold_baseline_comparison": {
            outcome.fold: outcome.metrics.get("baseline_comparison", {}) for outcome in outcomes
        },
        "selection": {
            "selected_rounds": selected_rounds,
            "selection_rule": "median best_iteration across validation folds",
            "final_test_used_for_selection": False,
            "refit_rows": int(len(frame)),
        },
    }

    gates = evaluate_bundle_gates(metrics, config.gates)
    passes_gates = all(gate["passed"] for gate in gates)

    # Deployable model: refit on every approved row using the selected rounds.
    final_model = _new_model(xgb, config, rounds=selected_rounds, early_stopping=False)
    final_model.fit(x, y, verbose=False)

    quality = dict(dataset_quality)
    quality.setdefault("normalized_schema_version", NORMALIZED_SCHEMA_VERSION)
    quality.setdefault("feature_schema_version", feature_schema.schema_version)

    model_version = version or ArtifactManager.new_version()
    model_config = _model_config_document(config, final_model, model_version, selected_rounds)
    manifest = _training_manifest(
        model_version=model_version,
        config=config,
        quality=quality,
        frame=frame,
        supervised=supervised,
    )
    model_card = build_model_card(
        version=model_version,
        config=config,
        metrics=metrics,
        quality=quality,
        gates=gates,
        rounds=selected_rounds,
    )

    messages = list(splits.messages)
    if not passes_gates:
        messages.append(
            "The candidate failed the promotion gates ("
            + ", ".join(gate["name"] for gate in gates if not gate["passed"])
            + ") and must not be served in production."
        )
    else:
        messages.append("All promotion gates passed. Promotion still requires an explicit operator action.")

    result = TrainingResult(
        model=final_model,
        model_config=model_config,
        feature_schema=feature_schema.model_dump(),
        metrics=metrics,
        manifest=manifest,
        dataset_quality=quality,
        model_card=model_card,
        version=model_version,
        rounds=selected_rounds,
        gates=gates,
        passes_gates=passes_gates,
        evaluation_predictions=pd.concat([*fold_predictions, test_predictions], ignore_index=True),
        messages=messages,
    )

    if save_bundle:
        manager = artifacts or ArtifactManager()
        result.bundle_path = manager.save_bundle(
            version=model_version,
            model=final_model,
            feature_schema=result.feature_schema,
            model_config=model_config,
            metrics=metrics,
            training_manifest=manifest,
            dataset_quality=quality,
            model_card=model_card,
            evaluation_predictions=result.evaluation_predictions,
        )
        # Re-read the manifest so artifact checksums are included in the result.
        try:
            result.manifest = json.loads((result.bundle_path / "training_manifest.json").read_text("utf-8"))
        except (OSError, ValueError):  # pragma: no cover - defensive
            pass

    if track_mlflow is None:
        track_mlflow = config.mlflow_enabled
    if track_mlflow:
        result.messages.extend(
            _log_to_mlflow(result, config, test_outcome, outcomes, dataset_quality)
        )

    return result


# --------------------------------------------------------------------------- #
# Fold training
# --------------------------------------------------------------------------- #
def _fit_and_evaluate_fold(
    xgb,
    *,
    x: pd.DataFrame,
    y: np.ndarray,
    frame: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    config: TrainingConfig,
    fold_name: str,
    kind: str,
    validation_start,
    validation_end,
    early_stopping: bool,
    fixed_rounds: int | None = None,
) -> tuple[FoldOutcome, pd.DataFrame]:
    x_train = x.iloc[train_index]
    y_train = y[train_index]
    x_validation = x.iloc[validation_index]
    y_validation = y[validation_index]

    model = _new_model(
        xgb,
        config,
        rounds=fixed_rounds or int(config.hyperparameters.get("n_estimators", 600)),
        early_stopping=early_stopping,
    )
    if early_stopping:
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            verbose=False,
        )
        best_iteration = getattr(model, "best_iteration", None)
        rounds = int(best_iteration) + 1 if best_iteration is not None else model.n_estimators
    else:
        model.fit(x_train, y_train, verbose=False)
        rounds = int(fixed_rounds or model.n_estimators)

    predictions = _prediction_frame(model, frame, validation_index, x_validation)
    report = evaluate_predictions(predictions)
    outcome = FoldOutcome(
        fold=fold_name,
        kind=kind,
        rounds=int(rounds),
        metrics=report,
        train_rows=int(len(train_index)),
        validation_rows=int(len(validation_index)),
        validation_start=_iso(validation_start),
        validation_end=_iso(validation_end),
    )
    return outcome, predictions


def _new_model(xgb, config: TrainingConfig, *, rounds: int, early_stopping: bool):
    hyperparameters = dict(config.hyperparameters)
    hyperparameters.pop("n_estimators", None)
    early_rounds = hyperparameters.pop("early_stopping_rounds", None)
    keyword_arguments: dict[str, Any] = {
        "objective": config.objective,
        "quantile_alpha": list(config.quantiles),
        "tree_method": config.tree_method,
        "random_state": config.random_state,
        "enable_categorical": config.enable_categorical,
        "max_cat_to_onehot": config.max_cat_to_onehot,
        "n_estimators": int(rounds),
        "n_jobs": int(os.cpu_count() or 0) or None,
        **hyperparameters,
    }
    if early_stopping and early_rounds:
        keyword_arguments["early_stopping_rounds"] = int(early_rounds)
        keyword_arguments["eval_metric"] = "quantile"
    return xgb.XGBRegressor(**keyword_arguments)


def _prediction_frame(model, frame: pd.DataFrame, index: np.ndarray, x_validation: pd.DataFrame) -> pd.DataFrame:
    raw = np.asarray(model.predict(x_validation), dtype="float64")
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if raw.shape[1] < 3:
        raise ValueError(
            f"Expected three quantile outputs (P10/P50/P90), received {raw.shape[1]}."
        )
    output = frame.iloc[index].copy().reset_index(drop=True)
    output["p10"] = raw[:, 0]
    output["p50"] = raw[:, 1]
    output["p90"] = raw[:, 2]
    return output


# --------------------------------------------------------------------------- #
# Bundle documents
# --------------------------------------------------------------------------- #
def _model_config_document(
    config: TrainingConfig,
    model,
    version: str,
    selected_rounds: int,
) -> dict[str, Any]:
    hyperparameters = {key: value for key, value in config.hyperparameters.items()}
    return {
        "model_name": config.model_name,
        "model_version": version,
        "objective": config.objective,
        "quantile_alpha": list(config.quantiles),
        "tree_method": config.tree_method,
        "forecast_horizon_days": config.horizon_days,
        "random_state": config.random_state,
        "enable_categorical": config.enable_categorical,
        "max_cat_to_onehot": config.max_cat_to_onehot,
        "target": config.target,
        "cost_basis": config.cost_basis,
        "currency": config.currency,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "selected_rounds": selected_rounds,
        "hyperparameters": hyperparameters,
        "trained_at": utc_now().isoformat(),
        "model_params": _model_params(model),
    }


def _model_params(model) -> dict[str, Any]:
    try:
        raw = model.get_params()
    except Exception:  # pragma: no cover - defensive
        return {}
    clean: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = list(value)
    return clean


def _training_manifest(
    *,
    model_version: str,
    config: TrainingConfig,
    quality: dict[str, Any],
    frame: pd.DataFrame,
    supervised: SupervisedBuildResult,
) -> dict[str, Any]:
    providers = quality.get("providers") or sorted(
        frame["provider"].dropna().astype(str).unique().tolist()
    )
    return {
        "model_version": model_version,
        "dataset_fingerprint": quality.get("dataset_fingerprint"),
        "dataset_start": quality.get("history_start"),
        "dataset_end": quality.get("history_end"),
        "trained_at": utc_now().isoformat(),
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
        "random_seed": config.random_state,
        "providers": list(providers),
        "currency": config.currency,
        "cost_basis": config.cost_basis,
        "feature_schema_version": supervised.schema.schema_version,
        "rows": int(len(frame)),
        "series_count": int(frame["series_id"].nunique()) if "series_id" in frame.columns else 0,
        "horizon_days": config.horizon_days,
        "target": config.target,
        "python_version": platform.python_version(),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("xgboost", "pandas", "numpy", "scikit-learn", "pyarrow", "shap", "mlflow"):
        try:
            module = __import__(name.replace("-", "_"))
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # pragma: no cover - optional dependency
            continue
    return versions


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return None
    commit = completed.stdout.strip()
    return commit or None


def build_model_card(
    *,
    version: str,
    config: TrainingConfig,
    metrics: dict[str, Any],
    quality: dict[str, Any],
    gates: list[dict[str, Any]],
    rounds: int,
) -> str:
    final = metrics.get("final_test", {})
    comparison = metrics.get("baseline_comparison", {})

    def number(value: Any, digits: int = 4) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    fold_rows = "\n".join(
        f"| {fold.get('fold')} | {number(fold.get('wape'))} | {number(fold.get('mae'), 2)} | "
        f"{number(fold.get('interval_coverage'), 3)} | {fold.get('rounds')} |"
        for fold in metrics.get("folds", [])
    )
    gate_rows = "\n".join(
        f"| {gate['name']} | {'PASS' if gate['passed'] else 'FAIL'} | {gate['detail']} |"
        for gate in gates
    )
    return f"""# Model card - {config.model_name} ({version})

## Purpose

Forecast daily cloud spending for the next {config.horizon_days} days with an
estimated P10/P50/P90 interval. The model supports FinOps planning; it does not
modify cloud resources and is not a billing statement.

## Model architecture

- One global XGBoost multi-quantile regressor (`{config.objective}`) trained
  across eligible provider/account/service series.
- Quantiles: P10, P50, P90.
- Tree method: `{config.tree_method}`; {rounds} boosting rounds selected from
  the validation folds.
- Direct multi-horizon design: horizons 1..{config.horizon_days} are predicted
  independently, without recursive lag substitution.

## Training data

- Currency: {config.currency}; cost basis: {config.cost_basis}; target:
  `{config.target}`.
- History: {quality.get('history_start', 'n/a')} to {quality.get('history_end', 'n/a')}
  ({quality.get('history_days', 'n/a')} calendar days).
- Eligible series: {quality.get('eligible_series_count', 'n/a')} of
  {quality.get('series_count', 'n/a')}; observed days:
  {quality.get('eligible_records', 'n/a')}.
- Missing days remain missing and are not treated as zero-cost days.

## Evaluation methodology

Chronological rolling-origin backtesting: {len(metrics.get('folds', []))} validation
folds, each simulating a {config.horizon_days}-day forecast, plus an untouched
final test window. Metrics are out-of-sample only.

| Fold | WAPE | MAE | P10-P90 coverage | Rounds |
|---|---|---|---|---|
{fold_rows}

Final test WAPE: {number(final.get('wape'))}; MAE: {number(final.get('mae'), 2)};
RMSE: {number(final.get('rmse'), 2)}; pinball loss (mean):
{number(final.get('pinball_mean'))}; interval coverage:
{number(final.get('interval_coverage'), 3)}; interval width:
{number(final.get('interval_width'), 2)}; quantile crossing rate:
{number(final.get('quantile_crossing_rate'), 5)}.

Baseline comparison (final test):
- Trailing 7-day average WAPE: {number(comparison.get('wape_trailing_7d_average'))}
- Seasonal-naive weekly WAPE: {number(comparison.get('wape_seasonal_naive_weekly'))}
- Improvement versus trailing average: {number(comparison.get('wape_improvement_vs_trailing_7d_average'), 3)}
- Pinball ratio versus best baseline: {number(comparison.get('pinball_ratio_vs_best_baseline'), 3)}

## Promotion gates

| Gate | Result | Detail |
|---|---|---|
{gate_rows}

## Known limitations

- Billing exports are delayed and revised; recency of the latest complete day
  directly affects forecast quality.
- The model is trained per currency and cost basis and must not be applied to a
  different cohort.
- Sparse series, missing days and non-recurring purchases increase uncertainty.
- Quantile intervals are empirically calibrated on historical data and can be
  wrong during unprecedented events.
- No CPU/memory telemetry is used; forecasts cannot explain *why* a cost
  changed, only that it changed.
- Predictions must not be used as the sole basis for contractual or financial
  commitments.
"""


# --------------------------------------------------------------------------- #
# Experiment tracking
# --------------------------------------------------------------------------- #
def _log_to_mlflow(
    result: TrainingResult,
    config: TrainingConfig,
    test_outcome: FoldOutcome,
    outcomes: list[FoldOutcome],
    dataset_quality: dict[str, Any],
) -> list[str]:
    try:
        import mlflow
    except Exception:  # pragma: no cover - optional dependency
        return ["MLflow is not installed; experiment tracking was skipped."]

    settings = get_settings()
    tracking_uri = settings.mlflow_tracking_uri or (get_settings().project_root / "mlruns").as_uri()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        with mlflow.start_run(run_name=result.version):
            mlflow.log_params(
                {
                    "objective": config.objective,
                    "quantiles": ",".join(str(value) for value in config.quantiles),
                    "horizon_days": config.horizon_days,
                    "selected_rounds": result.rounds,
                    "currency": config.currency,
                    "cost_basis": config.cost_basis,
                    "dataset_fingerprint": str(dataset_quality.get("dataset_fingerprint")),
                }
            )
            for outcome in outcomes:
                mlflow.log_metrics(
                    {
                        f"{outcome.fold}_wape": _finite(outcome.metrics.get("wape")),
                        f"{outcome.fold}_mae": _finite(outcome.metrics.get("mae")),
                        f"{outcome.fold}_coverage": _finite(outcome.metrics.get("interval_coverage")),
                    }
                )
            mlflow.log_metrics(
                {
                    "final_test_wape": _finite(test_outcome.metrics.get("wape")),
                    "final_test_mae": _finite(test_outcome.metrics.get("mae")),
                    "final_test_pinball": _finite(test_outcome.metrics.get("pinball_mean")),
                    "final_test_coverage": _finite(test_outcome.metrics.get("interval_coverage")),
                }
            )
            if result.bundle_path is not None:
                mlflow.log_artifacts(str(result.bundle_path))
        return [f"MLflow run logged to {tracking_uri}."]
    except Exception as exc:  # pragma: no cover - tracking must never break training
        return [f"MLflow logging failed ({type(exc).__name__}); training results are unaffected."]


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).date().isoformat()
