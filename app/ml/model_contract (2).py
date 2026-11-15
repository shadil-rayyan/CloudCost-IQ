"""Model artifact contract.

A model is only usable if its bundle is complete, self-consistent and passes
the documented validation gates. The contract is deliberately strict: the
dashboard never loads an unexplained pickle or a bundle with missing metadata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_ml_config
from app.core.exceptions import ModelContractError

BUNDLE_FILES = (
    "model.ubj",
    "feature_schema.json",
    "model_config.json",
    "metrics.json",
    "training_manifest.json",
    "dataset_quality.json",
    "model_card.md",
)

REQUIRED_CONFIG_FIELDS = (
    "model_name",
    "model_version",
    "objective",
    "quantile_alpha",
    "forecast_horizon_days",
    "target",
    "cost_basis",
    "currency",
    "feature_schema_version",
)


class FeatureSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = "1.0"
    numeric_features: list[str] = Field(default_factory=list)
    categorical_features: list[str] = Field(default_factory=list)
    category_mappings: dict[str, list[str]] = Field(default_factory=dict)
    all_features: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_name: str
    model_version: str
    objective: str = "reg:quantileerror"
    quantile_alpha: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    tree_method: str = "hist"
    forecast_horizon_days: int = 30
    random_state: int = 42
    target: str = "daily_billed_cost"
    cost_basis: str = "billed"
    currency: str = "USD"
    feature_schema_version: str = "1.0"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    enable_categorical: bool = True


class MetricsDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    final_test: dict[str, Any] = Field(default_factory=dict)
    folds: list[dict[str, Any]] = Field(default_factory=list)
    cv_summary: dict[str, Any] = Field(default_factory=dict)
    baseline_comparison: dict[str, Any] = Field(default_factory=dict)


class TrainingManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_version: str
    dataset_fingerprint: str | None = None
    dataset_start: str | None = None
    dataset_end: str | None = None
    trained_at: str | None = None
    git_commit: str | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)
    random_seed: int = 42
    providers: list[str] = Field(default_factory=list)
    currency: str = "USD"
    cost_basis: str = "billed"
    feature_schema_version: str = "1.0"
    rows: int = 0
    series_count: int = 0
    artifact_checksums: dict[str, str] = Field(default_factory=dict)


@dataclass(slots=True)
class ContractCheck:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(slots=True)
class BundleValidation:
    version: str
    path: str
    checks: list[ContractCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks) and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "path": self.path,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "errors": self.errors,
        }


def read_json(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ModelContractError(f"Unable to read model metadata file {Path(path).name}") from exc
    if not isinstance(loaded, dict):
        raise ModelContractError(f"Model metadata file {Path(path).name} must contain an object")
    return loaded


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundle(
    path: Path | str,
    *,
    include_inference_smoke_test: bool = True,
    expected_horizon: int | None = None,
) -> BundleValidation:
    """Check every gate required before a bundle may be promoted or served."""
    path = Path(path)
    version = path.name
    validation = BundleValidation(version=version, path=str(path))

    def check(name: str, passed: bool, detail: str = "") -> None:
        validation.checks.append(ContractCheck(name=name, passed=passed, detail=detail))

    if not path.exists():
        validation.errors.append("Artifact directory does not exist")
        return validation

    missing = [name for name in BUNDLE_FILES if not (path / name).exists()]
    check("artifact_files_present", not missing, "Missing: " + ", ".join(missing) if missing else "All files present")
    if missing:
        validation.errors.append("Bundle is incomplete")
        return validation

    ml_config = get_ml_config()
    gates = ml_config.get("evaluation", {})

    try:
        config = ModelConfig.model_validate(read_json(path / "model_config.json"))
    except (ModelContractError, ValueError) as exc:
        validation.errors.append(str(exc))
        return validation
    try:
        schema = FeatureSchema.model_validate(read_json(path / "feature_schema.json"))
        metrics = MetricsDocument.model_validate(read_json(path / "metrics.json"))
        manifest = TrainingManifest.model_validate(read_json(path / "training_manifest.json"))
        quality = read_json(path / "dataset_quality.json")
    except (ModelContractError, ValueError) as exc:
        validation.errors.append(str(exc))
        return validation

    check("model_version_recorded", bool(config.model_version), config.model_version)
    check(
        "required_config_fields",
        all(getattr(config, field, None) is not None for field in REQUIRED_CONFIG_FIELDS),
    )
    check(
        "forecast_horizon_valid",
        config.forecast_horizon_days > 0
        and (expected_horizon is None or config.forecast_horizon_days == expected_horizon),
        f"horizon={config.forecast_horizon_days}",
    )
    check(
        "feature_schema_compatible",
        schema.schema_version == config.feature_schema_version and bool(schema.all_features),
        f"schema={schema.schema_version}, features={len(schema.all_features)}",
    )
    check(
        "dataset_fingerprint_present",
        bool(manifest.dataset_fingerprint),
        str(manifest.dataset_fingerprint),
    )
    check(
        "dataset_quality_present",
        bool(quality) and bool(quality.get("series_count", quality.get("series"))),
        f"series={quality.get('series_count', quality.get('series'))}",
    )
    check(
        "evaluation_results_present",
        bool(metrics.final_test) and bool(metrics.folds),
        f"folds={len(metrics.folds)}",
    )
    check(
        "baseline_comparison_present",
        bool(metrics.baseline_comparison),
        ", ".join(metrics.baseline_comparison.keys()) if metrics.baseline_comparison else "missing",
    )
    crossing_rate = float(metrics.final_test.get("quantile_crossing_rate", 0.0) or 0.0)
    check(
        "quantile_crossing_acceptable",
        crossing_rate <= float(gates.get("max_quantile_crossing_rate", 0.02)),
        f"crossing_rate={crossing_rate:.4f}",
    )
    model_card = (path / "model_card.md").read_text(encoding="utf-8", errors="replace").strip()
    check("known_limitations_documented", len(model_card) > 200 and "limitation" in model_card.lower())
    check(
        "training_manifest_consistent",
        manifest.model_version == config.model_version
        and manifest.feature_schema_version == schema.schema_version,
        manifest.model_version,
    )

    checksums_path = path / "checksums.json"
    if checksums_path.exists():
        checksums = read_json(checksums_path)
        mismatched = [
            name
            for name, digest in checksums.items()
            if (path / name).exists() and sha256(path / name) != digest
        ]
        check("artifact_checksums_valid", not mismatched, ", ".join(mismatched) if mismatched else "ok")
    else:
        check("artifact_checksums_valid", False, "checksums.json missing")

    if include_inference_smoke_test:
        check("inference_smoke_test", *_smoke_test(path, schema, config))

    if not validation.passed:
        validation.errors.append("One or more validation gates failed")
    return validation


def _smoke_test(path: Path, schema: FeatureSchema, config: ModelConfig) -> tuple[bool, str]:
    try:
        import numpy as np
        import pandas as pd
        import xgboost as xgb

        model = xgb.XGBRegressor()
        model.load_model(str(path / "model.ubj"))
        row = {feature: 1.0 for feature in schema.numeric_features}
        frame = pd.DataFrame([row])
        for feature in schema.categorical_features:
            categories = schema.category_mappings.get(feature) or []
            frame[feature] = pd.Categorical([categories[0] if categories else None], categories=categories or None)
        ordered = frame[schema.all_features]
        prediction = np.asarray(model.predict(ordered))
        expected = len(config.quantile_alpha)
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)
        passed = prediction.shape == (1, expected) and np.isfinite(prediction).all()
        return passed, f"shape={prediction.shape}"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"smoke test failed: {type(exc).__name__}"


def validate_forecast_frame(frame, feature_schema: FeatureSchema) -> None:
    """Ensure inference inputs exactly match the trained feature schema."""
    missing = [feature for feature in feature_schema.all_features if feature not in frame.columns]
    if missing:
        raise ModelContractError(
            "Inference data is missing required model features: " + ", ".join(missing[:10])
        )
