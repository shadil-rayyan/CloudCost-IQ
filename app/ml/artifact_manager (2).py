"""Versioned model artifact storage (candidates, staging, production)."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import get_app_config, get_ml_config
from app.core.exceptions import ModelContractError, ModelNotAvailableError
from app.ml.model_contract import (
    BUNDLE_FILES,
    FeatureSchema,
    ModelConfig,
    MetricsDocument,
    TrainingManifest,
    read_json,
    sha256,
    validate_bundle,
)
from app.utils.dates import utc_now
from app.utils.files import write_parquet_atomic


@dataclass(slots=True)
class ModelBundle:
    version: str
    path: Path
    model: Any
    feature_schema: FeatureSchema
    model_config: ModelConfig
    metrics: dict[str, Any]
    manifest: dict[str, Any]
    dataset_quality: dict[str, Any]
    model_card: str
    stage: str


@dataclass(slots=True)
class ModelSummary:
    version: str
    stage: str
    path: str
    created_at: str | None
    currency: str | None
    cost_basis: str | None
    horizon_days: int | None
    wape: float | None
    coverage: float | None
    validated: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "stage": self.stage,
            "path": self.path,
            "created_at": self.created_at,
            "currency": self.currency,
            "cost_basis": self.cost_basis,
            "horizon_days": self.horizon_days,
            "wape": self.wape,
            "coverage": self.coverage,
            "validated": self.validated,
        }


class ArtifactManager:
    """Manage model bundles across candidate/staging/production stages."""

    def __init__(self, artifacts_dir: Path | str | None = None) -> None:
        base = Path(artifacts_dir or get_app_config().storage.artifacts_dir)
        config = get_ml_config().get("artifact", {})
        self.candidates_dir = Path(config.get("candidates_dir", base / "candidates"))
        self.staging_dir = Path(config.get("staging_dir", base / "staging"))
        self.production_dir = Path(config.get("production_dir", base / "production"))
        if not self.candidates_dir.is_absolute():
            self.candidates_dir = base / "candidates"
        if not self.staging_dir.is_absolute():
            self.staging_dir = base / "staging"
        if not self.production_dir.is_absolute():
            self.production_dir = base / "production"
        for directory in (self.candidates_dir, self.staging_dir, self.production_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ versions
    @staticmethod
    def new_version(prefix: str = "cciq-xgb-q30") -> str:
        return f"{prefix}-{utc_now().strftime('%Y%m%d-%H%M%S')}"

    def stage_dir(self, stage: str) -> Path:
        mapping = {
            "candidates": self.candidates_dir,
            "staging": self.staging_dir,
            "production": self.production_dir,
        }
        if stage not in mapping:
            raise ModelContractError(f"Unknown model stage '{stage}'")
        return mapping[stage]

    # --------------------------------------------------------------------- save
    def save_bundle(
        self,
        *,
        version: str,
        model,
        feature_schema: dict[str, Any],
        model_config: dict[str, Any],
        metrics: dict[str, Any],
        training_manifest: dict[str, Any],
        dataset_quality: dict[str, Any],
        model_card: str,
        evaluation_predictions=None,
        stage: str = "candidates",
    ) -> Path:
        destination = self.stage_dir(stage) / version
        destination.mkdir(parents=True, exist_ok=True)
        try:
            model.save_model(str(destination / "model.ubj"))
            _write_json(destination / "feature_schema.json", feature_schema)
            _write_json(destination / "model_config.json", model_config)
            _write_json(destination / "metrics.json", metrics)
            _write_json(destination / "training_manifest.json", training_manifest)
            _write_json(destination / "dataset_quality.json", dataset_quality)
            (destination / "model_card.md").write_text(model_card, encoding="utf-8")
            if evaluation_predictions is not None and len(evaluation_predictions):
                write_parquet_atomic(evaluation_predictions, destination / "evaluation_predictions.parquet")
            # training_manifest.json embeds the checksum map, so it cannot
            # include a checksum of itself. Every other bundle file is covered.
            checksums = {
                name: sha256(destination / name)
                for name in BUNDLE_FILES
                if name != "training_manifest.json" and (destination / name).exists()
            }
            if (destination / "evaluation_predictions.parquet").exists():
                checksums["evaluation_predictions.parquet"] = sha256(
                    destination / "evaluation_predictions.parquet"
                )
            _write_json(destination / "checksums.json", checksums)
            _write_json(
                destination / "training_manifest.json",
                {**training_manifest, "artifact_checksums": checksums},
            )
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        return destination

    # --------------------------------------------------------------------- load
    def find_bundle_path(self, version_or_path: str | Path, stage: str = "auto") -> tuple[Path, str]:
        candidate = Path(version_or_path)
        if candidate.exists() and candidate.is_dir():
            return candidate, _stage_of(candidate, self)
        version = str(version_or_path)
        order = ["production", "staging", "candidates"] if stage == "auto" else [stage]
        for name in order:
            path = self.stage_dir(name) / version
            if path.exists():
                return path, name
        raise ModelNotAvailableError(f"Model version '{version}' was not found")

    def load_bundle(self, version_or_path: str | Path, stage: str = "auto") -> ModelBundle:
        path, resolved_stage = self.find_bundle_path(version_or_path, stage)
        validation = validate_bundle(path)
        if not validation.passed:
            failed = [check.name for check in validation.checks if not check.passed]
            raise ModelContractError(
                "Model bundle failed validation gates: " + ", ".join(failed or ["unknown"])
            )
        import xgboost as xgb  # local import keeps dashboard startup fast

        model = xgb.XGBRegressor()
        model.load_model(str(path / "model.ubj"))
        return ModelBundle(
            version=path.name,
            path=path,
            model=model,
            feature_schema=FeatureSchema.model_validate(read_json(path / "feature_schema.json")),
            model_config=ModelConfig.model_validate(read_json(path / "model_config.json")),
            metrics=read_json(path / "metrics.json"),
            manifest=read_json(path / "training_manifest.json"),
            dataset_quality=read_json(path / "dataset_quality.json"),
            model_card=(path / "model_card.md").read_text(encoding="utf-8", errors="replace"),
            stage=resolved_stage,
        )

    # --------------------------------------------------------------------- list
    def list_models(self, stage: str | None = None, *, validate: bool = False) -> list[ModelSummary]:
        stages = [stage] if stage else ["production", "staging", "candidates"]
        summaries: list[ModelSummary] = []
        for stage_name in stages:
            directory = self.stage_dir(stage_name)
            for path in sorted(directory.iterdir(), reverse=True):
                if not path.is_dir():
                    continue
                summary = self._summarise(path, stage_name)
                if validate:
                    summary.validated = validate_bundle(path).passed
                summaries.append(summary)
        return summaries

    def production_version(self) -> str | None:
        versions = [path.name for path in self.production_dir.iterdir() if path.is_dir()]
        return sorted(versions)[-1] if versions else None

    def production_bundle(self) -> ModelBundle | None:
        version = self.production_version()
        if version is None:
            return None
        return self.load_bundle(version, stage="production")

    # ------------------------------------------------------------------ promote
    def promote(
        self,
        version: str,
        *,
        from_stage: str = "candidates",
        to_stage: str = "production",
        archive_existing: bool = False,
    ) -> Path:
        source = self.stage_dir(from_stage) / version
        if not source.exists():
            raise ModelNotAvailableError(f"Model '{version}' is not present in {from_stage}")
        validation = validate_bundle(source)
        if not validation.passed:
            failed = [check.name for check in validation.checks if not check.passed]
            raise ModelContractError(
                "Refusing to promote a model that fails validation gates: "
                + ", ".join(failed or ["unknown"])
            )
        destination = self.stage_dir(to_stage) / version
        if destination.exists():
            raise ModelContractError(f"Model '{version}' already exists in {to_stage}")
        if to_stage == "production" and archive_existing:
            current = self.production_version()
            if current and current != version:
                shutil.move(str(self.production_dir / current), str(self.candidates_dir / current))
        shutil.copytree(source, destination)
        return destination

    # ------------------------------------------------------------------ helpers
    def _summarise(self, path: Path, stage: str) -> ModelSummary:
        created: str | None = None
        currency = None
        basis = None
        horizon = None
        wape = None
        coverage = None
        try:
            config = read_json(path / "model_config.json")
            currency = config.get("currency")
            basis = config.get("cost_basis")
            horizon = config.get("forecast_horizon_days")
            created = config.get("trained_at")
        except ModelContractError:
            pass
        try:
            metrics = read_json(path / "metrics.json")
            final = metrics.get("final_test", {})
            wape = final.get("wape")
            coverage = final.get("interval_coverage")
            created = created or metrics.get("trained_at")
        except ModelContractError:
            pass
        if created is None:
            created = dt.datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()
        return ModelSummary(
            version=path.name,
            stage=stage,
            path=str(path),
            created_at=str(created) if created else None,
            currency=currency,
            cost_basis=basis,
            horizon_days=horizon,
            wape=float(wape) if wape is not None else None,
            coverage=float(coverage) if coverage is not None else None,
        )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _stage_of(path: Path, manager: ArtifactManager) -> str:
    resolved = path.resolve()
    for name, directory in (
        ("production", manager.production_dir),
        ("staging", manager.staging_dir),
        ("candidates", manager.candidates_dir),
    ):
        if directory.resolve() in resolved.parents or resolved == directory.resolve():
            return name
    return "unknown"
