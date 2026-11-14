"""Application configuration.

Configuration lives in two layers:

* ``configs/app_config.yaml`` - behavioural configuration (limits, reporting,
  data-quality thresholds). Loaded into :class:`AppConfig`.
* Environment variables prefixed with ``CLOUDCOST_`` - deployment overrides
  handled by :class:`Settings` (see ``.env.example``).

Nothing here reads billing data; this module only describes where things live
and how the application should behave.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Deployment-level settings read from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="CLOUDCOST_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = PROJECT_ROOT
    app_config_path: Path = PROJECT_ROOT / "configs" / "app_config.yaml"
    ml_config_path: Path = PROJECT_ROOT / "configs" / "ml_config.yaml"

    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Optional path overrides. When unset the YAML values are used.
    data_dir: Path | None = None
    artifacts_dir: Path | None = None
    database_path: Path | None = None

    mlflow_enabled: bool | None = None
    mlflow_tracking_uri: str | None = None

    max_upload_mb: int | None = None

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()


class StorageConfig(BaseModel):
    """Filesystem locations used by the application."""

    model_config = ConfigDict(validate_assignment=True)

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    rejected_dir: Path = Path("data/rejected")
    demo_dir: Path = Path("data/demo")
    metadata_dir: Path = Path("data/metadata")
    artifacts_dir: Path = Path("artifacts/models")
    database_path: Path = Path("data/metadata/cloudcost_iq.db")

    def resolve(self, root: Path) -> None:
        """Make every path absolute relative to ``root``."""
        for name in (
            "data_dir",
            "raw_dir",
            "processed_dir",
            "rejected_dir",
            "demo_dir",
            "metadata_dir",
            "artifacts_dir",
            "database_path",
        ):
            value: Path = getattr(self, name)
            if not value.is_absolute():
                setattr(self, name, (root / value).resolve())

    def ensure(self) -> None:
        for name in (
            "raw_dir",
            "processed_dir",
            "rejected_dir",
            "demo_dir",
            "metadata_dir",
            "artifacts_dir",
        ):
            Path(getattr(self, name)).mkdir(parents=True, exist_ok=True)
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)


class UploadConfig(BaseModel):
    max_file_size_mb: int = 512
    max_uncompressed_size_mb: int = 2048
    max_archive_members: int = 16
    max_rows: int = 5_000_000
    preview_rows: int = 50
    allowed_extensions: list[str] = Field(
        default_factory=lambda: ["csv", "parquet", "gz", "zip"]
    )

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def max_uncompressed_size_bytes(self) -> int:
        return self.max_uncompressed_size_mb * 1024 * 1024


class ReportingConfig(BaseModel):
    default_currency: str = "USD"
    default_cost_basis: str = "billed"
    allow_mixed_currency_totals: bool = False
    fiscal_year_start_month: int = 1

    @field_validator("default_currency")
    @classmethod
    def _currency(cls, value: str) -> str:
        return value.upper()


class BudgetConfig(BaseModel):
    monthly_budget: float | None = None


class QualityConfig(BaseModel):
    min_history_days: int = 180
    preferred_history_days: int = 365
    min_daily_observations: int = 120
    max_gap_days: int = 7
    duplicate_warning_ratio: float = 0.01
    anomaly_zscore_threshold: float = 3.0
    anomaly_lookback_days: int = 28


class AppConfig(BaseModel):
    """Full application configuration."""

    app_name: str = "CloudCost IQ"
    app_version: str = "1.0.0"
    normalized_schema_version: str = "1.0"

    storage: StorageConfig = Field(default_factory=StorageConfig)
    uploads: UploadConfig = Field(default_factory=UploadConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)

    def resolve_paths(self, root: Path) -> None:
        self.storage.resolve(root)

    def ensure_directories(self) -> None:
        self.storage.ensure()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping")
    return loaded


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    """Load the YAML configuration and apply environment overrides."""
    settings = get_settings()
    config = AppConfig.model_validate(_read_yaml(settings.app_config_path))

    if settings.data_dir is not None:
        base = settings.data_dir
        # Rebase all data-relative paths under the override.
        config.storage.raw_dir = base / "raw"
        config.storage.processed_dir = base / "processed"
        config.storage.rejected_dir = base / "rejected"
        config.storage.demo_dir = base / "demo"
        config.storage.metadata_dir = base / "metadata"
        config.storage.database_path = base / "metadata" / "cloudcost_iq.db"
        config.storage.data_dir = base
    if settings.artifacts_dir is not None:
        config.storage.artifacts_dir = settings.artifacts_dir
    if settings.database_path is not None:
        config.storage.database_path = settings.database_path
    if settings.max_upload_mb is not None:
        config.uploads.max_file_size_mb = settings.max_upload_mb

    config.resolve_paths(settings.project_root)
    return config


@lru_cache(maxsize=1)
def get_ml_config() -> dict[str, Any]:
    """Return the ML training configuration as a plain dictionary."""
    settings = get_settings()
    return _read_yaml(settings.ml_config_path)


def reset_config_cache() -> None:
    """Clear cached configuration (used by tests and after edits)."""
    get_settings.cache_clear()
    get_app_config.cache_clear()
    get_ml_config.cache_clear()
