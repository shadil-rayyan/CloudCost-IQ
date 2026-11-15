"""Build daily, multi-series cost panels from normalized billing records.

This module implements the dataset eligibility gates from the project
specification. Missing days stay missing (``observed=False``, ``cost=NaN``);
they are never turned into zero-cost days.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_ml_config
from app.utils.dates import date_span_days

SERIES_KEYS: tuple[str, ...] = (
    "ServiceProviderName",
    "BillingAccountId",
    "ServiceName",
    "BillingCurrency",
    "cost_basis",
)


@dataclass(slots=True)
class DatasetBuildConfig:
    currency: str = "USD"
    cost_basis: str = "billed"
    horizon_days: int = 30
    min_history_days: int = 180
    preferred_history_days: int = 365
    min_observations: int = 120
    max_gap_days: int = 7
    min_coverage_ratio: float = 0.25
    validation_folds: int = 3
    min_train_days: int = 90
    series_keys: tuple[str, ...] = SERIES_KEYS

    @classmethod
    def from_ml_config(cls, *, currency: str | None = None, cost_basis: str | None = None) -> DatasetBuildConfig:
        config = get_ml_config()
        training = config.get("training", {})
        model = config.get("model", {})
        return cls(
            currency=(currency or training.get("currency") or "USD").upper(),
            cost_basis=cost_basis or training.get("cost_basis") or "billed",
            horizon_days=int(model.get("forecast_horizon_days", 30)),
            min_history_days=int(training.get("min_history_days", 180)),
            preferred_history_days=int(training.get("preferred_history_days", 365)),
            min_observations=int(training.get("min_observations", 120)),
            max_gap_days=int(training.get("max_gap_days", 7)),
            validation_folds=int(training.get("validation_folds", 3)),
            min_train_days=int(training.get("min_train_days", 90)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "BillingCurrency": self.currency,
            "cost_basis": self.cost_basis,
            "horizon_days": self.horizon_days,
            "min_history_days": self.min_history_days,
            "preferred_history_days": self.preferred_history_days,
            "min_observations": self.min_observations,
            "max_gap_days": self.max_gap_days,
            "min_coverage_ratio": self.min_coverage_ratio,
            "validation_folds": self.validation_folds,
            "min_train_days": self.min_train_days,
            "series_keys": list(self.series_keys),
        }


@dataclass(slots=True)
class DatasetBuildResult:
    panel: pd.DataFrame
    quality: dict[str, Any]
    config: DatasetBuildConfig
    eligible_series: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    @property
    def is_sufficient(self) -> bool:
        return bool(self.quality.get("sufficient"))

    @property
    def eligible_panel(self) -> pd.DataFrame:
        if self.panel.empty or not self.eligible_series:
            return self.panel.iloc[0:0].copy()
        return self.panel.loc[self.panel["series_id"].isin(self.eligible_series)].copy()

    def insufficient_reason(self) -> str:
        if self.is_sufficient:
            return ""
        return " ".join(self.messages) or "The dataset does not meet the minimum training requirements."


def series_fingerprint(series_id: str, frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(series_id.encode("utf-8"))
    for date, cost in frame.loc[frame["observed"], ["date", "cost"]].itertuples(index=False):
        cost_text = "nan" if pd.isna(cost) else f"{float(cost):.6f}"
        digest.update(f"|{pd.Timestamp(date).date().isoformat()}:{cost_text}".encode())
    return digest.hexdigest()


def _series_id(values: tuple[Any, ...]) -> str:
    text = "|".join("" if value is None or pd.isna(value) else str(value) for value in values)
    return "s-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def build_daily_panel(frame: pd.DataFrame, config: DatasetBuildConfig) -> DatasetBuildResult:
    """Aggregate records into an eligible daily series panel.

    ``panel`` has one row per series per calendar day with:
    ``series_id, provider, billing_account_id, service_name, currency, cost_basis,
    date, cost, records, observed``.
    """
    messages: list[str] = []
    if frame is None or frame.empty:
        quality = _empty_quality(config, "No billing records were available.")
        return DatasetBuildResult(pd.DataFrame(), quality, config, [], [quality["reason"]])

    required = {"ServiceProviderName", "BillingAccountId", "ServiceName", "BillingCurrency", "cost_basis",
                "usage_start", "BilledCost"}
    missing = sorted(required - set(frame.columns))
    if missing:
        quality = _empty_quality(config, "Missing required columns: " + ", ".join(missing))
        return DatasetBuildResult(pd.DataFrame(), quality, config, [], [quality["reason"]])

    working = frame.copy()
    working["BillingCurrency"] = working["BillingCurrency"].astype("string").str.upper()
    cohort = working.loc[
        (working["BillingCurrency"] == config.currency.upper())
        & (working["cost_basis"] == config.cost_basis)
    ].copy()
    if cohort.empty:
        quality = _empty_quality(
            config,
            f"No records match the model cohort (currency={config.currency}, "
            f"cost basis={config.cost_basis}).",
        )
        return DatasetBuildResult(pd.DataFrame(), quality, config, [], [quality["reason"]])

    cohort["billing_date"] = pd.to_datetime(
        cohort["ChargePeriodStart"].where(
            cohort["ChargePeriodStart"].notna(), cohort["usage_start"]
        ),
        errors="coerce",
    ).dt.normalize()
    invalid_dates = int(cohort["billing_date"].isna().sum())
    cohort = cohort.loc[cohort["billing_date"].notna()].copy()
    cohort["BilledCost"] = pd.to_numeric(cohort["BilledCost"], errors="coerce")
    invalid_costs = int(cohort["BilledCost"].isna().sum())
    cohort = cohort.loc[cohort["BilledCost"].notna()].copy()

    keys = [key for key in config.series_keys if key in cohort.columns]
    hourly = (
        cohort.groupby([*keys, "billing_date"], dropna=False)
        .agg(cost=("BilledCost", "sum"), records=("BilledCost", "size"))
        .reset_index()
    )

    panels: list[pd.DataFrame] = []
    series_quality: list[dict[str, Any]] = []
    for key_values, group in hourly.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        group = group.sort_values("billing_date")
        first = group["billing_date"].min()
        last = group["billing_date"].max()
        calendar = pd.date_range(first, last, freq="D")
        dense = pd.DataFrame({"date": calendar})
        dense = dense.merge(
            group.rename(columns={"billing_date": "date"}), on="date", how="left"
        )
        dense["observed"] = dense["cost"].notna()
        dense["records"] = dense["records"].fillna(0).astype(int)
        dense["series_id"] = _series_id(key_values)
        for key, value in zip(keys, key_values, strict=False):
            if key not in {"BillingCurrency", "cost_basis"}:
                dense[key] = value
        # Series identity always carries currency and cost basis.
        dense["BillingCurrency"] = config.currency.upper()
        dense["cost_basis"] = config.cost_basis
        dense = dense.sort_values("date").reset_index(drop=True)

        observed_days = int(dense["observed"].sum())
        calendar_days = len(dense)
        span_days = date_span_days(first.date(), last.date())
        coverage_ratio = observed_days / calendar_days if calendar_days else 0.0
        gaps = _longest_gap(dense["observed"].to_numpy())
        series_quality.append(
            {
                "series_id": dense["series_id"].iloc[0],
                "ServiceProviderName": _series_value(dense, "ServiceProviderName"),
                "BillingAccountId": _series_value(dense, "BillingAccountId"),
                "ServiceName": _series_value(dense, "ServiceName"),
                "first_date": first.date().isoformat(),
                "last_date": last.date().isoformat(),
                "span_days": span_days,
                "observed_days": observed_days,
                "calendar_days": calendar_days,
                "coverage_ratio": float(coverage_ratio),
                "longest_gap_days": gaps,
                "sufficient": False,
            }
        )
        panels.append(dense)

    if not panels:
        quality = _empty_quality(config, "No eligible daily series could be constructed.")
        return DatasetBuildResult(pd.DataFrame(), quality, config, [], [quality["reason"]])

    panel = pd.concat(panels, ignore_index=True)
    panel["cost"] = panel["cost"].astype("float64")

    overall_start = panel["date"].min().date()
    overall_end = panel["date"].max().date()
    overall_span = date_span_days(overall_start, overall_end)

    eligible: list[str] = []
    for item in series_quality:
        reasons: list[str] = []
        if item["observed_days"] < config.min_observations:
            reasons.append(f"only {item['observed_days']} observed days (need {config.min_observations})")
        if item["span_days"] < config.min_history_days:
            reasons.append(f"only {item['span_days']} calendar days (need {config.min_history_days})")
        if item["coverage_ratio"] < config.min_coverage_ratio:
            reasons.append(
                f"coverage {item['coverage_ratio']:.0%} below {config.min_coverage_ratio:.0%}"
            )
        item["sufficient"] = not reasons
        item["rejection_reasons"] = reasons
        if item["sufficient"]:
            eligible.append(item["series_id"])

    # Minimum number of days required for the configured rolling-origin design.
    required_days = config.min_train_days + (config.validation_folds + 1) * config.horizon_days
    windows_ok = overall_span >= required_days
    history_ok = overall_span >= config.min_history_days
    eligible_ok = bool(eligible)

    if not history_ok:
        messages.append(
            f"History covers {overall_span} days; at least {config.min_history_days} are required."
        )
    if overall_span < config.preferred_history_days:
        messages.append(
            f"History covers {overall_span} days; {config.preferred_history_days}+ days are preferred "
            "for a reliable 30-day forecast."
        )
    if not eligible_ok:
        messages.append(
            "No series meets the minimum observation, history and coverage requirements."
        )
    if not windows_ok:
        messages.append(
            f"{required_days} days of history are needed for {config.validation_folds} rolling-origin "
            f"folds plus an untouched {config.horizon_days}-day test window; only {overall_span} are available."
        )

    sufficient = history_ok and eligible_ok and windows_ok
    if sufficient:
        messages.insert(
            0,
            f"Dataset is eligible: {len(eligible)} series, {overall_span} calendar days "
            f"({overall_start.isoformat()} to {overall_end.isoformat()}).",
        )

    eligible_panel = panel.loc[panel["series_id"].isin(eligible)]
    observed = eligible_panel.loc[eligible_panel["observed"]]
    fingerprint = _panel_fingerprint(observed)
    missing_days = int(len(eligible_panel) - len(observed))
    zero_cost_days = int((observed["cost"] == 0).sum())

    quality = {
        "schema_version": "1.0",
        "BillingCurrency": config.currency.upper(),
        "cost_basis": config.cost_basis,
        "target": "daily_billed_cost",
        "total_records_input": len(cohort),
        "eligible_records": len(observed),
        "rejected_records": int(len(cohort) - len(observed)),
        "invalid_records": int(invalid_dates + invalid_costs),
        "series_count": len(series_quality),
        "eligible_series_count": len(eligible),
        "rejected_series_count": int(len(series_quality) - len(eligible)),
        "history_start": overall_start.isoformat(),
        "history_end": overall_end.isoformat(),
        "history_days": overall_span,
        "preferred_history_days": config.preferred_history_days,
        "missing_periods": missing_days,
        "zero_cost_days": zero_cost_days,
        "candidate_rows": None,
        "coverage": {
            "calendar_days": len(eligible_panel),
            "observed_days": len(observed),
            "coverage_ratio": float(len(observed) / len(eligible_panel)) if len(eligible_panel) else 0.0,
        },
        "providers": sorted(eligible_panel["ServiceProviderName"].dropna().astype(str).unique().tolist()),
        "services": sorted(eligible_panel["ServiceName"].dropna().astype(str).unique().tolist())[:50],
        "series": series_quality,
        "sufficient": bool(sufficient),
        "reason": " ".join(messages),
        "dataset_fingerprint": fingerprint,
        "config": config.to_dict(),
        "notes": [
            "Missing days remain missing (observed=False); they are not zero-cost observations.",
            "Zero-cost days are only those with actual billing records totalling zero.",
            "Series identity is provider + billing account + service + currency + cost basis.",
        ],
    }
    return DatasetBuildResult(panel, quality, config, eligible, messages)


def _panel_fingerprint(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    digest = hashlib.sha256()
    ordered = frame.sort_values(["series_id", "date"])
    for series_id, date, cost in ordered[["series_id", "date", "cost"]].itertuples(index=False):
        digest.update(
            f"{series_id}|{pd.Timestamp(date).date().isoformat()}|{float(cost):.6f}\n".encode()
        )
    return digest.hexdigest()


def _longest_gap(observed: np.ndarray) -> int:
    longest = current = 0
    for flag in observed:
        if flag:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def _series_value(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return str(values.iloc[0])


def _empty_quality(config: DatasetBuildConfig, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "BillingCurrency": config.currency.upper(),
        "cost_basis": config.cost_basis,
        "target": "daily_billed_cost",
        "total_records_input": 0,
        "eligible_records": 0,
        "rejected_records": 0,
        "series_count": 0,
        "eligible_series_count": 0,
        "history_start": None,
        "history_end": None,
        "history_days": 0,
        "missing_periods": 0,
        "coverage": {"calendar_days": 0, "observed_days": 0, "coverage_ratio": 0.0},
        "providers": [],
        "series": [],
        "sufficient": False,
        "reason": reason,
        "dataset_fingerprint": "",
        "config": config.to_dict(),
    }


def select_series(
    frame: pd.DataFrame,
    *,
    provider: str | None = None,
    billing_account_id: str | None = None,
    service_name: str | None = None,
) -> pd.DataFrame:
    """Filter normalized records to a forecast scope."""
    selected = frame
    if provider:
        selected = selected.loc[selected["ServiceProviderName"] == provider]
    if billing_account_id:
        selected = selected.loc[selected["BillingAccountId"] == billing_account_id]
    if service_name:
        selected = selected.loc[selected["ServiceName"] == service_name]
    return selected.copy()
