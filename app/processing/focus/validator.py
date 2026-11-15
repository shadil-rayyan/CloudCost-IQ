"""Validation for FOCUS-compatible datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.processing.focus.schema import REQUIRED_COLUMNS
from app.processing.focus.versions import is_supported


@dataclass(slots=True)
class FocusValidationResult:
    version: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "valid": self.valid, "errors": self.errors, "warnings": self.warnings}


def validate_focus_frame(frame: pd.DataFrame, *, version: str = "1.4") -> FocusValidationResult:
    result = FocusValidationResult(version=version, valid=True)
    if not is_supported(version):
        result.errors.append(f"Unsupported FOCUS version: {version}")
        result.valid = False
        return result
    if frame is None or frame.empty:
        result.errors.append("FOCUS dataset is empty.")
        result.valid = False
        return result
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        result.errors.append("Missing required FOCUS columns: " + ", ".join(missing))
        result.valid = False
    # required non-null checks
    for col in ("ServiceProviderName", "BillingCurrency", "BilledCost", "ChargePeriodStart"):
        if col in frame.columns:
            nulls = int(pd.isna(frame[col]).sum())
            if nulls:
                result.warnings.append(f"{col} has {nulls} null values.")
    if "BilledCost" in frame.columns:
        non_numeric = pd.to_numeric(frame["BilledCost"], errors="coerce").isna() & frame["BilledCost"].notna()
        if int(non_numeric.sum()) > 0:
            result.errors.append("BilledCost contains non-numeric values.")
            result.valid = False
    return result
