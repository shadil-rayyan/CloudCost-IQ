"""Currency handling.

CloudCost IQ **never** converts currencies implicitly. Records keep their
original reporting currency and every total is grouped by currency. Combining
different currencies requires an explicit, documented, date-aware FX source,
which is out of scope for the MVP; this module exists to make that rule
enforceable rather than accidental.
"""

from __future__ import annotations

import pandas as pd

from app.core.constants import normalize_currency
from app.core.exceptions import CurrencyMismatchError


def distinct_currencies(frame: pd.DataFrame) -> list[str]:
    if frame is None or frame.empty or "BillingCurrency" not in frame.columns:
        return []
    values = frame["BillingCurrency"].dropna().astype(str).str.upper().unique().tolist()
    return sorted(value for value in values if value)


def single_currency(frame: pd.DataFrame, *, required: bool = True) -> str | None:
    currencies = distinct_currencies(frame)
    if len(currencies) == 1:
        return currencies[0]
    if not currencies:
        if required:
            raise CurrencyMismatchError("No billing currency is present in the selected data")
        return None
    raise CurrencyMismatchError(
        "Selected data contains multiple currencies "
        f"({', '.join(currencies)}). Filter to one currency before combining totals."
    )


def assert_single_currency(frame: pd.DataFrame) -> str:
    currency = single_currency(frame, required=True)
    assert currency is not None
    return currency


def currency_cohorts(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split data into independent single-currency cohorts."""
    if frame is None or frame.empty:
        return {}
    if "BillingCurrency" not in frame.columns:
        raise CurrencyMismatchError("Billing records have no currency column")
    cohorts: dict[str, pd.DataFrame] = {}
    for currency in distinct_currencies(frame):
        cohorts[currency] = frame.loc[frame["BillingCurrency"].astype(str).str.upper() == currency].copy()
    return cohorts


def is_compatible_reference_currency(currency: str, reference: str) -> bool:
    return (currency or "").upper() == (reference or "").upper()


def normalize_currency_code(value: object) -> str | None:
    return normalize_currency(value)
