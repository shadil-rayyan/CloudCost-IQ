"""Normalization of provider-specific billing rows into the common schema.

Provider parsers are responsible for *reading* their own columns; this module
owns the shared post-processing so every provider ends up with identical types,
identifiers and derived fields. Unavailable values are preserved as ``None`` -
nothing is imputed here.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import pandas as pd

from app.core.constants import (
    CHARGE_CATEGORIES,
    DATETIME_COLUMNS,
    NORMALIZED_COLUMNS,
    NORMALIZED_SCHEMA_VERSION,
    NUMERIC_COLUMNS,
    classify_service,
    normalize_currency,
)
from app.utils.dates import month_key, to_datetime_utc


def empty_normalized_frame() -> pd.DataFrame:
    """An empty frame with the exact normalized schema and dtypes."""
    data: dict[str, pd.Series] = {}
    for column in NORMALIZED_COLUMNS:
        if column in DATETIME_COLUMNS:
            data[column] = pd.Series(dtype="datetime64[ns]")
        elif column in NUMERIC_COLUMNS:
            data[column] = pd.Series(dtype="float64")
        else:
            data[column] = pd.Series(dtype="object")
    return pd.DataFrame(data)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    if value is pd.NA:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "nat"}:
        return None
    return text


def to_text_series(series: pd.Series) -> pd.Series:
    """Clean a column to strings with ``None`` for blanks."""
    return series.map(_clean_text)


def normalize_tag_payload(value: Any) -> str | None:
    """Serialize provider tags/labels to a stable JSON object string."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return text
        return json.dumps(parsed, sort_keys=True, default=str)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, (list, tuple)):
        collected: dict[str, Any] = {}
        for entry in value:
            if isinstance(entry, dict):
                key = entry.get("key") or entry.get("name")
                if key is not None:
                    collected[str(key)] = entry.get("value")
        if collected:
            return json.dumps(collected, sort_keys=True, default=str)
        return json.dumps(list(value), sort_keys=True, default=str)
    return json.dumps(value, default=str)


def _raw_row_json(row: pd.Series) -> str:
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if value is None or value is pd.NA:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, (pd.Timestamp,)):
            value = value.isoformat()
        payload[str(key)] = value
    return json.dumps(payload, default=str, ensure_ascii=False)


def serialize_raw_rows(raw_frame: pd.DataFrame) -> pd.Series:
    """Preserve original provider fields as one JSON document per record."""
    records = raw_frame.to_dict(orient="records")
    return pd.Series([json.dumps(record, default=str, ensure_ascii=False) for record in records])


def _build_record_ids(df: pd.DataFrame, import_batch_id: str) -> pd.Series:
    key_columns = [
        "provider",
        "billing_account_id",
        "service_name",
        "sku_id",
        "resource_id",
        "usage_start",
        "cost_amount",
        "currency",
        "cost_basis",
        "charge_type",
        "quantity",
    ]
    combined = pd.Series([""] * len(df), index=df.index, dtype="object")
    for column in key_columns:
        if column not in df.columns:
            continue
        part = df[column].astype("string").fillna("~")
        combined = combined + "|" + part
    combined = combined + "|" + pd.Series(range(len(df)), index=df.index).astype(str)
    hashes = combined.map(
        lambda text: hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:20]
    )
    prefix = import_batch_id.replace("-", "")[:10]
    return pd.Series(
        [f"{prefix}-{index:08d}-{digest}" for index, digest in enumerate(hashes)],
        index=df.index,
    )


def finalize_records(
    df: pd.DataFrame,
    *,
    provider: str,
    import_batch_id: str,
    source_file_id: str,
    cost_basis: str,
    schema_version: str = NORMALIZED_SCHEMA_VERSION,
    raw_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Coerce a mapped frame into the canonical normalized record set."""
    if df.empty:
        return empty_normalized_frame()

    frame = df.copy()
    for column in NORMALIZED_COLUMNS:
        if column not in frame.columns:
            frame[column] = None

    frame = frame[list(NORMALIZED_COLUMNS)].copy()

    for column in DATETIME_COLUMNS:
        frame[column] = to_datetime_utc(frame[column])

    # Fill period timestamps from whichever timestamps the provider supplied,
    # without inventing a duration when the provider does not report one.
    frame["charge_period_start"] = frame["charge_period_start"].fillna(frame["usage_start"])
    frame["usage_start"] = frame["usage_start"].fillna(frame["charge_period_start"])
    frame["charge_period_end"] = frame["charge_period_end"].fillna(frame["usage_end"])

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in NORMALIZED_COLUMNS:
        if column in DATETIME_COLUMNS or column in NUMERIC_COLUMNS:
            continue
        frame[column] = to_text_series(frame[column])

    # FOCUS files may contain several providers in one file; the parser maps
    # each row's ProviderName. Other formats have a single provider.
    if provider == "FOCUS":
        frame["provider"] = to_text_series(frame["provider"]).fillna("FOCUS")
    else:
        frame["provider"] = provider
    frame["cost_basis"] = cost_basis
    frame["schema_version"] = schema_version
    frame["source_file_id"] = source_file_id
    frame["import_batch_id"] = import_batch_id

    frame["currency"] = frame["currency"].map(normalize_currency)
    frame["invoice_month"] = frame["invoice_month"].fillna(month_key(frame["charge_period_start"]))

    frame["service_category"] = [
        category or classify_service(service)
        for category, service in zip(frame["service_category"], frame["service_name"], strict=False)
    ]

    frame["charge_category"] = frame["charge_category"].map(_normalize_charge_category)

    frame["tags_json"] = frame["tags_json"].map(normalize_tag_payload)

    if raw_frame is not None and len(raw_frame) == len(frame):
        frame["raw_fields_json"] = serialize_raw_rows(raw_frame.reset_index(drop=True)).values
    else:
        frame["raw_fields_json"] = frame["raw_fields_json"].map(
            lambda value: None if value is None else str(value)
        )

    frame["record_id"] = _build_record_ids(frame, import_batch_id)

    return frame.reset_index(drop=True)


def _normalize_charge_category(value: str | None) -> str:
    if not value:
        return "other"
    needle = value.strip().lower().replace("-", " ").replace("_", " ")
    mapping = {
        "usage": "usage",
        "used": "usage",
        "purchase": "purchase",
        "tax": "tax",
        "credit": "credit",
        "refund": "refund",
        "adjustment": "adjustment",
        "rounding": "adjustment",
        "discount": "discount",
        "fee": "fee",
        "support": "fee",
        "commitment": "commitment",
        "savings plan": "commitment",
        "reservation": "commitment",
    }
    for key, category in mapping.items():
        if key in needle:
            return category
    if needle in CHARGE_CATEGORIES:
        return needle
    if "unused" in needle:
        return "fee"
    return "other"


def build_charge_category_from_types(
    charge_type: pd.Series, *, provider: str
) -> pd.Series:
    """Derive a normalized charge category from provider charge types."""
    provider = provider.lower()

    def classify(value: Any) -> str:
        text = (str(value) if value is not None else "").strip().lower()
        if not text:
            return "usage"
        if provider == "aws":
            mapping = {
                "usage": "usage",
                "discountedusage": "usage",
                "savingsplancoveredusage": "usage",
                "savingsplannegation": "commitment",
                "rifee": "commitment",
                "savingsplanrecurringfee": "commitment",
                "savingsplanupfrontfee": "commitment",
                "fee": "fee",
                "tax": "tax",
                "credit": "credit",
                "refund": "refund",
                "adjustment": "adjustment",
                "bundleddiscount": "discount",
                "privatepricing": "discount",
                "edpdiscount": "discount",
                "support": "fee",
            }
            return mapping.get(text, "other")
        if provider == "azure":
            mapping = {
                "usage": "usage",
                "purchase": "purchase",
                "credit": "credit",
                "refund": "refund",
                "tax": "tax",
                "unusedreservation": "fee",
                "unusedsavingsplan": "fee",
                "roundingadjustment": "adjustment",
            }
            return mapping.get(text, "other")
        if provider == "gcp":
            mapping = {
                "regular": "usage",
                "spend": "usage",
                "tax": "tax",
                "adjustment": "adjustment",
                "rounding_error": "adjustment",
                "credit": "credit",
            }
            return mapping.get(text, "other")
        return _normalize_charge_category(text)

    return charge_type.map(classify)


def with_normalized_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure all normalized columns exist (used by parsers before finalize)."""
    for column in NORMALIZED_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame
