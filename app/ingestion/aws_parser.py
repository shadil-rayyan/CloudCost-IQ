"""AWS Cost and Usage Report (CUR 2.0 / Data Exports) parser."""

from __future__ import annotations

import pandas as pd

from app.ingestion.base_parser import BaseBillingParser, ColumnResolver, normalize_key
from app.processing.normalization import build_charge_category_from_types
from app.utils.dates import month_key, to_datetime_utc

_BILLED_COST_ALIASES = (
    "line_item_unblended_cost",
    "line_item_net_unblended_cost",
    "line_item_blended_cost",
)
_AMORTIZED_COST_ALIASES = (
    "line_item_net_amortized_cost",
    "line_item_amortized_cost",
    "reservation_effective_cost",
    "savings_plan_savings_plan_effective_cost",
    "line_item_unblended_cost",
    "line_item_net_unblended_cost",
)


class AWSCURParser(BaseBillingParser):
    """Maps AWS CUR 2.0 columns to the normalized billing model."""

    provider = "AWS"
    schema_name = "AWS Cost and Usage Report 2.0"
    description = "AWS Cost and Usage Report / Data Exports (CSV or Parquet)"
    signature_columns = (
        "line_item_usage_start_date",
        "line_item_unblended_cost",
        "product_servicecode",
        "line_item_line_item_type",
    )

    def detect_score(self, resolver: ColumnResolver) -> float:
        score = self.score_from_signatures(resolver)
        if resolver.has("identity_line_item_id", "bill_payer_account_id"):
            score = min(1.0, score + 0.2)
        return score

    def detect_schema_version(self, resolver: ColumnResolver) -> str:
        return "2.0"

    def required_fields(self) -> dict[str, tuple[str, ...]]:
        return {
            "usage_start": (
                "line_item_usage_start_date",
                "line_item_usage_start_time",
                "usage_start_date",
            ),
            "cost": _BILLED_COST_ALIASES + _AMORTIZED_COST_ALIASES,
            "currency": ("line_item_currency_code", "pricing_currency", "currency_code"),
            "service": ("product_servicecode", "product_product_name", "line_item_product_code"),
        }

    def map_frame(
        self,
        raw: pd.DataFrame,
        resolver: ColumnResolver,
        *,
        cost_basis: str,
        notes: list[str],
    ) -> pd.DataFrame:
        frame = pd.DataFrame(index=raw.index)

        frame["usage_start"] = resolver.datetime(
            raw, "line_item_usage_start_date", "line_item_usage_start_time", "usage_start_date"
        )
        frame["usage_end"] = resolver.datetime(
            raw, "line_item_usage_end_date", "line_item_usage_end_time", "usage_end_date"
        )
        frame["charge_period_start"] = resolver.datetime(
            raw, "bill_billing_period_start_date", "bill_billing_period_start"
        )
        frame["charge_period_end"] = resolver.datetime(
            raw, "bill_billing_period_end_date", "bill_billing_period_end"
        )

        amortized = cost_basis in {"amortized", "effective"}
        aliases = _AMORTIZED_COST_ALIASES if amortized else _BILLED_COST_ALIASES
        frame["cost_amount"] = resolver.numeric(raw, *aliases)
        if frame["cost_amount"].isna().all():
            fallback = _BILLED_COST_ALIASES if amortized else _AMORTIZED_COST_ALIASES
            frame["cost_amount"] = resolver.numeric(raw, *fallback)
            if frame["cost_amount"].notna().any():
                notes.append(
                    f"Requested {cost_basis} cost was unavailable; the closest available AWS cost column was used."
                )

        frame["currency"] = resolver.text(
            raw, "line_item_currency_code", "pricing_currency", "currency_code"
        )

        usage_account = resolver.text(raw, "line_item_usage_account_id", "line_item_usage_account")
        payer_account = resolver.text(raw, "bill_payer_account_id", "bill_billing_entity")
        frame["billing_account_id"] = payer_account.where(payer_account.notna(), usage_account)
        frame["linked_account_id"] = usage_account
        frame["billing_account_name"] = resolver.text(
            raw, "line_item_usage_account_name", "bill_payer_account_name"
        )

        frame["service_name"] = resolver.first_text(
            raw,
            [
                ("product_servicecode", "line_item_product_code"),
                ("product_product_name",),
            ],
        )
        frame["sku_id"] = resolver.text(raw, "product_sku", "product_skucode", "line_item_usage_type")
        frame["sku_description"] = resolver.first_text(
            raw,
            [
                ("line_item_usage_type", "product_usagetype"),
                ("product_product_name",),
            ],
        )
        frame["resource_id"] = resolver.text(raw, "line_item_resource_id", "resource_id")
        frame["region"] = resolver.first_text(
            raw, [("product_region_code", "product_region"), ("line_item_availability_zone",)]
        )

        quantity = resolver.numeric(raw, "line_item_usage_amount", "line_item_usage_quantity")
        unit = resolver.first_text(
            raw, [("pricing_unit", "line_item_usage_amount_unit"), ("product_unit",)]
        )
        frame["quantity"] = quantity
        frame["unit"] = unit

        charge_type = resolver.text(
            raw, "line_item_line_item_type", "line_item_line_item_description", "charge_type"
        )
        frame["charge_type"] = charge_type
        frame["charge_category"] = build_charge_category_from_types(charge_type, provider="aws")

        pricing_term = resolver.text(raw, "pricing_term", "pricing_model")
        frame["pricing_model"] = _aws_pricing_model(pricing_term, charge_type)

        invoice_source = resolver.datetime(
            raw, "bill_billing_period_start_date", "bill_billing_period_start"
        )
        invoice = month_key(invoice_source)
        frame["invoice_month"] = invoice.where(invoice.notna() & (invoice != ""), None)

        frame["tags_json"] = _aws_tags(raw, resolver)

        cost = frame["cost_amount"]
        category = frame["charge_category"]
        frame["tax_amount"] = cost.where(category == "tax")
        frame["credits_amount"] = cost.where(category.isin(["credit", "refund"]))
        return frame


def _aws_pricing_model(pricing_term: pd.Series, charge_type: pd.Series) -> pd.Series:
    def resolve(term: object, charge: object) -> str | None:
        term_text = str(term).strip().lower() if term is not None else ""
        charge_text = str(charge).strip().lower() if charge is not None else ""
        if term_text in {"on demand", "ondemand", "on-demand"}:
            return "on_demand"
        if "reserved" in term_text or "reservation" in charge_text or charge_text == "rifee":
            return "reservation"
        if "savings" in term_text or "savingsplan" in charge_text:
            return "savings_plan"
        if "spot" in term_text:
            return "spot"
        if charge_text in {"usage", "discountedusage", "savingsplancoveredusage"}:
            return "on_demand"
        return term_text or None

    return pd.Series(
        [resolve(term, charge) for term, charge in zip(pricing_term, charge_type, strict=False)],
        index=pricing_term.index,
    )


def _aws_tags(raw: pd.DataFrame, resolver: ColumnResolver) -> pd.Series:
    """Collect tags from ``resource_tags`` or ``resource_tags_user_*`` columns."""
    direct = resolver.find("resource_tags", "resource_tags_json")
    if direct is not None:
        return raw[direct]

    prefixed = resolver.columns_matching("resource_tags_user_")
    if not prefixed:
        return pd.Series([None] * len(raw), index=raw.index, dtype="object")

    tag_keys = {column: normalize_key(column).replace("resource_tags_user_", "") for column in prefixed}
    payloads: list[dict[str, object] | None] = []
    for _, row in raw[prefixed].iterrows():
        tags = {
            tag_keys[column]: row[column]
            for column in prefixed
            if row[column] is not None and not pd.isna(row[column])
        }
        payloads.append(tags or None)
    return pd.Series(payloads, index=raw.index)


# ``to_datetime_utc`` is imported for parity with other parsers and tests that
# reference the module-level import surface.
__all__ = ["AWSCURParser", "to_datetime_utc"]
