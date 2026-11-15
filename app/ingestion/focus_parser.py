"""FOCUS billing export parser.

FOCUS (FinOps Open Cost and Usage Specification) is the unified input format
supported by CloudCost IQ. A FOCUS file can contain one or several providers;
the *record* provider is taken from ``ProviderName`` where available, while
the *import* provider remains ``FOCUS`` (the file format).
"""

from __future__ import annotations

import pandas as pd

from app.core.exceptions import SchemaValidationError
from app.ingestion.base_parser import BaseBillingParser, ColumnResolver
from app.processing.normalization import build_charge_category_from_types
from app.utils.dates import month_key

_PROVIDER_ALIASES = {
    "aws": "AWS",
    "amazon": "AWS",
    "amazon web services": "AWS",
    "amazon web services, inc.": "AWS",
    "azure": "Azure",
    "microsoft": "Azure",
    "microsoft azure": "Azure",
    "gcp": "GCP",
    "google": "GCP",
    "google cloud": "GCP",
    "google cloud platform": "GCP",
}


class FOCUSParser(BaseBillingParser):
    """Maps FOCUS 1.x rows to the normalized billing model."""

    provider = "FOCUS"
    schema_name = "FOCUS"
    description = "FinOps Open Cost and Usage Specification export (FOCUS 1.0 / 1.1)"
    signature_columns = (
        "billedcost",
        "chargeperiodstart",
        "billingcurrency",
        "chargecategory",
    )

    def detect_score(self, resolver: ColumnResolver) -> float:
        score = self.score_from_signatures(resolver)
        if resolver.has("providername", "invoiceissuername"):
            score = min(1.0, score + 0.2)
        return score

    def detect_schema_version(self, resolver: ColumnResolver) -> str:
        version = resolver.find("focusversion", "focus_version", "specversion")
        if version is not None:
            return "declared"
        return "undeclared"

    def required_fields(self) -> dict[str, tuple[str, ...]]:
        return {
            "billed_cost": ("billedcost", "billed_cost"),
            "ChargePeriodStart": ("chargeperiodstart", "ChargePeriodStart"),
            "ChargePeriodEnd": ("chargeperiodend", "ChargePeriodEnd"),
            "billing_currency": ("billingcurrency", "billing_currency"),
        }

    def parse(self, raw, *, cost_basis, notes=None):  # type: ignore[override]
        resolver = ColumnResolver(raw.columns)
        self.validate_schema(resolver)
        collected = list(notes or [])
        version_column = resolver.find("focusversion", "focus_version", "specversion")
        declared_version: str | None = None
        if version_column is not None:
            values = raw[version_column].dropna().astype(str).unique().tolist()
            if values:
                declared_version = values[0]
                if len(values) > 1:
                    collected.append(
                        f"FOCUS file declares multiple schema versions ({', '.join(values[:5])}); "
                        "the first was recorded."
                    )
                major = declared_version.split(".")[0]
                if major not in {"0", "1"}:
                    raise SchemaValidationError(
                        f"Incompatible FOCUS schema version '{declared_version}'. "
                        "CloudCost IQ supports FOCUS 1.x."
                    )
        mapped = self.map_frame(raw, resolver, cost_basis=cost_basis, notes=collected)
        from app.ingestion.base_parser import (
            ParsedBatch,  # local import avoids cycle at module load
        )

        return ParsedBatch(
            frame=mapped,
            raw_frame=raw,
            provider=self.provider,
            schema_name=self.schema_name,
            schema_version=declared_version or "undeclared",
            notes=collected,
        )

    def map_frame(
        self,
        raw: pd.DataFrame,
        resolver: ColumnResolver,
        *,
        cost_basis: str,
        notes: list[str],
    ) -> pd.DataFrame:
        frame = pd.DataFrame(index=raw.index)
        basis = cost_basis if cost_basis in {"billed", "effective", "amortized", "list"} else "billed"
        cost_aliases = {
            "billed": ("billedcost", "billed_cost"),
            "actual": ("billedcost", "billed_cost"),
            "effective": ("effectivecost", "effective_cost"),
            "amortized": ("effectivecost", "effective_cost"),
            "list": ("listcost", "list_cost"),
        }[basis]
        frame["BilledCost"] = resolver.numeric(raw, *cost_aliases)
        if frame["BilledCost"].isna().all():
            frame["BilledCost"] = resolver.numeric(raw, "billedcost", "billed_cost")
            notes.append("Requested FOCUS cost column was unavailable; BilledCost was used instead.")

        frame["usage_start"] = resolver.datetime(raw, "chargeperiodstart", "ChargePeriodStart")
        frame["usage_end"] = resolver.datetime(raw, "chargeperiodend", "ChargePeriodEnd")
        frame["ChargePeriodStart"] = frame["usage_start"]
        frame["ChargePeriodEnd"] = frame["usage_end"]

        frame["BillingCurrency"] = resolver.text(raw, "billingcurrency", "billing_currency")
        frame["BillingAccountId"] = resolver.text(
            raw, "billingaccountid", "BillingAccountId"
        )
        frame["BillingAccountName"] = resolver.text(
            raw, "billingaccountname", "BillingAccountName"
        )
        frame["linked_account_id"] = resolver.text(raw, "subaccountid", "sub_account_id")
        frame["ServiceName"] = resolver.text(raw, "servicename", "ServiceName")
        frame["ServiceCategory"] = resolver.text(
            raw, "servicecategory", "ServiceCategory"
        )
        frame["SkuId"] = resolver.first_text(
            raw, [("skuid", "SkuId"), ("skupriceid", "sku_price_id")]
        )
        frame["SkuDescription"] = resolver.text(raw, "skumeter", "sku_meter", "skudetails")
        frame["ResourceId"] = resolver.text(raw, "resourceid", "ResourceId")
        frame["RegionName"] = resolver.first_text(
            raw, [("regionid", "region_id"), ("regionname", "region_name")]
        )
        frame["ConsumedQuantity"] = resolver.numeric(
            raw, "consumedquantity", "consumed_quantity", "pricingquantity"
        )
        frame["ConsumedUnit"] = resolver.first_text(
            raw, [("consumedunit", "consumed_unit"), ("pricingunit", "pricing_unit")]
        )

        charge_type = resolver.first_text(
            raw, [("chargesubcategory", "charge_subcategory"), ("chargecategory", "ChargeCategory")]
        )
        frame["charge_type"] = charge_type
        frame["ChargeCategory"] = build_charge_category_from_types(charge_type, provider="focus")
        frame["PricingCategory"] = _focus_pricing_model(
            resolver.text(raw, "pricingcategory", "pricing_category")
        )

        billing_period_start = resolver.datetime(
            raw, "billingperiodstart", "billing_period_start"
        )
        invoice = month_key(billing_period_start)
        if invoice.isna().all() or (invoice == "").all():
            invoice = month_key(frame["usage_start"])
        frame["invoice_month"] = invoice.where(invoice != "", None)
        frame["Tags"] = resolver.series(raw, "tags")

        provider_name = resolver.text(raw, "providername", "provider_name")
        frame["ServiceProviderName"] = [_map_provider(value) for value in provider_name]

        charge_category = frame["ChargeCategory"]
        cost = frame["BilledCost"]
        frame["tax_amount"] = cost.where(charge_category == "tax")
        frame["credits_amount"] = cost.where(charge_category.isin(["credit", "refund"]))
        return frame


def _map_provider(value: object) -> str:
    if value is None:
        return "FOCUS"
    text = str(value).strip()
    return _PROVIDER_ALIASES.get(text.lower(), "FOCUS")


def _focus_pricing_model(values: pd.Series) -> pd.Series:
    def resolve(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        mapping = {
            "on-demand": "on_demand",
            "on demand": "on_demand",
            "on_demand": "on_demand",
            "commitment-based": "commitment",
            "commitment based": "commitment",
            "spot": "spot",
            "reserved": "reservation",
        }
        return mapping.get(text, text or None)

    return values.map(resolve)
