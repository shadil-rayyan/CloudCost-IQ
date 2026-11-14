"""Azure Cost Management export parser."""

from __future__ import annotations

import pandas as pd

from app.ingestion.base_parser import BaseBillingParser, ColumnResolver
from app.processing.normalization import build_charge_category_from_types
from app.utils.dates import month_key


class AzureCostParser(BaseBillingParser):
    """Maps Azure Cost Management usage details to the normalized model.

    Azure exports do not always declare whether they contain Actual or
    Amortized cost, so the import form supplies the cost basis. Ambiguity is
    surfaced to the user rather than guessed.
    """

    provider = "Azure"
    schema_name = "Azure Cost Management export"
    description = "Azure Cost Management usage details / amortized or actual export"
    signature_columns = (
        "costinbillingcurrency",
        "subscriptionid",
        "meterid",
        "date",
    )

    def detect_score(self, resolver: ColumnResolver) -> float:
        score = self.score_from_signatures(resolver)
        if resolver.has("meterid") and resolver.has("subscriptionid"):
            score = min(1.0, score + 0.2)
        return score

    def required_fields(self) -> dict[str, tuple[str, ...]]:
        return {
            "usage_start": ("date", "usage_datetime", "usagedatetime", "usage_date", "usagedate", "charge_period_start", "chargeperiodstart"),
            "cost": ("cost_in_billing_currency", "costinbillingcurrency", "cost", "pretax_cost", "pretaxcost", "cost_amount"),
            "currency": ("billing_currency", "billingcurrency", "currency", "currency_code", "currencycode"),
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

        has_datetime = resolver.has("usagedatetime", "usage_start", "chargeperiodstart")
        start = resolver.datetime(
            raw, "usagedatetime", "usagedate", "date", "usage_start", "chargeperiodstart"
        )
        end = resolver.datetime(raw, "usage_end", "chargeperiodend", "usageend")
        if not has_datetime and end.isna().all():
            # Azure "actual cost" exports are daily; the charge period is the
            # calendar day the export reports.
            end = start + pd.Timedelta(days=1)
            notes.append(
                "Azure export reports daily granularity; charge period end was derived as date + 1 day."
            )
        frame["usage_start"] = start
        frame["usage_end"] = end
        frame["charge_period_start"] = resolver.datetime(raw, "chargeperiodstart").where(
            lambda values: values.notna(), start
        )
        frame["charge_period_end"] = resolver.datetime(raw, "chargeperiodend").where(
            lambda values: values.notna(), end
        )

        frame["cost_amount"] = resolver.numeric(
            raw, "costinbillingcurrency", "cost_amount", "cost", "pretaxcost"
        )
        if resolver.has("costusd") and resolver.has("billingcurrency") and resolver.has("currency"):
            billing = resolver.text(raw, "billingcurrency")
            usd = resolver.text(raw, "currency")
            if (billing != usd).any():
                notes.append(
                    "File contains both billing-currency and USD cost columns; billing currency was used."
                )
        frame["currency"] = resolver.text(raw, "billingcurrency", "currency", "currencycode")

        subscription = resolver.text(raw, "subscriptionid", "subscription_id")
        billing_account = resolver.text(raw, "billingaccountid", "billing_account_id")
        frame["billing_account_id"] = billing_account.where(billing_account.notna(), subscription)
        frame["subscription_id"] = subscription
        frame["billing_account_name"] = resolver.text(
            raw, "billingaccountname", "subscriptionname"
        )

        frame["service_name"] = resolver.first_text(
            raw,
            [
                ("servicename", "consumedservice"),
                ("metercategory", "servicefamily"),
            ],
        )
        frame["sku_id"] = resolver.text(raw, "meterid", "meter_id", "skuid")
        frame["sku_description"] = resolver.first_text(
            raw, [("metername",), ("metersubcategory",), ("skudescription",)]
        )
        frame["resource_id"] = resolver.text(raw, "resourceid", "resource_id")
        frame["region"] = resolver.first_text(
            raw,
            [
                ("resourcelocation", "resourceregion"),
                ("location",),
                ("regionid", "regionname"),
            ],
        )

        frame["quantity"] = resolver.numeric(raw, "quantity", "usagequantity", "consumedquantity")
        frame["unit"] = resolver.first_text(raw, [("unitofmeasure", "unit"), ("consumedunit",)])

        charge_type = resolver.text(raw, "chargetype", "type", "charge_category")
        frame["charge_type"] = charge_type
        frame["charge_category"] = build_charge_category_from_types(charge_type, provider="azure")
        frame["pricing_model"] = _azure_pricing_model(
            resolver.text(raw, "pricingmodel", "pricing_model")
        )

        period_start = resolver.datetime(raw, "billingperiodstart", "invoicedate")
        invoice = month_key(period_start)
        if invoice.isna().all() or (invoice == "").all():
            invoice = month_key(start)
        frame["invoice_month"] = invoice.where(invoice != "", None)
        frame["tags_json"] = resolver.series(raw, "tags", "resourcetags")

        cost = frame["cost_amount"]
        category = frame["charge_category"]
        frame["tax_amount"] = cost.where(category == "tax")
        frame["credits_amount"] = cost.where(category.isin(["credit", "refund"]))
        return frame


def _azure_pricing_model(values: pd.Series) -> pd.Series:
    def resolve(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        mapping = {
            "on demand": "on_demand",
            "ondemand": "on_demand",
            "reservation": "reservation",
            "savingsplan": "savings_plan",
            "savings plan": "savings_plan",
            "spot": "spot",
        }
        return mapping.get(text, text or None)

    return values.map(resolve)
