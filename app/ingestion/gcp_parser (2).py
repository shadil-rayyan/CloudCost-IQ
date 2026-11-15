"""Google Cloud detailed billing export parser."""

from __future__ import annotations

import pandas as pd

from app.ingestion.base_parser import BaseBillingParser, ColumnResolver, jsonify_credits
from app.processing.normalization import build_charge_category_from_types
from app.utils.dates import month_key


class GCPBillingParser(BaseBillingParser):
    """Maps Google Cloud BigQuery detailed billing exports to the common model.

    GCP reports gross ``cost`` and separate ``credits``. The normalized
    ``cost_amount`` is the **net billed cost** (cost + credits) so totals are
    comparable with AWS and Azure, while ``credits_amount`` keeps the credit
    portion for transparency.
    """

    provider = "GCP"
    schema_name = "Google Cloud detailed billing export"
    description = "Google Cloud Billing export to BigQuery (CSV or Parquet download)"
    signature_columns = (
        "billing_account_id",
        "service_description",
        "usage_start_time",
        "cost",
    )

    def detect_score(self, resolver: ColumnResolver) -> float:
        score = self.score_from_signatures(resolver)
        if resolver.has("project_id") or resolver.has("projectid"):
            score = min(1.0, score + 0.2)
        return score

    def required_fields(self) -> dict[str, tuple[str, ...]]:
        return {
            "usage_start": ("usage_start_time", "usage_start", "usagestarttime"),
            "cost": ("cost", "cost_amount"),
            "currency": ("currency", "billing_currency"),
            "billing_account": ("billing_account_id", "billingaccountid"),
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
            raw, "usage_start_time", "usage_start", "usagestarttime"
        )
        frame["usage_end"] = resolver.datetime(
            raw, "usage_end_time", "usage_end", "usageendtime"
        )
        frame["charge_period_start"] = resolver.datetime(
            raw, "charge_period_start", "invoice_month_start"
        ).where(lambda values: values.notna(), frame["usage_start"])
        frame["charge_period_end"] = resolver.datetime(
            raw, "charge_period_end", "invoice_month_end"
        ).where(lambda values: values.notna(), frame["usage_end"])

        gross_cost = resolver.numeric(raw, "cost", "cost_amount")

        credits = resolver.numeric(raw, "credits_amount", "credit_amount")
        if credits.isna().all():
            credits_column = resolver.find("credits", "credits_json")
            if credits_column is not None:
                credits = raw[credits_column].map(jsonify_credits).astype("float64")
                if credits.notna().any():
                    notes.append(
                        "GCP credits were read from the nested credits field and netted into cost_amount."
                    )
        credits = credits.astype("float64")

        frame["cost_amount"] = gross_cost + credits.fillna(0.0)
        frame["credits_amount"] = credits
        frame["currency"] = resolver.text(raw, "currency", "billing_currency", "currency_code")

        frame["billing_account_id"] = resolver.text(
            raw, "billing_account_id", "billingaccountid"
        )
        frame["billing_account_name"] = resolver.text(
            raw, "billing_account_name", "billingaccountname"
        )
        frame["project_id"] = resolver.first_text(
            raw, [("project_id", "projectid"), ("project_number", "projectnumber")]
        )
        frame["project_name"] = resolver.text(raw, "project_name", "projectname")

        frame["service_name"] = resolver.first_text(
            raw,
            [
                ("service_description", "servicedescription"),
                ("service_id", "serviceid"),
            ],
        )
        frame["sku_id"] = resolver.text(raw, "sku_id", "skuid")
        frame["sku_description"] = resolver.text(
            raw, "sku_description", "skudescription", "sku_details"
        )
        frame["resource_id"] = resolver.first_text(
            raw,
            [
                ("resource_name", "resourcename"),
                ("resource_global_name", "resourceglobalname"),
            ],
        )
        frame["region"] = resolver.first_text(
            raw,
            [
                ("location_region", "locationregion"),
                ("location_location", "locationlocation"),
                ("region",),
            ],
        )

        frame["quantity"] = resolver.numeric(
            raw, "usage_amount", "usageamount", "usage_pricing_amount", "usagepricingamount"
        )
        frame["unit"] = resolver.first_text(
            raw, [("usage_unit", "usageunit"), ("pricing_unit", "pricingunit")]
        )

        charge_type = resolver.text(raw, "cost_type", "costtype", "transaction_type")
        frame["charge_type"] = charge_type
        frame["charge_category"] = build_charge_category_from_types(charge_type, provider="gcp")
        frame["pricing_model"] = _gcp_pricing_model(
            resolver.text(raw, "pricing_model", "pricingmodel", "cost_type")
        )

        invoice = resolver.text(raw, "invoice_month", "invoicemonth")
        if invoice.isna().all():
            invoice = month_key(frame["usage_start"])
        frame["invoice_month"] = invoice.where(invoice.notna() & (invoice != ""), None)
        frame["tags_json"] = resolver.series(raw, "labels", "tags")

        category = frame["charge_category"]
        frame["tax_amount"] = frame["cost_amount"].where(category == "tax")
        # Credits are already netted into cost_amount; expose them separately.
        return frame


def _gcp_pricing_model(values: pd.Series) -> pd.Series:
    def resolve(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        mapping = {
            "regular": "on_demand",
            "on_demand": "on_demand",
            "commitment": "commitment",
            "committed": "commitment",
            "preemptible": "spot",
            "spot": "spot",
        }
        return mapping.get(text, text or None)

    return values.map(resolve)
