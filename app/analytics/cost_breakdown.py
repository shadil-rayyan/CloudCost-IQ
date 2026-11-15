"""Cost breakdowns by service, account, region, project and SKU."""

from __future__ import annotations

import pandas as pd

from app.processing.aggregation import cost_by_dimension

DIMENSION_COLUMNS: dict[str, str] = {
    "ServiceProviderName": "ServiceProviderName",
    "service": "ServiceName",
    "ServiceCategory": "ServiceCategory",
    "account": "BillingAccountId",
    "subscription": "subscription_id",
    "project": "project_id",
    "RegionName": "RegionName",
    "sku": "SkuDescription",
    "resource": "ResourceId",
    "ChargeCategory": "ChargeCategory",
    "PricingCategory": "PricingCategory",
}


def resolve_dimension(dimension: str) -> str:
    return DIMENSION_COLUMNS.get(dimension, dimension)


def breakdown(
    frame: pd.DataFrame,
    dimension: str = "service",
    *,
    top_n: int | None = None,
    include_other: bool = True,
) -> pd.DataFrame:
    """Cost grouped by a dimension, with shares computed per currency/basis."""
    column = resolve_dimension(dimension)
    result = cost_by_dimension(frame, column, top_n=None)
    if result.empty or not top_n:
        return result

    kept: list[pd.DataFrame] = []
    for (currency, basis), group in result.groupby(["BillingCurrency", "cost_basis"], dropna=False):
        ordered = group.sort_values("cost", ascending=False)
        head = ordered.head(top_n)
        tail = ordered.iloc[top_n:]
        kept.append(head)
        if include_other and not tail.empty:
            kept.append(
                pd.DataFrame(
                    [
                        {
                            column: f"(other {len(tail)} groups)",
                            "cost": float(tail["cost"].sum()),
                            "records": int(tail["records"].sum()),
                            "share": float(tail["share"].sum()),
                            "BillingCurrency": currency,
                            "cost_basis": basis,
                        }
                    ]
                )
            )
    return pd.concat(kept, ignore_index=True).sort_values(
        ["BillingCurrency", "cost_basis", "cost"], ascending=[True, True, False]
    )


def top_services(frame: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    return breakdown(frame, "service", top_n=top_n)


def breakdown_by_service_category(frame: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    return breakdown(frame, "ServiceCategory", top_n=top_n)


def provider_service_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Service cost pivoted by provider (single currency/basis assumptions kept)."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(["ServiceName", "ServiceProviderName", "BillingCurrency", "cost_basis"], dropna=False)
        .agg(cost=("BilledCost", "sum"))
        .reset_index()
    )
    return grouped.sort_values("cost", ascending=False)
