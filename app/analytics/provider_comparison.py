"""Provider comparison analytics."""

from __future__ import annotations

import pandas as pd


def provider_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare providers within each currency/cost-basis cohort."""
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "ServiceProviderName",
                "cost",
                "records",
                "share",
                "services",
                "accounts",
                "BillingCurrency",
                "cost_basis",
            ]
        )
    grouped = (
        frame.groupby(["ServiceProviderName", "BillingCurrency", "cost_basis"], dropna=False)
        .agg(
            cost=("BilledCost", "sum"),
            records=("BilledCost", "size"),
            services=("ServiceName", "nunique"),
            accounts=("BillingAccountId", "nunique"),
        )
        .reset_index()
    )
    totals = grouped.groupby(["BillingCurrency", "cost_basis"])["cost"].transform("sum").replace(0, pd.NA)
    grouped["share"] = (grouped["cost"] / totals).fillna(0.0)
    return grouped.sort_values(["BillingCurrency", "cost_basis", "cost"], ascending=[True, True, False])


def provider_trend(frame: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    from app.processing.aggregation import aggregate_costs

    grouped = aggregate_costs(frame, freq=freq, group_keys=("ServiceProviderName",))
    return grouped.rename(columns={"period": "period"}).sort_values("period")


def provider_share_table(frame: pd.DataFrame) -> pd.DataFrame:
    comparison = provider_comparison(frame)
    if comparison.empty:
        return comparison
    return comparison[
        ["ServiceProviderName", "BillingCurrency", "cost_basis", "cost", "share", "services", "accounts"]
    ]
