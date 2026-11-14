"""Overall cost summary and KPI calculations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.processing.aggregation import add_billing_date


def _group_key_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Summaries per currency and cost basis (never combined)."""
    if frame is None or frame.empty:
        return []
    working = add_billing_date(frame)
    summaries: list[dict[str, Any]] = []
    for (currency, basis), group in working.groupby(["currency", "cost_basis"], dropna=False):
        cost = pd.to_numeric(group["cost_amount"], errors="coerce").fillna(0.0)
        dates = group["billing_date"].dropna()
        providers = (
            group.groupby("provider")["cost_amount"]
            .sum()
            .sort_values(ascending=False)
            .to_dict()
        )
        summaries.append(
            {
                "currency": currency,
                "cost_basis": basis,
                "total_cost": float(cost.sum()),
                "records": int(len(group)),
                "first_date": dates.min().date().isoformat() if not dates.empty else None,
                "last_date": dates.max().date().isoformat() if not dates.empty else None,
                "days": int(dates.nunique()),
                "average_daily_cost": float(cost.sum() / dates.nunique()) if dates.nunique() else 0.0,
                "provider_costs": {str(key): float(value) for key, value in providers.items()},
                "positive_cost": float(cost[cost > 0].sum()),
                "negative_cost": float(cost[cost < 0].sum()),
            }
        )
    return sorted(summaries, key=lambda item: item["total_cost"], reverse=True)


def summarize_costs(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a complete, currency-safe summary of the selected records."""
    summaries = _group_key_rows(frame)
    currencies = sorted({item["currency"] for item in summaries if item["currency"]})
    bases = sorted({item["cost_basis"] for item in summaries if item["cost_basis"]})
    combined_total = None
    if len(currencies) == 1 and len(bases) == 1 and summaries:
        combined_total = float(sum(item["total_cost"] for item in summaries))
    return {
        "summaries": summaries,
        "records": int(len(frame)) if frame is not None else 0,
        "currencies": currencies,
        "cost_bases": bases,
        "combined_total": combined_total,
        "is_multi_currency": len(currencies) > 1,
        "is_multi_basis": len(bases) > 1,
        "mixed_totals_blocked": len(currencies) > 1 or len(bases) > 1,
    }


def kpis(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Dashboard KPI cards derived from the summary."""
    summary = summarize_costs(frame)
    cards: list[dict[str, Any]] = []
    for item in summary["summaries"]:
        cards.append(
            {
                "label": "Total cost",
                "value": item["total_cost"],
                "currency": item["currency"],
                "caption": f"{item['cost_basis']} basis | {item['records']:,} records",
            }
        )
    if not cards:
        cards.append(
            {
                "label": "Total cost",
                "value": 0.0,
                "currency": None,
                "caption": "No billing records in the selected scope",
            }
        )
    return cards


def provider_totals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["provider", "currency", "cost_basis", "cost", "share", "records"])
    grouped = (
        frame.groupby(["provider", "currency", "cost_basis"], dropna=False)
        .agg(cost=("cost_amount", "sum"), records=("cost_amount", "size"))
        .reset_index()
    )
    totals = grouped.groupby(["currency", "cost_basis"])["cost"].transform("sum").replace(0, pd.NA)
    grouped["share"] = (grouped["cost"] / totals).fillna(0.0)
    return grouped.sort_values("cost", ascending=False).reset_index(drop=True)


def monthly_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Invoice-month totals, preserving invoice month semantics."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["invoice_month", "currency", "cost_basis", "cost", "records"])
    return (
        frame.groupby(["invoice_month", "currency", "cost_basis"], dropna=False)
        .agg(cost=("cost_amount", "sum"), records=("cost_amount", "size"))
        .reset_index()
        .sort_values(["invoice_month", "currency"])
    )


def fastest_growing_services(
    frame: pd.DataFrame, *, lookback_days: int = 30, top_n: int = 5
) -> pd.DataFrame:
    """Compare the last ``lookback_days`` against the preceding period."""
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=["service_name", "currency", "cost_basis", "recent", "previous", "change", "change_pct"]
        )
    working = add_billing_date(frame)
    latest = working["billing_date"].max()
    if pd.isna(latest):
        return pd.DataFrame()
    recent_start = latest - pd.Timedelta(days=lookback_days - 1)
    previous_start = recent_start - pd.Timedelta(days=lookback_days)
    recent = working.loc[working["billing_date"] >= recent_start]
    previous = working.loc[
        (working["billing_date"] >= previous_start) & (working["billing_date"] < recent_start)
    ]
    keys = ["service_name", "currency", "cost_basis"]
    recent_totals = recent.groupby(keys, dropna=False)["cost_amount"].sum().rename("recent")
    previous_totals = previous.groupby(keys, dropna=False)["cost_amount"].sum().rename("previous")
    combined = pd.concat([recent_totals, previous_totals], axis=1).fillna(0.0).reset_index()
    combined["change"] = combined["recent"] - combined["previous"]
    combined["change_pct"] = combined.apply(
        lambda row: (row["change"] / row["previous"]) if row["previous"] else None, axis=1
    )
    combined = combined.loc[combined["change"] > 0]
    return combined.sort_values("change", ascending=False).head(top_n).reset_index(drop=True)
