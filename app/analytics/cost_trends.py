"""Cost trend series and period-over-period analysis."""

from __future__ import annotations

import pandas as pd

from app.processing.aggregation import add_billing_date, aggregate_costs


def cost_trend(
    frame: pd.DataFrame,
    *,
    freq: str = "D",
    group_by: tuple[str, ...] = ("ServiceProviderName",),
) -> pd.DataFrame:
    """Long-format cost trend, grouped by time, currency and cost basis."""
    return aggregate_costs(frame, freq=freq, group_keys=group_by)


def trend_pivot(
    frame: pd.DataFrame,
    *,
    freq: str = "D",
    dimension: str = "ServiceProviderName",
) -> pd.DataFrame:
    """Pivot a single currency/basis trend for stacked charting.

    Returns an empty frame when more than one currency/basis is present: the
    caller must filter first rather than see an invalid combined chart.
    """
    trend = cost_trend(frame, freq=freq, group_by=(dimension,))
    if trend.empty:
        return pd.DataFrame()
    if trend["BillingCurrency"].nunique() > 1 or trend["cost_basis"].nunique() > 1:
        return pd.DataFrame()
    pivot = trend.pivot_table(
        index="period", columns=dimension, values="cost", aggfunc="sum", fill_value=0.0
    )
    pivot = pivot.sort_index()
    pivot.index = pd.to_datetime(pivot.index)
    return pivot


def daily_series(single_series_frame: pd.DataFrame) -> pd.Series:
    """Daily total cost for one already-filtered series."""
    if single_series_frame is None or single_series_frame.empty:
        return pd.Series(dtype="float64")
    working = add_billing_date(single_series_frame)
    return (
        working.groupby("billing_date")["BilledCost"]
        .sum()
        .sort_index()
        .astype("float64")
    )


def moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    if series.empty:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def cumulative_costs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "BillingCurrency", "cost_basis", "cumulative_cost"])
    working = add_billing_date(frame)
    daily = (
        working.groupby(["billing_date", "BillingCurrency", "cost_basis"], dropna=False)["BilledCost"]
        .sum()
        .reset_index()
        .sort_values("billing_date")
    )
    daily["cumulative_cost"] = daily.groupby(["BillingCurrency", "cost_basis"])["BilledCost"].cumsum()
    return daily.rename(columns={"billing_date": "date"})


def period_change(frame: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Percentage change of cost between consecutive periods."""
    trend = cost_trend(frame, freq=freq, group_by=())
    if trend.empty:
        return trend
    trend = trend.sort_values("period")
    trend["previous_cost"] = trend.groupby(["BillingCurrency", "cost_basis"])["cost"].shift(1)
    trend["change"] = trend["cost"] - trend["previous_cost"]
    trend["change_pct"] = trend.apply(
        lambda row: (row["change"] / row["previous_cost"]) if row["previous_cost"] else None, axis=1
    )
    return trend.reset_index(drop=True)


def month_to_date(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    working = add_billing_date(frame)
    latest = working["billing_date"].max()
    if pd.isna(latest):
        return pd.DataFrame()
    month_start = latest.replace(day=1)
    current = working.loc[working["billing_date"] >= month_start]
    return (
        current.groupby(["BillingCurrency", "cost_basis"], dropna=False)["BilledCost"]
        .sum()
        .reset_index()
        .rename(columns={"BilledCost": "month_to_date"})
    )
