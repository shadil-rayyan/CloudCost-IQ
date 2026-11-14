"""Cost aggregation helpers shared by analytics, ML and the API.

All aggregations preserve the ``currency`` and ``cost_basis`` dimensions: the
application never sums incompatible currencies or cost definitions.
"""

from __future__ import annotations

import pandas as pd

from app.core.exceptions import CostBasisMismatchError


def _date_column(frame: pd.DataFrame) -> str:
    if "charge_period_start" in frame.columns and frame["charge_period_start"].notna().any():
        return "charge_period_start"
    return "usage_start"


def add_billing_date(frame: pd.DataFrame, column: str = "billing_date") -> pd.DataFrame:
    """Attach a normalized UTC calendar date used for daily aggregation."""
    result = frame.copy()
    source = _date_column(result)
    result[column] = pd.to_datetime(result[source], errors="coerce").dt.normalize()
    return result


def aggregate_costs(
    frame: pd.DataFrame,
    *,
    freq: str = "D",
    group_keys: tuple[str, ...] | list[str] | None = None,
    value_column: str = "cost_amount",
) -> pd.DataFrame:
    """Aggregate cost by time period and optional dimensions.

    The result keeps ``currency`` and ``cost_basis`` so callers can never
    accidentally combine them.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=["period", *(group_keys or []), "cost", "records", "currency", "cost_basis"]
        )
    working = add_billing_date(frame)
    period = pd.PeriodIndex(working["billing_date"], freq=freq)
    working["period"] = period.to_timestamp()

    keys = ["period", "currency", "cost_basis"]
    for key in group_keys or ():
        if key in working.columns and key not in keys:
            keys.append(key)

    grouped = (
        working.groupby(keys, dropna=False)
        .agg(cost=(value_column, "sum"), records=(value_column, "size"))
        .reset_index()
        .rename(columns={"billing_date": "period"})
    )
    return grouped.sort_values(keys).reset_index(drop=True)


def daily_costs(
    frame: pd.DataFrame,
    *,
    group_keys: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    return aggregate_costs(frame, freq="D", group_keys=group_keys)


def monthly_costs(
    frame: pd.DataFrame,
    *,
    group_keys: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    return aggregate_costs(frame, freq="M", group_keys=group_keys)


def assert_single_cost_basis(frame: pd.DataFrame) -> str:
    bases = frame["cost_basis"].dropna().astype(str).unique().tolist() if "cost_basis" in frame else []
    if len(bases) > 1:
        raise CostBasisMismatchError(
            f"Selected data mixes cost bases ({', '.join(sorted(bases))}); filter to one basis."
        )
    return bases[0] if bases else ""


def total_amount(frame: pd.DataFrame, *, value_column: str = "cost_amount") -> float:
    if frame is None or frame.empty or value_column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[value_column], errors="coerce").fillna(0.0).sum())


def net_cost(frame: pd.DataFrame) -> pd.Series:
    """Cost already includes credits in most exports; expose credits separately."""
    if frame is None or frame.empty:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame["cost_amount"], errors="coerce").fillna(0.0)


def cost_by_dimension(
    frame: pd.DataFrame,
    dimension: str,
    *,
    top_n: int | None = None,
    value_column: str = "cost_amount",
) -> pd.DataFrame:
    """Group cost by a categorical dimension (service, account, region...)."""
    if frame is None or frame.empty or dimension not in frame.columns:
        return pd.DataFrame(columns=[dimension, "cost", "records", "share", "currency", "cost_basis"])
    working = frame.copy()
    working[dimension] = working[dimension].fillna("(not reported)")
    grouped = (
        working.groupby([dimension, "currency", "cost_basis"], dropna=False)
        .agg(cost=(value_column, "sum"), records=(value_column, "size"))
        .reset_index()
    )
    totals = grouped.groupby(["currency", "cost_basis"])["cost"].transform("sum")
    grouped["share"] = (grouped["cost"] / totals.replace(0, pd.NA)).fillna(0.0)
    grouped = grouped.sort_values(["currency", "cost_basis", "cost"], ascending=[True, True, False])
    if top_n:
        grouped = grouped.groupby(["currency", "cost_basis"], group_keys=False).head(top_n)
    return grouped.reset_index(drop=True)
