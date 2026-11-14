"""Transparent rule definitions used by the recommendation engine.

Every rule documents the data it needs, the detection logic, the evidence it
emits and its limitations. No rule recommends changing infrastructure
automatically, and none uses CPU/memory data (billing exports do not contain
it).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from app.analytics.cost_summary import fastest_growing_services
from app.processing.aggregation import add_billing_date


@dataclass(slots=True)
class RuleContext:
    frame: pd.DataFrame
    monthly_budget: float | None = None
    currency: str | None = None
    cost_basis: str | None = None
    today: dt.date | None = None


@dataclass(slots=True)
class RecommendationDraft:
    rule_id: str
    title: str
    observation: str
    evidence: str
    suggested_action: str
    risk: str
    priority: str
    provider: str | None = None
    account: str | None = None
    service: str | None = None
    estimated_savings: float | None = None
    currency: str | None = None
    period_start: dt.date | None = None
    period_end: dt.date | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


RuleFunction = Callable[[RuleContext], list[RecommendationDraft]]


def _single_cohort(ctx: RuleContext) -> tuple[str | None, str | None]:
    """Rules run per currency/basis; mixed scopes are skipped by the engine."""
    if ctx.frame is None or ctx.frame.empty:
        return ctx.currency, ctx.cost_basis
    currencies = ctx.frame["currency"].dropna().unique().tolist()
    bases = ctx.frame["cost_basis"].dropna().unique().tolist()
    if len(currencies) > 1 or len(bases) > 1:
        return None, None
    return (
        ctx.currency or (currencies[0] if currencies else None),
        ctx.cost_basis or (bases[0] if bases else None),
    )


def service_cost_increase(
    ctx: RuleContext,
    *,
    threshold: float = 0.25,
    minimum_change: float = 20.0,
    window_days: int = 7,
) -> list[RecommendationDraft]:
    """Flag services whose recent cost is materially above the prior window."""
    drafts: list[RecommendationDraft] = []
    frame = ctx.frame
    if frame is None or frame.empty:
        return drafts
    currency, basis = _single_cohort(ctx)
    if currency is None:
        return drafts

    working = add_billing_date(frame)
    latest = working["billing_date"].max()
    if pd.isna(latest):
        return drafts
    recent_start = latest - pd.Timedelta(days=window_days - 1)
    previous_start = recent_start - pd.Timedelta(days=window_days)
    recent = working.loc[working["billing_date"] >= recent_start]
    previous = working.loc[
        (working["billing_date"] >= previous_start) & (working["billing_date"] < recent_start)
    ]
    keys = ["provider", "billing_account_id", "service_name"]
    recent_totals = recent.groupby(keys, dropna=False)["cost_amount"].sum().rename("recent")
    previous_totals = previous.groupby(keys, dropna=False)["cost_amount"].sum().rename("previous")
    combined = pd.concat([recent_totals, previous_totals], axis=1).fillna(0.0).reset_index()

    for _, row in combined.iterrows():
        previous_cost = float(row["previous"])
        recent_cost = float(row["recent"])
        change = recent_cost - previous_cost
        if previous_cost <= 0 or change < minimum_change:
            continue
        change_ratio = change / previous_cost
        if change_ratio < threshold:
            continue
        drafts.append(
            RecommendationDraft(
                rule_id="service_cost_increase",
                title=f"{row['service_name']} cost increased {change_ratio:.0%} week over week",
                provider=row["provider"],
                account=row["billing_account_id"],
                service=row["service_name"],
                observation=(
                    f"Daily cost for {row['service_name']} averaged "
                    f"{recent_cost / window_days:,.2f} {currency}/day in the latest {window_days} days, "
                    f"up from {previous_cost / window_days:,.2f} {currency}/day in the prior period."
                ),
                evidence=(
                    f"Recent total {recent_cost:,.2f} {currency} vs previous total "
                    f"{previous_cost:,.2f} {currency} ({change:+,.2f}, {change_ratio:+.1%})."
                ),
                suggested_action=(
                    "Compare deployment, instance-count, data-transfer and retention changes for this "
                    "service over the two periods, and confirm whether the increase is expected."
                ),
                risk=(
                    "Cost increases can be legitimate (growth, migrations, seasonal load). Billing data "
                    "cannot show CPU or memory usage, so no rightsizing action is inferred."
                ),
                priority="high" if change_ratio >= 0.5 else "medium",
                currency=currency,
                period_start=recent_start.date(),
                period_end=latest.date(),
                metadata={"recent_cost": recent_cost, "previous_cost": previous_cost, "window_days": window_days},
            )
        )
    return drafts


def budget_projection(ctx: RuleContext) -> list[RecommendationDraft]:
    """Project month-end spend against the configured workspace budget."""
    if not ctx.monthly_budget or ctx.frame is None or ctx.frame.empty:
        return []
    currency, basis = _single_cohort(ctx)
    if currency is None:
        return []
    working = add_billing_date(ctx.frame)
    latest = working["billing_date"].max()
    if pd.isna(latest):
        return []
    latest = pd.Timestamp(latest)
    month_start = latest.replace(day=1)
    month_to_date = working.loc[working["billing_date"] >= month_start, "cost_amount"].sum()
    days_elapsed = max((latest - month_start).days + 1, 1)
    days_in_month = latest.days_in_month
    projected = float(month_to_date) / days_elapsed * days_in_month
    if projected <= ctx.monthly_budget:
        return []
    overage = projected - ctx.monthly_budget
    return [
        RecommendationDraft(
            rule_id="budget_projection",
            title=f"Projected month-end spend exceeds budget by {overage:,.2f} {currency}",
            observation=(
                f"Month-to-date spend is {float(month_to_date):,.2f} {currency} after {days_elapsed} days. "
                f"At the current run rate the month would close at {projected:,.2f} {currency}."
            ),
            evidence=(
                f"Configured monthly budget: {ctx.monthly_budget:,.2f} {currency}. "
                f"Projection assumes the month-to-date daily average continues."
            ),
            suggested_action=(
                "Review the largest services in the current month and decide whether to adjust usage, "
                "or update the budget if the increase is planned."
            ),
            risk=(
                "The projection is a linear run-rate estimate, not a forecast model, and ignores known "
                "upcoming changes or one-off charges."
            ),
            priority="high" if overage / ctx.monthly_budget >= 0.2 else "medium",
            currency=currency,
            period_start=month_start.date(),
            period_end=latest.date(),
            estimated_savings=float(overage),
            metadata={
                "month_to_date": float(month_to_date),
                "projected_month_end": projected,
                "monthly_budget": ctx.monthly_budget,
                "note": "estimated savings is the amount above budget, not a guaranteed saving",
            },
        )
    ]


def spending_concentration(ctx: RuleContext, *, share_threshold: float = 0.6) -> list[RecommendationDraft]:
    """Flag when one service dominates spend in a currency/basis cohort."""
    frame = ctx.frame
    if frame is None or frame.empty:
        return []
    currency, basis = _single_cohort(ctx)
    if currency is None:
        return []
    grouped = (
        frame.groupby(["service_name", "provider"], dropna=False)["cost_amount"]
        .sum()
        .reset_index()
        .sort_values("cost_amount", ascending=False)
    )
    total = float(grouped["cost_amount"].sum())
    if total <= 0 or grouped.empty:
        return []
    top = grouped.iloc[0]
    share = float(top["cost_amount"]) / total
    if share < share_threshold:
        return []
    latest = add_billing_date(frame)["billing_date"].max()
    return [
        RecommendationDraft(
            rule_id="spending_concentration",
            title=f"{top['service_name']} accounts for {share:.0%} of selected spend",
            provider=top["provider"],
            service=top["service_name"],
            observation=(
                f"{top['service_name']} represents {share:.1%} of the selected {currency} spend "
                f"({float(top['cost_amount']):,.2f} {currency})."
            ),
            evidence=f"Total selected cost: {total:,.2f} {currency}; top service cost: {float(top['cost_amount']):,.2f} {currency}.",
            suggested_action=(
                "Confirm the service is intentionally the primary workload and review its cost drivers "
                "and ownership."
            ),
            risk="High concentration is not inherently wasteful; this is a review prompt, not a finding of overspend.",
            priority="medium" if share < 0.8 else "high",
            currency=currency,
            period_end=latest.date() if pd.notna(latest) else None,
            metadata={"share": share, "total_cost": total},
        )
    ]


def storage_growth(ctx: RuleContext, *, threshold: float = 0.2, window_days: int = 30) -> list[RecommendationDraft]:
    """Flag storage-category spend growth over the last 30 days."""
    frame = ctx.frame
    if frame is None or frame.empty or "service_category" not in frame.columns:
        return []
    currency, basis = _single_cohort(ctx)
    if currency is None:
        return []
    working = add_billing_date(frame)
    storage = working.loc[working["service_category"] == "storage"]
    if storage.empty:
        return []
    latest = storage["billing_date"].max()
    if pd.isna(latest):
        return []
    recent_start = latest - pd.Timedelta(days=window_days - 1)
    previous_start = recent_start - pd.Timedelta(days=window_days)
    recent = storage.loc[storage["billing_date"] >= recent_start, "cost_amount"].sum()
    previous = storage.loc[
        (storage["billing_date"] >= previous_start) & (storage["billing_date"] < recent_start),
        "cost_amount",
    ].sum()
    if previous <= 0 or recent <= 0:
        return []
    change_ratio = (recent - previous) / previous
    if change_ratio < threshold:
        return []
    return [
        RecommendationDraft(
            rule_id="storage_growth",
            title=f"Storage cost increased {change_ratio:.0%} over {window_days} days",
            provider=_first_non_null(storage["provider"]),
            service="Storage (category)",
            observation=(
                f"Storage-category spend was {float(recent):,.2f} {currency} in the latest {window_days} days "
                f"versus {float(previous):,.2f} {currency} in the prior {window_days} days."
            ),
            evidence=f"Change: {float(recent) - float(previous):+,.2f} {currency} ({change_ratio:+.1%}).",
            suggested_action=(
                "Review data growth, snapshot retention, unfinished multipart uploads and lifecycle "
                "policies for storage services."
            ),
            risk=(
                "Growth may be intentional. Billing data does not report bucket-level capacity or access "
                "patterns, so no deletion action is recommended automatically."
            ),
            priority="medium" if change_ratio < 0.5 else "high",
            currency=currency,
            period_start=recent_start.date(),
            period_end=latest.date(),
            metadata={"recent_cost": float(recent), "previous_cost": float(previous)},
        )
    ]


def unused_commitment_charges(ctx: RuleContext) -> list[RecommendationDraft]:
    """Surface unused reservation / savings-plan charges reported by providers."""
    frame = ctx.frame
    if frame is None or frame.empty or "charge_type" not in frame.columns:
        return []
    currency, basis = _single_cohort(ctx)
    if currency is None:
        return []
    text = frame["charge_type"].fillna("").astype(str).str.lower()
    unused_mask = text.str.contains("unusedreservation|unusedsavingsplan|savingsplanrecurringfee|^rifee$", regex=True)
    unused = frame.loc[unused_mask]
    if unused.empty:
        return []
    total = float(unused["cost_amount"].sum())
    if total <= 0:
        return []
    provider = _first_non_null(unused["provider"])
    return [
        RecommendationDraft(
            rule_id="unused_commitment_charges",
            title=f"Unused commitment charges detected ({total:,.2f} {currency})",
            provider=provider,
            observation=(
                "The billing export contains unused reservation or savings-plan charges "
                f"totalling {total:,.2f} {currency}."
            ),
            evidence=(
                "Charge types observed: "
                + ", ".join(sorted(unused["charge_type"].dropna().astype(str).unique().tolist())[:5])
            ),
            suggested_action=(
                "Review commitment coverage and utilization in the provider console before the next "
                "purchase or renewal."
            ),
            risk=(
                "Unused charges can be offset by future consumption within the commitment term. "
                "Do not cancel commitments without provider-side utilization data."
            ),
            priority="high" if total > 0 else "medium",
            currency=currency,
            period_start=pd.to_datetime(unused["usage_start"], errors="coerce").min(),
            period_end=pd.to_datetime(unused["usage_start"], errors="coerce").max(),
            metadata={"unused_charges": total},
        )
    ]


def service_portfolio_change(ctx: RuleContext, *, top_n: int = 5) -> list[RecommendationDraft]:
    """Highlight the services contributing the largest recent cost growth."""
    frame = ctx.frame
    if frame is None or frame.empty:
        return []
    currency, basis = _single_cohort(ctx)
    if currency is None:
        return []
    growth = fastest_growing_services(frame, top_n=top_n, lookback_days=30)
    if growth.empty:
        return []
    top = growth.iloc[0]
    if float(top["recent"]) < 10:
        return []
    return [
        RecommendationDraft(
            rule_id="service_growth_leader",
            title=f"Largest 30-day cost increase: {top['service_name']}",
            provider=_first_non_null(frame["provider"]),
            service=top["service_name"],
            observation=(
                f"{top['service_name']} increased by {float(top['change']):,.2f} {currency} "
                "between the last 30 days and the previous 30 days."
            ),
            evidence=(
                f"Recent: {float(top['recent']):,.2f}; previous: {float(top['previous']):,.2f}; "
                f"change: {float(top['change']):+,.2f} {currency}."
            ),
            suggested_action="Review the deployment and usage changes behind this service's growth.",
            risk="Cost growth is not proof of waste; verify workload intent before changing anything.",
            priority="medium",
            currency=currency,
            metadata={"recent": float(top["recent"]), "previous": float(top["previous"])},
        )
    ]


RULES: tuple[RuleFunction, ...] = (
    service_cost_increase,
    budget_projection,
    spending_concentration,
    storage_growth,
    unused_commitment_charges,
    service_portfolio_change,
)


def _first_non_null(series: pd.Series) -> str | None:
    if series is None:
        return None
    values = series.dropna().astype(str)
    return values.iloc[0] if not values.empty else None
