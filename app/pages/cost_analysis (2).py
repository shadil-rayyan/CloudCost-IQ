"""Cost Analysis: filters, trends, breakdowns and detailed billing records."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.analytics.cost_breakdown import DIMENSION_COLUMNS, breakdown
from app.analytics.cost_summary import monthly_totals, summarize_costs
from app.analytics.cost_trends import cost_trend
from app.analytics.provider_comparison import provider_comparison
from app.ui.components import (
    bar_chart,
    cohort_selector,
    currency_for,
    download_csv,
    empty_state,
    kpi_cards,
    load_costs,
    page_header,
    scope_sidebar,
    trend_chart,
)


def render() -> None:
    page_header(
        "Cost Analysis",
        "Filter and break down spending across providers, accounts, services, regions and dates. "
        "Currencies and cost bases are always reported separately.",
    )
    filters = scope_sidebar(
        include_accounts=True,
        include_services=True,
        include_regions=True,
        key_prefix="analysis",
    )
    frame = load_costs(filters)
    if frame.empty:
        empty_state(
            "No billing records match the selected scope.",
            "Adjust the filters or import billing data first.",
        )
        return

    summary = summarize_costs(frame)
    kpi_cards(summary)
    cohort = cohort_selector(summary, key="analysis_cohort")
    if cohort is None:
        return
    currency, basis = currency_for(summary, cohort)
    cohort_frame = frame.loc[
        (frame["currency"] == cohort["currency"]) & (frame["cost_basis"] == cohort["cost_basis"])
    ].copy()
    if cohort_frame.empty:
        st.info("The selected reporting cohort has no records.")
        return

    st.caption(
        f"Analyzing {len(cohort_frame):,} records in {cohort['currency']} on the "
        f"{cohort['cost_basis']} cost basis."
    )
    _trend_section(cohort_frame, currency, basis)
    _breakdown_section(cohort_frame, currency, basis)
    _comparison_section(cohort_frame)
    _monthly_section(cohort_frame, currency)
    _records_section(cohort_frame)


def _trend_section(frame: pd.DataFrame, currency: str | None, basis: str | None) -> None:
    st.subheader("Cost trend")
    controls = st.columns(3)
    with controls[0]:
        frequency = st.selectbox(
            "Granularity",
            options=["D", "W", "M"],
            format_func={"D": "Daily", "W": "Weekly", "M": "Monthly"}.get,
            index=0,
            key="analysis_freq",
        )
    with controls[1]:
        dimension = st.selectbox(
            "Split by",
            options=["provider", "service_name", "billing_account_id", "region"],
            format_func={
                "provider": "Provider",
                "service_name": "Service",
                "billing_account_id": "Account",
                "region": "Region",
            }.get,
            index=0,
            key="analysis_trend_dimension",
        )
    with controls[2]:
        st.caption("Weekly and monthly buckets use the billing date (UTC).")

    trend = cost_trend(frame, freq=frequency, group_by=(dimension,))
    trend_chart(trend, dimension=dimension, currency=currency, basis=basis, height=380)
    download_csv(trend, f"cost_trend_{frequency}.csv", "Download trend CSV")


def _breakdown_section(frame: pd.DataFrame, currency: str | None, basis: str | None) -> None:
    st.subheader("Cost breakdown")
    controls = st.columns(2)
    with controls[0]:
        dimension = st.selectbox(
            "Breakdown dimension",
            options=list(DIMENSION_COLUMNS.keys()),
            format_func=lambda value: value.replace("_", " ").title(),
            index=0,
            key="analysis_breakdown_dimension",
        )
    with controls[1]:
        top_n = st.slider("Rows shown", min_value=5, max_value=50, value=15, step=5)

    result = breakdown(frame, dimension, top_n=top_n)
    if result.empty:
        st.info("No breakdown is available for this dimension.")
        return
    column = result.columns[0]
    bar_chart(result, label_column=column, currency=currency, height=340)
    display = result.rename(
        columns={
            "cost": f"Cost ({currency or ''})".strip(),
            "share": "Share",
            "records": "Records",
            column: column.replace("_", " ").title(),
        }
    )
    st.dataframe(display, hide_index=True, use_container_width=True)
    download_csv(result, f"cost_breakdown_{dimension}.csv", "Download breakdown CSV")


def _comparison_section(frame: pd.DataFrame) -> None:
    st.subheader("Provider comparison")
    comparison = provider_comparison(frame)
    if comparison.empty:
        return
    display = comparison.rename(
        columns={"cost": "Cost", "share": "Share", "records": "Records", "services": "Services"}
    )
    st.dataframe(display, hide_index=True, use_container_width=True)


def _monthly_section(frame: pd.DataFrame, currency: str | None) -> None:
    st.subheader("Invoice-month totals")
    monthly = monthly_totals(frame)
    if monthly.empty:
        return
    st.caption(
        "Invoice month is preserved separately from the usage date, so revised invoices stay "
        "attributable to their billing period."
    )
    st.dataframe(
        monthly.rename(columns={"cost": f"Cost ({currency or ''})".strip()}),
        hide_index=True,
        use_container_width=True,
    )
    download_csv(monthly, "invoice_month_totals.csv", "Download monthly totals")


def _records_section(frame: pd.DataFrame) -> None:
    st.subheader("Detailed billing records")
    st.caption(
        "Records preserve provider-specific fields in `raw_fields_json`. Missing optional values "
        "remain null and are never imputed."
    )
    limit = st.slider("Rows displayed", min_value=100, max_value=5000, value=500, step=100)
    visible_columns = [
        column
        for column in (
            "provider",
            "billing_account_id",
            "service_name",
            "service_category",
            "region",
            "usage_start",
            "invoice_month",
            "cost_amount",
            "currency",
            "cost_basis",
            "charge_category",
            "resource_id",
        )
        if column in frame.columns
    ]
    ordered = frame.sort_values(
        "usage_start" if "usage_start" in frame.columns else frame.columns[0],
        ascending=False,
    )
    st.dataframe(
        ordered[visible_columns].head(limit),
        hide_index=True,
        use_container_width=True,
        height=420,
    )
    download_csv(
        frame[visible_columns].head(limit),
        "billing_records_sample.csv",
        "Download displayed records",
    )
