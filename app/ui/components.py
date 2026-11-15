"""Shared Streamlit components for the CloudCost IQ dashboard."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.analytics.store import CostFilter, CostStore, StoreDimensions
from app.core.config import get_app_config
from app.database.repositories import ImportRepository
from app.utils.formatting import (
    format_compact_currency,
    format_currency,
    format_percent,
)

PROVIDER_COLORS = {
    "AWS": "#f59e0b",
    "Azure": "#3b82f6",
    "GCP": "#10b981",
    "FOCUS": "#8b5cf6",
}


@st.cache_data(ttl=60, show_spinner=False)
def cached_dimensions() -> StoreDimensions:
    return CostStore().dimensions()


@st.cache_data(ttl=30, show_spinner=False)
def _load(
    providers: tuple[str, ...],
    accounts: tuple[str, ...],
    services: tuple[str, ...],
    regions: tuple[str, ...],
    currencies: tuple[str, ...],
    cost_basis: str | None,
    start_date: dt.date | None,
    end_date: dt.date | None,
) -> pd.DataFrame:
    filters = CostFilter(
        providers=providers,
        billing_accounts=accounts,
        services=services,
        regions=regions,
        currencies=currencies,
        cost_basis=cost_basis,
        start_date=start_date,
        end_date=end_date,
    )
    return CostStore().load(filters)


def load_costs(filters: CostFilter | None = None) -> pd.DataFrame:
    filters = filters or CostFilter()
    return _load(
        tuple(filters.providers),
        tuple(filters.billing_accounts),
        tuple(filters.services),
        tuple(filters.regions),
        tuple(filters.currencies),
        filters.cost_basis,
        filters.start_date,
        filters.end_date,
    )


@st.cache_data(ttl=15, show_spinner=False)
def cached_imports(limit: int = 200):
    return ImportRepository().list(limit=limit)


def clear_data_cache() -> None:
    _load.clear()
    cached_dimensions.clear()
    cached_imports.clear()


def page_header(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def empty_state(message: str, hint: str | None = None) -> None:
    st.info(message)
    if hint:
        st.caption(hint)


def scope_sidebar(
    *,
    include_accounts: bool = False,
    include_services: bool = False,
    include_regions: bool = False,
    include_dates: bool = True,
    key_prefix: str = "",
) -> CostFilter:
    """Render shared scope controls in the sidebar and return a CostFilter."""
    dimensions = cached_dimensions()
    prefix = f"{key_prefix}_" if key_prefix else ""
    st.sidebar.markdown("### Scope")
    if not dimensions.records:
        st.sidebar.caption("No billing data has been imported yet.")
        return CostFilter()

    providers = st.sidebar.multiselect(
        "Cloud provider", dimensions.providers, key=f"{prefix}providers"
    )
    currencies = st.sidebar.multiselect("Currency", dimensions.currencies, key=f"{prefix}currencies")
    basis_choice = st.sidebar.selectbox(
        "Cost basis",
        ["(all bases)"] + dimensions.cost_bases,
        key=f"{prefix}basis",
    )
    cost_basis = None if basis_choice == "(all bases)" else basis_choice

    accounts = (
        st.sidebar.multiselect("Billing account", dimensions.billing_accounts, key=f"{prefix}accounts")
        if include_accounts
        else []
    )
    services = (
        st.sidebar.multiselect("Service", dimensions.services, key=f"{prefix}services")
        if include_services
        else []
    )
    regions = (
        st.sidebar.multiselect("Region", dimensions.regions, key=f"{prefix}regions")
        if include_regions
        else []
    )

    start_date: dt.date | None = None
    end_date: dt.date | None = None
    if include_dates and dimensions.date_min and dimensions.date_max:
        selection = st.sidebar.date_input(
            "Billing date range",
            value=(dimensions.date_min, dimensions.date_max),
            min_value=dimensions.date_min,
            max_value=dimensions.date_max,
            key=f"{prefix}dates",
        )
        if isinstance(selection, tuple) and len(selection) == 2:
            start_date, end_date = selection

    return CostFilter(
        providers=tuple(providers),
        billing_accounts=tuple(accounts),
        services=tuple(services),
        regions=tuple(regions),
        currencies=tuple(currencies),
        cost_basis=cost_basis,
        start_date=start_date,
        end_date=end_date,
    )


def cohort_selector(summary: dict[str, Any], key: str = "cohort") -> dict[str, Any] | None:
    """Pick one currency/cost-basis cohort for charts that must not mix them."""
    summaries = summary.get("summaries", [])
    if not summaries:
        return None
    if len(summaries) == 1:
        return summaries[0]
    labels = [
        f"{item.get('BillingCurrency') or 'unknown currency'} / {item.get('cost_basis') or 'unknown basis'}"
        for item in summaries
    ]
    index = st.selectbox("Reporting cohort", range(len(labels)), format_func=lambda i: labels[i], key=key)
    return summaries[index]


def kpi_cards(summary: dict[str, Any]) -> None:
    summaries = summary.get("summaries", [])
    if not summaries:
        return
    columns = st.columns(min(len(summaries), 4))
    for column, item in zip(columns, summaries[:4], strict=False):
        with column:
            st.metric(
                label=f"{item['BillingCurrency']} total ({item['cost_basis']})",
                value=format_compact_currency(item["total_cost"], item["BillingCurrency"]),
                delta=f"{item['records']:,} records",
                delta_color="off",
            )
    if summary.get("is_multi_currency") or summary.get("is_multi_basis"):
        st.caption(
            "Multiple currencies or cost bases are present. Totals are reported per cohort and are "
            "never combined or converted."
        )


def trend_chart(
    trend: pd.DataFrame,
    *,
    dimension: str = "ServiceProviderName",
    currency: str | None = None,
    basis: str | None = None,
    height: int = 360,
) -> None:
    if trend is None or trend.empty:
        st.info("No cost trend is available for this scope.")
        return
    working = trend.copy()
    if currency is not None:
        working = working.loc[working["BillingCurrency"] == currency]
    if basis is not None:
        working = working.loc[working["cost_basis"] == basis]
    if working.empty:
        st.info("No cost trend is available for this cohort.")
        return
    working["period"] = pd.to_datetime(working["period"])
    figure = go.Figure()
    if dimension in working.columns and working[dimension].nunique() > 1:
        for value, group in working.groupby(dimension, dropna=False):
            figure.add_trace(
                go.Scatter(
                    x=group["period"],
                    y=group["cost"],
                    name=str(value),
                    mode="lines",
                    line={"color": PROVIDER_COLORS.get(str(value)), "width": 2},
                )
            )
    else:
        figure.add_trace(
            go.Scatter(
                x=working["period"],
                y=working["cost"],
                mode="lines",
                line={"color": "#2563eb", "width": 2},
                name="Cost",
            )
        )
    figure.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        yaxis_title=f"Cost ({currency or ''})".strip(),
        xaxis_title=None,
        legend={"orientation": "h", "y": 1.08},
        hovermode="x unified",
    )
    st.plotly_chart(figure, use_container_width=True)


def bar_chart(
    frame: pd.DataFrame,
    *,
    label_column: str,
    value_column: str = "cost",
    currency: str | None = None,
    height: int = 360,
) -> None:
    if frame is None or frame.empty:
        st.info("Nothing to chart for this scope.")
        return
    ordered = frame.sort_values(value_column, ascending=True).tail(15)
    figure = go.Figure(
        go.Bar(
            x=ordered[value_column],
            y=ordered[label_column].astype(str),
            orientation="h",
            marker_color="#3b82f6",
        )
    )
    figure.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis_title=f"Cost ({currency or ''})".strip(),
        yaxis_title=None,
    )
    st.plotly_chart(figure, use_container_width=True)


def download_csv(frame: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    if frame is None or frame.empty:
        return
    payload = frame.to_csv(index=False).encode("utf-8")
    st.download_button(label, payload, file_name=filename, mime="text/csv")


def show_import_warnings(warnings: Iterable[dict[str, Any]]) -> None:
    warnings = list(warnings or [])
    if not warnings:
        return
    with st.expander(f"Import warnings ({len(warnings)})", expanded=False):
        for warning in warnings:
            severity = warning.get("severity", "info")
            icon = {"error": "🔴", "warning": "🟠"}.get(severity, "🔵")
            st.markdown(
                f"{icon} **{warning.get('code', 'note')}** — {warning.get('message', '')}"
            )


def currency_for(summary: dict[str, Any], cohort: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if cohort is not None:
        return cohort.get("BillingCurrency"), cohort.get("cost_basis")
    currencies = summary.get("currencies") or []
    bases = summary.get("cost_bases") or []
    return (currencies[0] if len(currencies) == 1 else None, bases[0] if len(bases) == 1 else None)


def formatting_helpers():
    return {
        "BillingCurrency": format_currency,
        "compact_currency": format_compact_currency,
        "percent": format_percent,
    }


def app_context() -> dict[str, Any]:
    config = get_app_config()
    return {
        "app_name": config.app_name,
        "version": config.app_version,
        "BillingCurrency": config.reporting.default_currency,
        "cost_basis": config.reporting.default_cost_basis,
    }
