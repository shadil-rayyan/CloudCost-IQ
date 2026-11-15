"""Overview dashboard: KPIs, provider comparison, trends and platform status."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.analytics.anomaly_detection import (
    anomaly_summary,
    detect_statistical_anomalies,
)
from app.analytics.cost_breakdown import top_services
from app.analytics.cost_summary import summarize_costs
from app.analytics.cost_trends import cost_trend
from app.analytics.provider_comparison import provider_comparison
from app.core.config import get_app_config
from app.ml.predict import ForecastService
from app.processing.data_quality import freshness_report
from app.recommendations.recommendation_engine import RecommendationEngine
from app.ui.components import (
    bar_chart,
    cached_imports,
    cohort_selector,
    currency_for,
    empty_state,
    kpi_cards,
    load_costs,
    page_header,
    scope_sidebar,
    trend_chart,
)
from app.utils.formatting import format_currency


def render() -> None:
    page_header(
        "Overview",
        "Unified multi-cloud cost visibility. Totals are always shown per currency and cost basis.",
    )
    filters = scope_sidebar()
    frame = load_costs(filters)
    if frame.empty:
        empty_state(
            "No billing records are available for this scope.",
            "Import an AWS, Azure, GCP or FOCUS billing export from the Import Billing Data page.",
        )
        _platform_status()
        return

    summary = summarize_costs(frame)
    kpi_cards(summary)
    cohort = cohort_selector(summary)
    currency, basis = currency_for(summary, cohort)

    if cohort is None:
        st.warning("Multiple reporting cohorts exist; select one to see trend and breakdown charts.")
        _platform_status()
        return

    cohort_frame = frame.loc[
        (frame["BillingCurrency"] == cohort["BillingCurrency"]) & (frame["cost_basis"] == cohort["cost_basis"])
    ]
    if cohort_frame.empty:
        st.info("The selected reporting cohort has no records.")
        _platform_status()
        return

    st.subheader("Daily cost trend")
    trend = cost_trend(cohort_frame, freq="D", group_by=("ServiceProviderName",))
    trend_chart(trend, currency=currency, basis=basis)

    left, right = st.columns(2)
    with left:
        st.subheader("Provider comparison")
        comparison = provider_comparison(cohort_frame)
        bar_chart(comparison, label_column="ServiceProviderName", currency=currency, height=300)
        st.dataframe(
            comparison[["ServiceProviderName", "cost", "share", "services", "accounts"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "cost": st.column_config.NumberColumn(f"Cost ({currency})", format="%.2f"),
                "share": st.column_config.NumberColumn("Share", format="%.1%%"),
            },
        )
    with right:
        st.subheader("Top services")
        services = top_services(cohort_frame, top_n=10)
        bar_chart(services, label_column="ServiceName", currency=currency, height=300)

    st.subheader("Recent cost anomalies")
    threshold = get_app_config().quality.anomaly_zscore_threshold
    anomalies = detect_statistical_anomalies(
        cohort_frame,
        lookback_days=get_app_config().quality.anomaly_lookback_days,
        threshold=threshold,
    )
    summary_anomalies = anomaly_summary(anomalies)
    if anomalies.empty:
        st.success("No unusual daily spending was detected with the current statistical rule.")
    else:
        st.caption(
            f"{summary_anomalies['count']} unusual day(s) detected with a rolling median/MAD rule. "
            "These are statistical signals, not confirmed incidents."
        )
        display = anomalies.head(5).copy()
        display["observed_cost"] = display["observed_cost"].map(
            lambda value: format_currency(value, currency)
        )
        display["expected_cost"] = display["expected_cost"].map(
            lambda value: format_currency(value, currency)
        )
        st.dataframe(
            display[["detection_date", "ServiceProviderName", "ServiceName", "observed_cost", "expected_cost", "severity"]],
            hide_index=True,
            use_container_width=True,
        )

    _forecast_status()
    _recommendation_status()
    _freshness(cohort_frame)


def _forecast_status() -> None:
    st.subheader("Forecast status")
    availability = ForecastService().availability()
    if availability.available:
        config = availability.model_config or {}
        metrics = (availability.metrics or {}).get("final_test", {})
        columns = st.columns(3)
        columns[0].metric("Production model", availability.version or "-")
        columns[1].metric("Horizon", f"{config.get('forecast_horizon_days', '-')} days")
        wape = metrics.get("wape")
        columns[2].metric("Out-of-sample WAPE", f"{wape:.1%}" if isinstance(wape, (int, float)) else "-")
        st.caption("Forecasts are available on the ML Predictions page.")
    else:
        st.info(availability.reason)
        st.caption("The cost analytics pages continue to work without a trained model.")


def _recommendation_status() -> None:
    st.subheader("Optimization recommendations")
    records = RecommendationEngine().list()
    if not records:
        st.caption("No recommendations have been generated yet. Open Cost Optimization to run the rules.")
        return
    counts: dict[str, int] = {}
    for record in records:
        counts[str(record.status)] = counts.get(str(record.status), 0) + 1
    columns = st.columns(min(len(counts), 4))
    labels = {
        "new": "New",
        "under_review": "Under review",
        "accepted": "Accepted",
        "rejected": "Rejected",
        "completed": "Completed",
    }
    for column, (status, count) in zip(columns, sorted(counts.items()), strict=False):
        column.metric(labels.get(status, status), count)
    st.caption("Every recommendation requires human review; nothing is changed automatically.")


def _freshness(frame: pd.DataFrame) -> None:
    st.subheader("Data freshness")
    freshness = freshness_report(frame, cached_imports())
    columns = st.columns(3)
    columns[0].metric("Latest usage date", freshness.get("latest_usage_date") or "-")
    columns[1].metric("Latest invoice month", freshness.get("latest_invoice_month") or "-")
    columns[2].metric("Latest import", (freshness.get("latest_import_completed_at") or "-")[:19])
    st.caption(
        "Cloud billing exports are delivered and revised with a delay. "
        "The latest dates describe available export data, not real-time spend."
    )


def _platform_status() -> None:
    st.subheader("Platform status")
    imports = cached_imports()
    completed = sum(1 for record in imports if str(record.status) == "completed")
    failed = sum(1 for record in imports if str(record.status) == "failed")
    columns = st.columns(3)
    columns[0].metric("Completed imports", completed)
    columns[1].metric("Failed imports", failed)
    columns[2].metric("Production model", "available" if ForecastService().availability().available else "not registered")
