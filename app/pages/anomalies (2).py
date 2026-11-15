"""Anomaly Detection: transparent statistical signals and forecast intervals."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from app.analytics.anomaly_detection import (
    anomaly_summary,
    detect_from_forecast,
    detect_statistical_anomalies,
)
from app.analytics.cost_summary import summarize_costs
from app.core.exceptions import CloudCostError
from app.ml.predict import ForecastRequest, ForecastService
from app.ui.components import (
    cohort_selector,
    currency_for,
    download_csv,
    empty_state,
    load_costs,
    page_header,
    scope_sidebar,
)
from app.utils.formatting import format_currency, format_percent


def render() -> None:
    page_header(
        "Anomaly Detection",
        "Identify unusual spending with a transparent rolling median/MAD rule, or compare actual "
        "cost with a validated model's P10-P90 forecast interval. Signals are never described as "
        "confirmed incidents.",
    )
    filters = scope_sidebar(
        include_accounts=True,
        include_services=True,
        key_prefix="anomaly",
    )
    frame = load_costs(filters)
    if frame.empty:
        empty_state("No billing records match the selected scope.")
        return

    summary = summarize_costs(frame)
    cohort = cohort_selector(summary, key="anomaly_cohort")
    if cohort is None:
        return
    currency, basis = currency_for(summary, cohort)
    cohort_frame = frame.loc[
        (frame["currency"] == cohort["currency"]) & (frame["cost_basis"] == cohort["cost_basis"])
    ].copy()

    service = ForecastService()
    availability = service.availability()
    modes = ["Statistical (rolling median/MAD)"]
    if availability.available:
        modes.append("Forecast interval (validated model)")
    mode = st.radio("Detection method", modes, horizontal=True)

    if mode.startswith("Statistical"):
        _statistical(cohort_frame, currency)
    else:
        _forecast_interval(service, availability, filters, currency, basis)


def _statistical(frame: pd.DataFrame, currency: str | None) -> None:
    default_threshold = 3.0
    default_lookback = 28
    columns = st.columns(3)
    with columns[0]:
        threshold = st.slider(
            "Robust z-score threshold", min_value=1.5, max_value=8.0, value=default_threshold, step=0.5
        )
    with columns[1]:
        lookback = st.slider(
            "Baseline window (days)", min_value=7, max_value=90, value=default_lookback, step=7
        )
    with columns[2]:
        st.caption(
            "The baseline only uses previous days (shifted rolling statistics), so a spike cannot "
            "inflate its own expected cost."
        )

    anomalies = detect_statistical_anomalies(frame, lookback_days=lookback, threshold=threshold)
    summary = anomaly_summary(anomalies)
    metrics = st.columns(3)
    metrics[0].metric("Unusual days", summary["count"])
    metrics[1].metric("High severity", summary["by_severity"].get("high", 0))
    metrics[2].metric("Latest signal", summary["latest_date"] or "-")

    if anomalies.empty:
        st.success("No unusual daily spending was detected with the current rule.")
        st.caption(
            "A clean result means the rule found no large deviation; it does not prove that spend "
            "is optimal. Sparse or short series cannot be evaluated reliably."
        )
        return

    display = anomalies.copy()
    for column in ("observed_cost", "expected_cost", "difference"):
        display[column] = display[column].map(lambda value: format_currency(value, currency))
    display["difference_pct"] = display["difference_pct"].map(format_percent)
    st.dataframe(
        display[
            [
                "detection_date",
                "provider",
                "billing_account_id",
                "service_name",
                "observed_cost",
                "expected_cost",
                "difference",
                "difference_pct",
                "severity",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Severity is derived from the robust z-score. Review the affected service and billing "
        "period before assuming a workload change; a billing correction can look identical."
    )
    download_csv(anomalies, "statistical_anomalies.csv", "Download anomalies CSV")

    with st.expander("Detection explanations", expanded=False):
        for _, row in anomalies.head(20).iterrows():
            st.markdown(f"- **{row['detection_date']} / {row['service_name']}** — {row['explanation']}")


def _forecast_interval(service: ForecastService, availability, filters, currency, basis) -> None:
    st.info(
        f"Using production model {availability.version}. The forecast is generated as of a past "
        "origin date so actual spending can be compared with the validated P10-P90 interval."
    )
    origin = st.date_input(
        "Forecast origin",
        value=filters.end_date or dt.date.today(),
        key="anomaly_origin",
    )
    horizon = int((availability.model_config or {}).get("forecast_horizon_days", 30))
    if st.button("Compare actual cost with forecast interval", type="primary"):
        request = ForecastRequest(
            provider=filters.providers[0] if len(filters.providers) == 1 else None,
            billing_account_id=filters.billing_accounts[0] if len(filters.billing_accounts) == 1 else None,
            service_name=filters.services[0] if len(filters.services) == 1 else None,
            currency=currency,
            cost_basis=basis,
            horizon_days=horizon,
            origin_date=origin,
        )
        try:
            result = service.generate(request, persist=False)
        except CloudCostError as exc:
            st.error(str(exc))
            return
        actual_source = load_costs(filters)
        actual = (
            actual_source.loc[actual_source["currency"] == currency]
            .assign(date=lambda value: pd.to_datetime(value["usage_start"], errors="coerce").dt.normalize())
            .groupby("date", as_index=False)["cost_amount"]
            .sum()
            .rename(columns={"cost_amount": "cost"})
        )
        forecast = (
            result.rows.assign(date=lambda value: pd.to_datetime(value["target_date"]).dt.normalize())
            .groupby("date", as_index=False)[["p10", "p50", "p90"]]
            .sum()
        )
        anomalies = detect_from_forecast(actual, forecast)
        if anomalies.empty:
            st.success("Actual daily cost stayed inside the forecast interval for the compared period.")
            return
        display = anomalies.copy()
        display["difference_pct"] = display["difference_pct"].map(format_percent)
        st.dataframe(display, hide_index=True, use_container_width=True)
        download_csv(anomalies, "forecast_interval_anomalies.csv", "Download anomalies CSV")
