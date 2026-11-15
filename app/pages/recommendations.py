"""Cost Optimization: evidence-based recommendations with human review."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.analytics.cost_summary import summarize_costs
from app.core.exceptions import CloudCostError
from app.recommendations.recommendation_engine import RecommendationEngine
from app.recommendations.recommendation_status import status_label, statuses
from app.ui.components import (
    cohort_selector,
    download_csv,
    empty_state,
    kpi_cards,
    load_costs,
    page_header,
    scope_sidebar,
)
from app.utils.formatting import format_currency


def render() -> None:
    page_header(
        "Cost Optimization",
        "Rule-based recommendations with explicit evidence, risk and limitations. Every item "
        "requires human review; CloudCost IQ never changes cloud resources automatically.",
    )
    filters = scope_sidebar(key_prefix="recommendations")
    frame = load_costs(filters)
    if frame.empty:
        empty_state("No billing records match the selected scope.")
        return

    summary = summarize_costs(frame)
    kpi_cards(summary)
    cohort = cohort_selector(summary, key="recommendations_cohort")
    if cohort is None:
        return
    cohort_frame = frame.loc[
        (frame["BillingCurrency"] == cohort["BillingCurrency"]) & (frame["cost_basis"] == cohort["cost_basis"])
    ].copy()

    engine = RecommendationEngine()
    columns = st.columns([1, 1, 3])
    generate = columns[0].button("Run recommendation rules", type="primary")
    columns[1].button("Refresh list")
    columns[2].caption(
        "Rules: service cost increase, budget projection, spending concentration, storage growth, "
        "unused commitment charges and largest 30-day growth."
    )
    if generate:
        try:
            drafts = engine.generate(cohort_frame)
        except Exception:  # pragma: no cover - defensive UI guard
            st.error("The recommendation rules could not be evaluated on this scope.")
            return
        st.success(f"{len(drafts)} evidence-based recommendation(s) generated or refreshed.")

    records = engine.list()
    if not records:
        empty_state(
            "No recommendations have been generated yet.",
            "Run the recommendation rules on a scoped dataset. Empty results are valid when no rule "
            "condition is met.",
        )
        _limitations()
        return

    _filters_and_table(records, engine)
    _limitations()


def _filters_and_table(records, engine: RecommendationEngine) -> None:
    st.subheader("Recommendations")
    controls = st.columns(3)
    with controls[0]:
        status_filter = st.multiselect(
            "Status", statuses(), default=statuses(), key="recommendation_status_filter"
        )
    with controls[1]:
        priority_filter = st.multiselect(
            "Priority", ["high", "medium", "low"], default=["high", "medium", "low"], key="recommendation_priority_filter"
        )
    selected = [
        record
        for record in records
        if str(record.status) in status_filter and record.priority in priority_filter
    ]

    counts: dict[str, int] = {}
    for record in records:
        counts[str(record.status)] = counts.get(str(record.status), 0) + 1
    metric_columns = st.columns(max(1, min(len(counts), 5)))
    for column, (status, count) in zip(metric_columns, sorted(counts.items()), strict=False):
        column.metric(status_label(status), count)

    table = pd.DataFrame(
        [
            {
                "Recommendation ID": record.recommendation_id,
                "Priority": record.priority,
                "Status": status_label(str(record.status)),
                "Title": record.title,
                "Provider": record.provider,
                "Account": record.account,
                "Service": record.service,
                "Estimated savings": record.estimated_savings,
                "Currency": record.currency,
                "Created": record.created_at.date().isoformat() if record.created_at else None,
            }
            for record in selected
        ]
    )
    if table.empty:
        st.info("No recommendations match the current filters.")
    else:
        st.dataframe(table, hide_index=True, use_container_width=True)
        download_csv(table, "recommendations.csv", "Download recommendations CSV")

    if not selected:
        return
    st.subheader("Review a recommendation")
    by_id = {record.recommendation_id: record for record in selected}
    choice = st.selectbox(
        "Recommendation",
        list(by_id),
        format_func=lambda value: (
            f"{by_id[value].priority.upper()} | {by_id[value].title} ({by_id[value].recommendation_id})"
        ),
        key="recommendation_choice",
    )
    record = by_id[choice]
    _render_record(record)
    _render_review(record, engine)
    _render_transparency(records)


def _render_record(record) -> None:
    st.markdown(f"### {record.title}")
    columns = st.columns(4)
    columns[0].metric("Priority", record.priority.title())
    columns[1].metric("Provider", record.provider or "-")
    columns[2].metric("Status", status_label(str(record.status)))
    columns[3].metric(
        "Estimated savings",
        format_currency(record.estimated_savings, record.currency)
        if record.estimated_savings is not None
        else "Not estimated",
    )
    st.markdown(
        f"**Affected account / service:** {record.account or 'not reported'} / "
        f"{record.service or 'not reported'}"
    )
    st.markdown(f"**Observation:** {record.observation}")
    st.markdown(f"**Evidence:** {record.evidence}")
    st.markdown(f"**Suggested investigation:** {record.suggested_action}")
    st.markdown(f"**Risk / limitation:** {record.risk}")
    period = (
        f"{record.period_start.isoformat() if record.period_start else '?'} to "
        f"{record.period_end.isoformat() if record.period_end else '?'}"
    )
    st.caption(
        f"Rule `{record.rule_id}` | period {period} | created "
        f"{record.created_at.isoformat()[:19] if record.created_at else '-'}"
    )
    if record.metadata:
        with st.expander("Supporting metadata", expanded=False):
            st.json(record.metadata)


def _render_review(record, engine: RecommendationEngine) -> None:
    st.markdown("**Update review status**")
    current = str(record.status)
    choices = [status for status in statuses() if status != current]
    columns = st.columns([2, 1, 2])
    with columns[0]:
        new_status = st.selectbox(
            "New status",
            choices,
            format_func=status_label,
            key=f"recommendation_new_status_{record.recommendation_id}",
        )
    with columns[1]:
        update = st.button("Save status", key=f"recommendation_update_{record.recommendation_id}")
    if update:
        try:
            updated = engine.update_status(record.recommendation_id, new_status)
        except CloudCostError as exc:
            st.error(str(exc))
            return
        if updated:
            st.success(f"Status updated to {status_label(new_status)}.")
            st.rerun()
        else:
            st.error("The recommendation could not be updated.")


def _render_transparency(records) -> None:
    with st.expander("How recommendations are produced", expanded=False):
        st.markdown(
            "- Recommendations are generated by transparent rules over the selected "
            "currency/cost-basis cohort.\n"
            "- No CPU, memory or resource-utilisation telemetry is used, so no machine rightsizing "
            "is recommended from billing cost alone.\n"
            "- Estimated savings are only reported where the rule defines them (for example the "
            "amount above a configured budget) and are not guaranteed savings.\n"
            "- No cloud resource is modified, resized, stopped or purchased by this application."
        )
        st.caption(f"{len(records)} recommendation(s) are stored in workspace metadata.")


def _limitations() -> None:
    st.caption(
        "Billing exports describe charges, not workload intent. A recommendation is a review "
        "prompt, not a confirmed inefficiency."
    )
