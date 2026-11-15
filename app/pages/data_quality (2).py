"""Data Quality: coverage, gaps, freshness and import warnings."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.processing.data_quality import (
    coverage_by_series,
    freshness_report,
    import_warning_summary,
    missing_period_details,
    quality_summary,
)
from app.ui.components import cached_imports, download_csv, empty_state, load_costs, page_header, scope_sidebar
from app.utils.formatting import format_number, format_percent


def render() -> None:
    page_header(
        "Data Quality",
        "Missing billing days are not assumed to be zero-cost days. This page distinguishes "
        "observed records, candidate gaps, incomplete coverage and delayed exports.",
    )
    filters = scope_sidebar(key_prefix="quality")
    frame = load_costs(filters)
    if frame.empty:
        empty_state(
            "No billing records are available for a quality assessment.",
            "Import billing data first; quality reporting describes real application state only.",
        )
        return

    imports = cached_imports()
    summary = quality_summary(frame, imports)
    coverage = summary.get("coverage", {})
    metrics = st.columns(4)
    metrics[0].metric("Records", format_number(summary.get("records")))
    metrics[1].metric("Series", format_number(coverage.get("series")))
    metrics[2].metric("Coverage ratio", format_percent(coverage.get("coverage_ratio")))
    metrics[3].metric("Rejected records", format_number(summary.get("rejected_records")))

    cohorts = summarize_cohorts(frame)
    st.subheader("Reporting cohorts")
    st.dataframe(cohorts, hide_index=True, use_container_width=True)
    st.caption(
        "Currencies and cost bases are never combined. Each cohort has its own coverage and "
        "freshness indicators."
    )

    st.subheader("Freshness")
    freshness = freshness_report(frame, imports)
    columns = st.columns(4)
    columns[0].metric("Earliest usage", freshness.get("earliest_usage_date") or "-")
    columns[1].metric("Latest usage", freshness.get("latest_usage_date") or "-")
    columns[2].metric("Latest invoice month", freshness.get("latest_invoice_month") or "-")
    columns[3].metric("Latest import", (freshness.get("latest_import_completed_at") or "-")[:19])
    st.caption(
        "Cloud billing exports are delivered and revised with a delay. The latest available date "
        "is not a real-time spend figure."
    )

    st.subheader("Per-series coverage")
    series_coverage = coverage_by_series(frame)
    if series_coverage.empty:
        st.info("No series coverage is available.")
    else:
        display = series_coverage.rename(
            columns={
                "first_date": "First date",
                "last_date": "Last date",
                "calendar_days": "Calendar days",
                "days_with_records": "Days with records",
                "coverage_ratio": "Coverage",
                "zero_cost_days": "Zero-cost days",
                "missing_days": "Candidate missing days",
                "records": "Records",
                "cost": "Cost",
            }
        )
        st.dataframe(display, hide_index=True, use_container_width=True, height=380)
        download_csv(series_coverage, "series_coverage.csv", "Download coverage CSV")

    st.subheader("Missing periods")
    missing = missing_period_details(frame)
    st.caption(
        f"{missing.get('series_with_gaps', 0)} series have candidate gaps totalling "
        f"{missing.get('candidate_missing_days', 0)} day(s). "
        + (missing.get("note") or "")
    )
    examples = missing.get("examples") or []
    if examples:
        rows = []
        for item in examples:
            row = {f"series_{index}": value for index, value in enumerate(item["series"].values())}
            row["series"] = item["series"]
            row["missing_days"] = item["missing_days"]
            row["first_missing"] = ", ".join(item["first_missing"])
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Import validation warnings")
    warnings = import_warning_summary(imports)
    if warnings.empty:
        st.success("No import warnings or errors have been recorded.")
    else:
        severity_counts = warnings["severity"].value_counts().astype(int).to_dict()
        columns = st.columns(3)
        columns[0].metric("Errors", severity_counts.get("error", 0))
        columns[1].metric("Warnings", severity_counts.get("warning", 0))
        columns[2].metric("Info", severity_counts.get("info", 0))
        st.dataframe(warnings, hide_index=True, use_container_width=True)
        download_csv(warnings, "import_warnings.csv", "Download warnings CSV")


def summarize_cohorts(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (currency, basis), group in frame.groupby(["currency", "cost_basis"], dropna=False):
        rows.append(
            {
                "Currency": currency,
                "Cost basis": basis,
                "Records": len(group),
                "Providers": ", ".join(sorted(group["provider"].dropna().astype(str).unique())),
                "First date": pd.to_datetime(group["usage_start"], errors="coerce").min(),
                "Last date": pd.to_datetime(group["usage_start"], errors="coerce").max(),
            }
        )
    return pd.DataFrame(rows)
