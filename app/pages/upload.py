"""Import Billing Data: inspect, validate and confirm a billing export."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from app.core.exceptions import CloudCostError
from app.ingestion.file_detector import PROVIDER_CHOICES
from app.ingestion.import_manager import ImportManager, ImportOptions
from app.ui.components import (
    clear_data_cache,
    download_csv,
    empty_state,
    page_header,
    show_import_warnings,
)
from app.utils.formatting import format_number


def render() -> None:
    page_header(
        "Import Billing Data",
        "Upload AWS CUR 2.0, Azure Cost Management, GCP detailed billing or a FOCUS export. "
        "Nothing is stored until you confirm the preview.",
    )

    options = ImportOptions()
    uploaded = st.file_uploader(
        "Billing export file",
        type=["csv", "parquet", "gz", "zip"],
        help="Supported: CSV, Parquet, GZIP-compressed CSV/Parquet and safely extracted ZIP.",
    )

    left, middle, right = st.columns(3)
    with left:
        provider = st.selectbox("Cloud provider", PROVIDER_CHOICES, index=0)
    with middle:
        cost_basis = st.selectbox(
            "Cost basis",
            ["billed", "actual", "amortized", "effective", "list"],
            index=0,
            help="Billed cost is the default analytics and ML target. Actual and amortized "
            "costs are kept in separate cohorts and are never mixed.",
        )
    with right:
        dataset_kind = st.selectbox(
            "Dataset kind",
            ["real", "demo", "test"],
            index=0,
            help="Label demo/test uploads explicitly. They are never treated as real billing history.",
        )
    options = replace(options, provider=provider, cost_basis=cost_basis, dataset_kind=dataset_kind)

    inspect_clicked = st.button("Inspect and validate file", type="primary", disabled=uploaded is None)
    if inspect_clicked and uploaded is not None:
        manager = ImportManager()
        try:
            staged = manager.stage_upload(uploaded.getvalue(), uploaded.name)
            analysis = manager.analyze(staged, options)
        except CloudCostError as exc:
            st.error(str(exc))
        except Exception:  # pragma: no cover - defensive UI guard
            st.error("The file could not be analyzed. Check that it is an unmodified billing export.")
        else:
            st.session_state["upload_analysis"] = analysis
            st.session_state["upload_options"] = options

    analysis = st.session_state.get("upload_analysis")
    if analysis is None:
        empty_state(
            "No file has been inspected in this session.",
            "Select a billing export above and choose Inspect and validate file.",
        )
        _import_help()
        return

    _render_analysis(analysis)
    stored_options: ImportOptions = st.session_state.get("upload_options", options)
    _render_confirmation(analysis, stored_options)


def _render_analysis(analysis) -> None:
    detected = analysis.detected
    st.subheader("Detected schema")
    columns = st.columns(4)
    columns[0].metric("Provider", detected.provider)
    columns[1].metric("Schema", detected.parser.schema_name)
    columns[2].metric("Schema version", analysis.schema_version)
    columns[3].metric("Detection score", f"{detected.score:.0%}")

    if detected.score < 0.8:
        st.warning(
            "Schema detection confidence is low. Confirm the provider selection before importing."
        )
    st.caption(
        "Detection candidates: "
        + ", ".join(f"{name} ({score:.0%})" for name, score in detected.candidates)
    )

    summary = analysis.summary()
    st.subheader("File summary")
    metrics = st.columns(4)
    metrics[0].metric("Records read", format_number(summary["records_read"]))
    metrics[1].metric("Valid records", format_number(summary["valid_records"]))
    metrics[2].metric("Rejected records", format_number(summary["rejected_records"]))
    metrics[3].metric("Duplicate rows", format_number(summary["duplicate_rows"]))
    details = st.columns(4)
    details[0].metric("Currency", ", ".join(summary["currencies"]) or "-")
    details[1].metric("Coverage start", summary["coverage_start"] or "-")
    details[2].metric("Coverage end", summary["coverage_end"] or "-")
    details[3].metric("File size", summary["file_size"])

    if summary["exact_duplicate_imports"]:
        st.warning(
            "This exact file fingerprint was imported before: "
            + ", ".join(summary["exact_duplicate_imports"][:5])
        )
    if summary["replaceable_imports"]:
        st.info(
            f"{len(summary['replaceable_imports'])} existing import(s) are fully covered by this "
            "file and can be replaced on confirmation."
        )
    if summary["partial_overlap_imports"]:
        st.warning(
            f"{len(summary['partial_overlap_imports'])} existing import(s) partially overlap this "
            "file. They will not be replaced; overlapping periods would be double counted."
        )

    st.subheader("Validation")
    issues = analysis.report.issues
    if not issues:
        st.success("No validation issues were found.")
    else:
        issue_rows = pd.DataFrame([issue.to_dict() for issue in issues])
        st.dataframe(
            issue_rows[["severity", "code", "count", "message", "column"]],
            hide_index=True,
            use_container_width=True,
        )
        errors = issue_rows.loc[issue_rows["severity"] == "error"]
        if not errors.empty:
            st.error(
                f"{int(errors['count'].sum())} record(s) failed error-level checks and will be "
                "excluded from the stored dataset. They remain downloadable below."
            )
        if not analysis.report.rejected_frame.empty:
            download_csv(
                analysis.report.rejected_frame,
                f"rejected_{analysis.staged.import_id}.csv",
                "Download rejected records",
            )

    show_import_warnings(analysis.warning_issues)

    st.subheader("Raw preview")
    st.dataframe(analysis.preview, hide_index=True, use_container_width=True, height=260)

    st.subheader("Normalized preview")
    preview = analysis.normalized_preview
    if preview.empty:
        st.info("No normalized records are available.")
    else:
        visible = [
            column
            for column in (
                "ServiceProviderName",
                "BillingAccountId",
                "ServiceName",
                "ServiceCategory",
                "RegionName",
                "usage_start",
                "BilledCost",
                "BillingCurrency",
                "cost_basis",
                "ChargeCategory",
            )
            if column in preview.columns
        ]
        st.dataframe(preview[visible], hide_index=True, use_container_width=True, height=260)


def _render_confirmation(analysis, options: ImportOptions) -> None:
    st.divider()
    st.subheader("Confirm import")
    replace_overlapping = st.checkbox(
        "Replace existing imports that are fully covered by this file",
        value=False,
        help="Existing metadata is marked superseded and its Parquet file is removed before the "
        "new import is stored. Partially overlapping imports are never replaced automatically.",
    )
    columns = st.columns([1, 1, 4])
    confirm = columns[0].button(
        "Confirm import",
        type="primary",
        disabled=not analysis.is_importable,
    )
    cancel = columns[1].button("Cancel")
    if not analysis.is_importable:
        columns[2].caption("Import is disabled because no valid records remain.")

    if cancel:
        ImportManager().cancel(analysis.staged)
        st.session_state.pop("upload_analysis", None)
        st.session_state.pop("upload_options", None)
        st.info("The staged upload was discarded.")
        return

    if confirm:
        manager = ImportManager()
        try:
            outcome = manager.commit(
                analysis,
                replace(options, replace_overlapping=replace_overlapping),
            )
        except CloudCostError as exc:
            st.error(str(exc))
            return
        finally:
            st.session_state.pop("upload_analysis", None)
            st.session_state.pop("upload_options", None)
        clear_data_cache()
        if outcome.status == "completed":
            st.success(
                f"Import {outcome.import_id} completed: {outcome.valid_records:,} valid records, "
                f"{outcome.rejected_records:,} rejected. Coverage {outcome.coverage_start or '-'} "
                f"to {outcome.coverage_end or '-'}."
            )
            if outcome.replaced_import_ids:
                st.info("Superseded imports: " + ", ".join(outcome.replaced_import_ids))
            st.caption("Open Import History for the full audit trail or Overview to analyze the data.")
        else:
            st.error(outcome.error_message or "The import failed.")


def _import_help() -> None:
    with st.expander("Supported billing exports", expanded=False):
        st.markdown(
            "- **AWS**: Cost and Usage Report 2.0 / Data Exports (CSV or Parquet).\n"
            "- **Azure**: Cost Management actual/amortized cost exports, CSV or Parquet.\n"
            "- **GCP**: Detailed Cloud Billing export to BigQuery, downloaded as CSV or Parquet.\n"
            "- **FOCUS**: FOCUS 1.x specification files; the version is read from the file, and "
            "provider-specific extension columns are preserved.\n\n"
            "Cost bases are never combined. Billed cost is the default target; actual and "
            "amortized costs remain separate."
        )
