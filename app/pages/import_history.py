"""Import History: audit trail for every billing import."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui.components import cached_imports, download_csv, empty_state, page_header, show_import_warnings
from app.utils.formatting import format_number


def render() -> None:
    page_header(
        "Import History",
        "Every import is tracked with its fingerprint, schema, processing status, coverage and "
        "validation outcome. Superseded imports stay visible in the audit trail.",
    )
    records = cached_imports()
    if not records:
        empty_state(
            "No billing imports have been recorded yet.",
            "Open Import Billing Data to upload an AWS, Azure, GCP or FOCUS export.",
        )
        return

    providers = sorted({record.provider for record in records if record.provider})
    statuses = sorted({str(record.status) for record in records if record.status})
    controls = st.columns(3)
    with controls[0]:
        provider_filter = st.multiselect("Provider", providers, key="history_provider")
    with controls[1]:
        status_filter = st.multiselect("Status", statuses, key="history_status")
    with controls[2]:
        search = st.text_input("Filename contains", key="history_search")

    selected = [
        record
        for record in records
        if (not provider_filter or record.provider in provider_filter)
        and (not status_filter or str(record.status) in status_filter)
        and (not search or search.lower() in record.filename.lower())
    ]
    if not selected:
        st.info("No imports match the current filters.")
        return

    st.subheader("Imports")
    metrics = st.columns(4)
    metrics[0].metric("Imports", len(selected))
    metrics[1].metric("Completed", sum(1 for r in selected if str(r.status) == "completed"))
    metrics[2].metric("Failed", sum(1 for r in selected if str(r.status) == "failed"))
    metrics[3].metric(
        "Valid records",
        format_number(sum(record.valid_records for record in selected)),
    )

    table = pd.DataFrame(
        [
            {
                "Import ID": record.import_id,
                "Provider": record.provider,
                "Filename": record.filename,
                "Status": str(record.status),
                "Dataset": str(record.dataset_kind or "real"),
                "Schema": record.schema_name,
                "Version": record.schema_version,
                "Currency": record.currency,
                "Basis": record.cost_basis,
                "Coverage start": record.date_coverage_start,
                "Coverage end": record.date_coverage_end,
                "Rows read": record.records_read,
                "Valid": record.valid_records,
                "Rejected": record.rejected_records,
                "Uploaded": record.uploaded_at.isoformat()[:19] if record.uploaded_at else None,
                "Seconds": record.processing_seconds,
            }
            for record in selected
        ]
    )
    st.dataframe(table, hide_index=True, use_container_width=True)
    download_csv(table, "import_history.csv", "Download import history CSV")

    st.subheader("Import detail")
    by_id = {record.import_id: record for record in selected}
    choice = st.selectbox("Import", list(by_id), key="history_detail")
    record = by_id[choice]
    columns = st.columns(4)
    columns[0].metric("Status", str(record.status))
    columns[1].metric("Valid records", format_number(record.valid_records))
    columns[2].metric("Rejected records", format_number(record.rejected_records))
    columns[3].metric("Warnings", format_number(record.warning_count))
    st.caption(
        f"Fingerprint `{record.file_fingerprint or '-'}` | format {record.file_format or '-'} | "
        f"size {format_number(record.file_size_bytes)} bytes"
    )
    if record.error_message:
        st.error(record.error_message)
    if record.replaced_import_ids:
        st.info("Replaced imports: " + ", ".join(record.replaced_import_ids))

    show_import_warnings(record.warnings)
    if record.import_config:
        with st.expander("Import configuration and detection", expanded=False):
            st.json(record.import_config)
