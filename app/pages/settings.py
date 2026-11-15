"""Settings: workspace preferences, budgets and platform information."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.analytics.store import CostStore
from app.core.config import get_app_config, get_ml_config
from app.database.repositories import ImportRepository, SettingsRepository
from app.ml.artifact_manager import ArtifactManager
from app.processing.storage import ProcessedStore
from app.ui.components import clear_data_cache, page_header
from app.utils.formatting import format_number


def render() -> None:
    page_header(
        "Settings",
        "Workspace reporting preferences and operational information. Settings are stored locally "
        "in SQLite and never contain cloud credentials.",
    )
    config = get_app_config()
    repository = SettingsRepository()
    values = repository.get_all()

    st.subheader("Workspace")
    with st.form("workspace_settings"):
        workspace_name = st.text_input("Workspace name", value=values.get("workspace_name", "My workspace"))
        columns = st.columns(2)
        with columns[0]:
            currency = st.selectbox(
                "Reporting currency",
                ["USD", "EUR", "GBP", "INR", "AUD", "CAD", "JPY", "SGD", "AED", "BRL"],
                index=_currency_index(values.get("reporting_currency", "USD")),
                help="Display preference only. Billing records keep their original currency and are "
                "never converted or summed across currencies.",
            )
        with columns[1]:
            basis = st.selectbox(
                "Default cost basis",
                ["billed", "actual", "amortized", "effective", "list"],
                index=_basis_index(values.get("default_cost_basis", "billed")),
                help="Billed cost is the default ML target. Actual and amortized costs stay separate.",
            )
        monthly_budget = st.text_input(
            "Monthly budget (optional)",
            value=values.get("monthly_budget", "") or "",
            help="Used by the budget-projection recommendation rule. Leave blank for no budget.",
        )
        submitted = st.form_submit_button("Save settings", type="primary")
    if submitted:
        if monthly_budget.strip():
            try:
                budget_value = float(monthly_budget)
                if budget_value <= 0:
                    raise ValueError
            except ValueError:
                st.error("Monthly budget must be a positive number or left blank.")
                return
        else:
            budget_value = None
        repository.set_many(
            {
                "workspace_name": workspace_name.strip() or "My workspace",
                "reporting_currency": currency,
                "default_cost_basis": basis,
                "monthly_budget": "" if budget_value is None else budget_value,
            }
        )
        clear_data_cache()
        st.success("Workspace settings saved.")

    st.subheader("Data and storage")
    store = CostStore()
    dimensions = store.dimensions()
    stats = {
        "records": dimensions.records,
        "providers": len(dimensions.providers),
        "currencies": len(dimensions.currencies),
        "cost_bases": len(dimensions.cost_bases),
    }
    columns = st.columns(4)
    for column, (label, value) in zip(columns, stats.items(), strict=False):
        column.metric(label.replace("_", " ").title(), format_number(value))
    if dimensions.date_min and dimensions.date_max:
        st.caption(f"Billing coverage: {dimensions.date_min} to {dimensions.date_max}.")
    st.caption(
        "Normalized records are stored as immutable Parquet files per import; metadata lives in "
        "SQLite. Raw uploads are retained for lineage and can be removed by workspace operators."
    )

    st.subheader("Upload limits and validation")
    upload = config.uploads
    limits = pd.DataFrame(
        [
            {"Setting": "Max upload size", "Value": f"{upload.max_file_size_mb} MB"},
            {"Setting": "Max uncompressed archive size", "Value": f"{upload.max_uncompressed_size_mb} MB"},
            {"Setting": "Max archive members", "Value": str(upload.max_archive_members)},
            {"Setting": "Max rows", "Value": format_number(upload.max_rows)},
            {"Setting": "Allowed extensions", "Value": ", ".join(upload.allowed_extensions)},
            {"Setting": "Normalized schema version", "Value": config.normalized_schema_version},
        ]
    )
    st.dataframe(limits, hide_index=True, use_container_width=True)

    st.subheader("Forecasting")
    ml_config = get_ml_config()
    model = ml_config.get("model", {})
    training = ml_config.get("training", {})
    gates = ml_config.get("evaluation", {})
    summary = pd.DataFrame(
        [
            {"Setting": "Architecture", "Value": model.get("model_name", "-")},
            {"Setting": "Objective", "Value": model.get("objective", "-")},
            {"Setting": "Quantiles", "Value": ", ".join(str(q) for q in model.get("quantile_alpha", []))},
            {"Setting": "Horizon (days)", "Value": model.get("forecast_horizon_days", "-")},
            {"Setting": "Target", "Value": training.get("target", "-")},
            {"Setting": "Cost basis", "Value": training.get("cost_basis", "-")},
            {"Setting": "Currency cohort", "Value": training.get("BillingCurrency", "-")},
            {"Setting": "Min history (days)", "Value": training.get("min_history_days", "-")},
            {"Setting": "Max WAPE gate", "Value": gates.get("max_wape", "-")},
            {"Setting": "Min interval coverage", "Value": gates.get("min_interval_coverage", "-")},
            {"Setting": "Max quantile crossing", "Value": gates.get("max_quantile_crossing_rate", "-")},
        ]
    )
    st.dataframe(summary, hide_index=True, use_container_width=True)
    manager = ArtifactManager()
    models = manager.list_models()
    st.caption(
        f"Model bundles: {sum(1 for m in models if m.stage == 'candidates')} candidate(s), "
        f"{sum(1 for m in models if m.stage == 'staging')} staging, "
        f"{sum(1 for m in models if m.stage == 'production')} production. "
        "Only validated production bundles are served to the dashboard."
    )

    st.subheader("Maintenance")
    if st.button("Clear dashboard data caches"):
        clear_data_cache()
        st.success("Cached dashboard queries were cleared. Data was not modified.")

    st.subheader("Reset uploaded data")
    if st.button("Clear all uploaded billing data", type="secondary"):
        store = ProcessedStore()
        repo = ImportRepository()
        for import_id in store.list_import_ids():
            store.delete(import_id)
            repo.delete(import_id)
        clear_data_cache()
        st.success("All uploaded billing data and import history were cleared.")
        st.rerun()
    st.caption(
        "CloudCost IQ does not request or store cloud provider credentials. MVP 1/2 operate on "
        "uploaded billing exports only."
    )


def _currency_index(value: str) -> int:
    options = ["USD", "EUR", "GBP", "INR", "AUD", "CAD", "JPY", "SGD", "AED", "BRL"]
    try:
        return options.index((value or "USD").upper())
    except ValueError:
        return 0


def _basis_index(value: str) -> int:
    options = ["billed", "actual", "amortized", "effective", "list"]
    try:
        return options.index((value or "billed").lower())
    except ValueError:
        return 0
