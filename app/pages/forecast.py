"""ML Predictions: validated XGBoost quantile forecasts only.

This page never fabricates a forecast. When no production bundle has passed the
promotion gates it explains what is required and keeps the analytics pages
usable.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.config import get_ml_config
from app.core.exceptions import CloudCostError
from app.database.repositories import ForecastRepository
from app.ml.predict import UNAVAILABLE_MESSAGE, ForecastRequest, ForecastService
from app.ui.components import (
    cached_dimensions,
    download_csv,
    empty_state,
    page_header,
)
from app.utils.formatting import format_currency, format_number, format_percent


def render() -> None:
    page_header(
        "ML Predictions",
        "30-day daily cost forecasts with P10/P50/P90 uncertainty from a validated global "
        "XGBoost quantile model. Forecasts are only shown when a registered production bundle "
        "passes every validation gate.",
    )
    service = ForecastService()
    availability = service.availability()
    if not availability.available:
        _render_unavailable(service, availability.reason)
        return
    _render_model_summary(service, availability)
    _render_generate(service, availability)
    _render_history()


def _render_unavailable(service: ForecastService, reason: str) -> None:
    st.warning(reason or UNAVAILABLE_MESSAGE)
    st.caption(
        "The cost analysis, anomaly and recommendation pages continue to work without a trained "
        "model. No sample or randomly generated prediction is displayed."
    )

    dimensions = cached_dimensions()
    ml_config = get_ml_config()
    training = ml_config.get("training", {})
    columns = st.columns(4)
    columns[0].metric("Billing records", format_number(dimensions.records))
    columns[1].metric("Currency cohorts", len(dimensions.currencies) or 0)
    columns[2].metric("Cost bases", len(dimensions.cost_bases) or 0)
    if dimensions.date_min and dimensions.date_max:
        history_days = (dimensions.date_max - dimensions.date_min).days + 1
        columns[3].metric("Calendar history", f"{history_days} days")
    else:
        columns[3].metric("Calendar history", "-")

    st.subheader("Model readiness requirements")
    st.markdown(
        "\n".join(
            [
                f"- **Frequency:** daily billing aggregation (the model target is `{training.get('target', 'daily_billed_cost')}`).",
                f"- **Minimum history:** {training.get('min_history_days', 180)} calendar days "
                f"(preferred {training.get('preferred_history_days', 365)}).",
                f"- **Minimum observed daily records:** {training.get('min_observations', 120)} per series.",
                f"- **Horizon:** {ml_config.get('model', {}).get('forecast_horizon_days', 30)} days, direct multi-horizon.",
                "- **Currency and cost basis:** one compatible cohort per model artifact.",
                "- **Evaluation:** rolling-origin backtesting, baselines, pinball loss and interval coverage.",
            ]
        )
    )
    st.info(
        "Train a candidate in `notebooks/colab_training.ipynb` or with `scripts/train_model.py`, "
        "then validate and promote it with `scripts/validate_model.py`."
    )

    models = service.model_options()
    if models:
        st.subheader("Registered model bundles")
        st.dataframe(
            pd.DataFrame(models)[
                [
                    "version",
                    "stage",
                    "currency",
                    "cost_basis",
                    "horizon_days",
                    "wape",
                    "coverage",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Candidates and staging bundles are not served. Promotion requires an explicit "
            "operator action after the validation gates pass."
        )
    else:
        empty_state("No candidate model bundles exist yet.")


def _render_model_summary(service: ForecastService, availability) -> None:
    model_config = availability.model_config or {}
    manifest = availability.manifest or {}
    metrics = availability.metrics or {}
    final = metrics.get("final_test", {})
    comparison = metrics.get("baseline_comparison", {})

    st.success("A validated production model is registered and available for inference.")
    columns = st.columns(5)
    columns[0].metric("Model version", availability.version or "-")
    columns[1].metric("Stage", availability.stage or "-")
    columns[2].metric("Horizon", f"{model_config.get('forecast_horizon_days', '-')} days")
    columns[3].metric("Currency / basis", f"{model_config.get('currency', '-')} / {model_config.get('cost_basis', '-')}")
    columns[4].metric(
        "Out-of-sample WAPE",
        format_percent(final.get("wape")) if final.get("wape") is not None else "-",
    )

    st.caption(
        f"Trained {str(manifest.get('trained_at') or model_config.get('trained_at') or '-')[:19]} on "
        f"data from {manifest.get('dataset_start', '?')} to {manifest.get('dataset_end', '?')}. "
        f"Target `{model_config.get('target', '-')}`, quantiles {model_config.get('quantile_alpha', [])}."
    )

    with st.expander("Model performance and validation", expanded=False):
        metrics_columns = st.columns(4)
        metrics_columns[0].metric("MAE", format_number(final.get("mae"), 2))
        metrics_columns[1].metric("RMSE", format_number(final.get("rmse"), 2))
        metrics_columns[2].metric("P10-P90 coverage", format_percent(final.get("interval_coverage")))
        metrics_columns[3].metric("Mean interval width", format_number(final.get("interval_width"), 2))
        st.caption(
            "All metrics are out-of-sample from chronological rolling-origin backtesting. "
            "Pinball loss (mean): "
            f"{format_number(final.get('pinball_mean'), 4)}; quantile crossing rate: "
            f"{format_percent(final.get('quantile_crossing_rate'), 2)}."
        )
        if comparison:
            baseline_rows = [
                {
                    "Baseline": "Trailing 7-day average",
                    "WAPE": comparison.get("wape_trailing_7d_average"),
                    "Improvement vs model": comparison.get("wape_improvement_vs_trailing_7d_average"),
                },
                {
                    "Baseline": "Seasonal-naive weekly",
                    "WAPE": comparison.get("wape_seasonal_naive_weekly"),
                    "Improvement vs model": comparison.get("wape_improvement_vs_seasonal_naive_weekly"),
                },
            ]
            st.dataframe(pd.DataFrame(baseline_rows), hide_index=True, use_container_width=True)
        folds = metrics.get("folds", [])
        if folds:
            st.markdown("**Rolling-origin folds**")
            fold_frame = pd.DataFrame(folds)
            visible = [
                column
                for column in ("fold", "validation_start", "validation_end", "rows", "wape", "mae", "interval_coverage", "rounds")
                if column in fold_frame.columns
            ]
            st.dataframe(fold_frame[visible], hide_index=True, use_container_width=True)

    with st.expander("Model card", expanded=False):
        st.markdown(availability.model_card or "No model card was included in the bundle.")
        manifest_rows = {
            "Dataset fingerprint": manifest.get("dataset_fingerprint"),
            "Git commit": manifest.get("git_commit"),
            "Rows": manifest.get("rows"),
            "Series": manifest.get("series_count"),
            "Providers": ", ".join(manifest.get("providers", [])),
            "Feature schema version": manifest.get("feature_schema_version"),
        }
        st.json({key: value for key, value in manifest_rows.items() if value is not None})
        if manifest.get("package_versions"):
            st.json(manifest["package_versions"])


def _render_generate(service: ForecastService, availability) -> None:
    st.subheader("Generate a forecast")
    dimensions = cached_dimensions()
    model_config = availability.model_config or {}
    currency = model_config.get("currency", "USD")
    basis = model_config.get("cost_basis", "billed")

    controls = st.columns(4)
    with controls[0]:
        provider_choice = st.selectbox(
            "Provider", ["(all providers)"] + dimensions.providers, key="forecast_provider"
        )
    with controls[1]:
        account_choice = st.selectbox(
            "Billing account", ["(all accounts)"] + dimensions.billing_accounts, key="forecast_account"
        )
    with controls[2]:
        service_choice = st.selectbox(
            "Service", ["(all services)"] + dimensions.services, key="forecast_service"
        )
    with controls[3]:
        horizon = st.slider(
            "Horizon (days)",
            min_value=1,
            max_value=int(model_config.get("forecast_horizon_days", 30)),
            value=int(model_config.get("forecast_horizon_days", 30)),
            key="forecast_horizon",
        )

    past_origin = st.checkbox(
        "Backtest from a past origin date",
        value=False,
        help="Generate the forecast as it would have been produced at an earlier date, so actuals "
        "after that origin can be compared with the forecast interval.",
    )
    origin_date: dt.date | None = None
    if past_origin and dimensions.date_min and dimensions.date_max:
        origin_date = st.date_input(
            "Forecast origin",
            value=dimensions.date_max - dt.timedelta(days=int(model_config.get("forecast_horizon_days", 30))),
            min_value=dimensions.date_min,
            max_value=dimensions.date_max,
            key="forecast_origin",
        )

    st.caption(
        f"Forecasts are restricted to the model cohort: {currency} on the {basis} cost basis. "
        "Selecting an incompatible cohort is refused rather than converted."
    )
    generate = st.button("Generate forecast", type="primary", key="forecast_generate")
    if generate:
        request = ForecastRequest(
            provider=None if provider_choice.startswith("(") else provider_choice,
            billing_account_id=None if account_choice.startswith("(") else account_choice,
            service_name=None if service_choice.startswith("(") else service_choice,
            currency=currency,
            cost_basis=basis,
            horizon_days=horizon,
            origin_date=origin_date if isinstance(origin_date, dt.date) else None,
        )
        try:
            result = service.generate(request, persist=True)
        except CloudCostError as exc:
            st.error(str(exc))
            return
        except Exception:  # pragma: no cover - defensive UI guard
            st.error("The forecast could not be generated. Check the data coverage and model version.")
            return
        st.session_state["latest_forecast"] = result
        st.success(f"Forecast {result.forecast_id} generated with model {result.model_version}.")

    result = st.session_state.get("latest_forecast")
    if result is not None:
        _render_forecast_result(result)


def _render_forecast_result(result) -> None:
    st.divider()
    st.subheader("Forecast result")
    st.caption(
        f"Model {result.model_version} ({result.model_stage}) | generated "
        f"{result.generated_at.isoformat()[:19]} | origin {result.origin_date} | "
        f"scope {result.summary.get('series_count', 0)} series"
    )
    for warning in result.warnings:
        st.warning(warning)

    totals = result.totals()
    columns = st.columns(3)
    columns[0].metric(f"P10 total ({result.currency})", format_currency(totals.get("p10"), result.currency))
    columns[1].metric(f"P50 total ({result.currency})", format_currency(totals.get("p50"), result.currency))
    columns[2].metric(f"P90 total ({result.currency})", format_currency(totals.get("p90"), result.currency))
    crossing = result.summary.get("quantile_crossing", {})
    if crossing.get("rows"):
        st.caption(
            f"{crossing['rows']} row(s) had crossing quantiles and were reordered so "
            "P10 <= P50 <= P90 before display."
        )

    _forecast_chart(result)

    st.subheader("Daily forecast")
    daily = (
        result.rows.groupby("target_date", as_index=False)[["p10", "p50", "p90"]]
        .sum()
        .sort_values("target_date")
    )
    st.dataframe(
        daily.rename(
            columns={
                "target_date": "Date",
                "p10": f"P10 ({result.currency})",
                "p50": f"P50 ({result.currency})",
                "p90": f"P90 ({result.currency})",
            }
        ),
        hide_index=True,
        use_container_width=True,
        height=320,
    )
    download_csv(result.rows, f"forecast_{result.forecast_id}.csv", "Download forecast CSV")

    _breakdown(result, "by_provider", "Provider")
    _breakdown(result, "by_service", "Service", limit=15)
    _breakdown(result, "by_account", "Billing account", limit=15)

    with st.expander("Model inputs and provenance", expanded=False):
        st.json(
            {
                "model_version": result.model_version,
                "model_stage": result.model_stage,
                "origin_date": str(result.origin_date),
                "horizon_days": result.horizon_days,
                "currency": result.currency,
                "cost_basis": result.cost_basis,
                "quantile_alpha": result.model_config.get("quantile_alpha"),
                "target": result.model_config.get("target"),
                "results_path": result.results_path,
            }
        )


def _forecast_chart(result) -> None:
    daily = (
        result.rows.groupby("target_date", as_index=False)[["p10", "p50", "p90"]]
        .sum()
        .sort_values("target_date")
    )
    if daily.empty:
        return
    figure = go.Figure()
    history = result.history.copy()
    if not history.empty:
        history["date"] = pd.to_datetime(history["date"])
        figure.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["cost"],
                name="Historical daily cost",
                line={"color": "#0f172a", "width": 2},
            )
        )
    x = pd.to_datetime(daily["target_date"])
    figure.add_trace(
        go.Scatter(
            x=x,
            y=daily["p90"],
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=daily["p10"],
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.18)",
            line={"width": 0},
            name="P10-P90 interval",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=daily["p50"],
            name="P50 forecast",
            line={"color": "#2563eb", "width": 2, "dash": "dash"},
        )
    )
    figure.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        yaxis_title=f"Daily cost ({result.currency})",
        xaxis_title=None,
        legend={"orientation": "h", "y": 1.08},
        hovermode="x unified",
    )
    st.plotly_chart(figure, use_container_width=True)


def _breakdown(result, key: str, label: str, limit: int = 10) -> None:
    rows = result.summary.get(key, [])
    if not rows:
        return
    st.markdown(f"**{label} breakdown**")
    frame = pd.DataFrame(rows).head(limit)
    label_column = frame.columns[0]
    frame = frame.rename(
        columns={
            label_column: label,
            "p10": f"P10 ({result.currency})",
            "p50": f"P50 ({result.currency})",
            "p90": f"P90 ({result.currency})",
        }
    )
    st.dataframe(frame, hide_index=True, use_container_width=True)


def _render_history() -> None:
    st.subheader("Forecast history")
    records = ForecastRepository().list(limit=25)
    if not records:
        st.caption("No forecasts have been generated yet.")
        return
    rows = [
        {
            "forecast_id": record.forecast_id,
            "generated_at": record.generated_at.isoformat()[:19],
            "model_version": record.model_version,
            "scope": record.metrics.get("scope") if record.metrics else None,
            "currency": record.currency,
            "cost_basis": record.cost_basis,
            "horizon_days": record.horizon_days,
            "p50_total": (record.metrics.get("totals", {}) or {}).get("p50"),
        }
        for record in records
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
