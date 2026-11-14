"""CloudCost IQ Streamlit entrypoint.

Run from the project root with::

    streamlit run app/main.py

Page modules live in :mod:`app.pages`. They expose a ``render()`` function and
are wired here explicitly with :func:`st.navigation`, so Streamlit's automatic
``pages/`` discovery does not run the modules as bare scripts.
"""

from __future__ import annotations

import streamlit as st

from app.core.config import get_app_config
from app.core.logging import configure_logging
from app.pages import (
    anomalies,
    cost_analysis,
    data_quality,
    forecast,
    import_history,
    overview,
    recommendations,
    settings,
    upload,
)


def _configure() -> None:
    config = get_app_config()
    config.ensure_directories()
    configure_logging()
    st.set_page_config(
        page_title=config.app_name,
        page_icon=":material/cloud:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def main() -> None:
    _configure()
    config = get_app_config()

    pages = [
        st.Page(overview.render, title="Overview", url_path="overview", default=True),
        st.Page(upload.render, title="Import Billing Data", url_path="import"),
        st.Page(cost_analysis.render, title="Cost Analysis", url_path="cost-analysis"),
        st.Page(forecast.render, title="ML Predictions", url_path="forecast"),
        st.Page(anomalies.render, title="Anomaly Detection", url_path="anomalies"),
        st.Page(recommendations.render, title="Cost Optimization", url_path="recommendations"),
        st.Page(import_history.render, title="Import History", url_path="import-history"),
        st.Page(data_quality.render, title="Data Quality", url_path="data-quality"),
        st.Page(settings.render, title="Settings", url_path="settings"),
    ]

    st.sidebar.markdown(
        f"<div style='padding-top:0.5rem;color:#64748b;font-size:0.8rem;'>"
        f"{config.app_name} v{config.app_version}</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        "Billing data is analyzed locally. Nothing is modified in your cloud accounts."
    )
    navigation = st.navigation({"CloudCost IQ": pages}, position="sidebar")
    navigation.run()


if __name__ == "__main__":
    main()
