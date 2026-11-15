"""Streamlit UI smoke/E2E tests covering core pages and flows via AppTest."""

from __future__ import annotations

import shutil
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest

PAGES = [
    ("overview", "app.pages.overview"),
    ("upload", "app.pages.upload"),
    ("cost_analysis", "app.pages.cost_analysis"),
    ("forecast", "app.pages.forecast"),
    ("anomalies", "app.pages.anomalies"),
    ("recommendations", "app.pages.recommendations"),
    ("import_history", "app.pages.import_history"),
    ("data_quality", "app.pages.data_quality"),
    ("settings", "app.pages.settings"),
]


def _seed_workspace_with_demo() -> None:
    """Copy demo processed data and promoted production model into isolated test workspace."""
    from app.core.config import get_app_config

    cfg = get_app_config()
    cfg.ensure_directories()

    src_processed = PROJECT_ROOT / "data" / "processed"
    if src_processed.exists():
        for p in src_processed.glob("*.parquet"):
            shutil.copy2(p, cfg.storage.processed_dir / p.name)

    src_db = PROJECT_ROOT / "data" / "metadata" / "cloudcost_iq.db"
    if src_db.exists():
        cfg.storage.database_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_db, cfg.storage.database_path)

    src_prod = PROJECT_ROOT / "artifacts" / "models" / "production"
    dst_prod = cfg.storage.artifacts_dir / "production"
    dst_prod.mkdir(parents=True, exist_ok=True)
    if src_prod.exists():
        for model_dir in src_prod.iterdir():
            if model_dir.is_dir() and not (dst_prod / model_dir.name).exists():
                shutil.copytree(model_dir, dst_prod / model_dir.name)


def _run_page_script(module_name: str, tmp_path: Path, *, seed_demo: bool = False) -> AppTest:
    if seed_demo:
        _seed_workspace_with_demo()
    script = tmp_path / f"run_{module_name.split('.')[-1]}.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys
            from pathlib import Path
            ROOT = Path(r"{PROJECT_ROOT}")
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from {module_name} import render
            render()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    at = AppTest.from_file(str(script), default_timeout=60)
    at.run()
    return at


def _assert_no_exceptions(at: AppTest) -> None:
    assert len(at.exception) == 0, f"UI exceptions: {[getattr(e, 'value', e) for e in at.exception]}"


@pytest.mark.parametrize("name,module_name", PAGES)
def test_pages_render_without_exceptions(name: str, module_name: str, tmp_path: Path) -> None:
    at = _run_page_script(module_name, tmp_path, seed_demo=False)
    _assert_no_exceptions(at)


def test_main_app_entrypoint_runs(tmp_path: Path) -> None:
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "main.py"), default_timeout=60)
    at.run()
    _assert_no_exceptions(at)


def test_overview_with_demo_data(tmp_path: Path) -> None:
    at = _run_page_script("app.pages.overview", tmp_path, seed_demo=True)
    _assert_no_exceptions(at)
    # Expect meaningful UI output when data is present
    charts = len(getattr(at, "plotly_chart", []))
    assert (len(at.metric) + len(at.title) + len(at.header) + len(at.subheader) + len(at.dataframe) + charts) >= 1


def test_cost_analysis_with_demo_data(tmp_path: Path) -> None:
    at = _run_page_script("app.pages.cost_analysis", tmp_path, seed_demo=True)
    _assert_no_exceptions(at)


def test_import_history_with_demo_data(tmp_path: Path) -> None:
    at = _run_page_script("app.pages.import_history", tmp_path, seed_demo=True)
    _assert_no_exceptions(at)


def test_data_quality_with_demo_data(tmp_path: Path) -> None:
    at = _run_page_script("app.pages.data_quality", tmp_path, seed_demo=True)
    _assert_no_exceptions(at)


def test_forecast_with_promoted_model(tmp_path: Path) -> None:
    """With seeded data and a promoted (validated) production model, forecast page should render without errors."""
    at = _run_page_script("app.pages.forecast", tmp_path, seed_demo=True)
    _assert_no_exceptions(at)
    # Page should present forecast/model information or controls
    all_text = " ".join(
        str(getattr(el, "value", getattr(el, "body", "")))
        for group in (at.markdown, at.info, at.success, at.warning, at.caption, at.title, at.header, at.subheader)
        for el in group
    ).lower()
    assert "forecast" in all_text or "model" in all_text or len(at.button) + len(at.selectbox) > 0
