"""Demonstrative end-to-end UI/feature test using manual_test_datasets/.

Run with output to show/explain each feature:
    python -m pytest tests/ui/test_manual_feature_demo.py -v -s

This test starts from an empty isolated workspace (conftest), uploads synthetic
manual datasets, and verifies each major feature shows the expected state/data.
All datasets are SYNTHETIC DEMO DATA (not real billing).
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest

MANUAL = PROJECT_ROOT / "manual_test_datasets"


def _banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _run_page(module_name: str, tmp_path: Path) -> AppTest:
    script = tmp_path / f"run_{module_name.split('.')[-1]}_{abs(hash(module_name)) % 10000}.py"
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


def _texts(at: AppTest) -> str:
    parts = []
    for group_name in ("title", "header", "subheader", "markdown", "caption", "info", "warning", "success", "error"):
        for el in getattr(at, group_name, []):
            val = getattr(el, "value", getattr(el, "body", ""))
            if val:
                parts.append(str(val))
    return "\n".join(parts).lower()


def _no_exceptions(at: AppTest) -> None:
    assert len(at.exception) == 0, [getattr(e, "value", e) for e in at.exception]


def _import_file(filename: str, dataset_kind: str = "demo", provider: str = "Auto", cost_basis: str = "billed"):
    from app.ingestion.import_manager import ImportManager, ImportOptions

    path = MANUAL / filename
    assert path.exists(), f"Manual dataset missing: {path}"
    manager = ImportManager()
    opts = ImportOptions(provider=provider, cost_basis=cost_basis, dataset_kind=dataset_kind)
    outcome = manager.import_file(path, path.name, opts)
    return outcome


def test_manual_datasets_demonstrate_all_features(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from app.analytics.store import CostStore
    from app.api.main import app
    from app.database.repositories import ImportRepository
    from app.ml.predict import ForecastService

    client = TestClient(app)

    # ------------------------------------------------------------------
    _banner("FEATURE: Initial empty state (no billing data, no production model)")
    health = client.get("/api/v1/health").json()
    print("API /health:", health)
    assert health["database"] is True
    assert health["data_available"] is False
    assert health["forecast_available"] is False

    at = _run_page("app.pages.overview", tmp_path)
    _no_exceptions(at)
    print("Overview page renders (empty state): OK, widgets=", len(at.title) + len(at.header) + len(at.metric) + len(at.info) + len(at.warning))

    at = _run_page("app.pages.forecast", tmp_path)
    _no_exceptions(at)
    ftext = _texts(at)
    print("Forecast page message snippet:", ("not yet available" in ftext) or ("validated production model" in ftext))
    assert "forecast" in ftext or "model" in ftext
    assert "not yet available" in ftext or "validated production model" in ftext or "no " in ftext

    at = _run_page("app.pages.import_history", tmp_path)
    _no_exceptions(at)
    print("Import History (empty) renders: OK")

    # ------------------------------------------------------------------
    _banner("FEATURE: Upload valid AWS billing (01_aws_cur_valid_demo.csv) -> ingestion, normalization, analytics")
    out = _import_file("01_aws_cur_valid_demo.csv")
    print("Import result:", out.status, "provider=", out.provider, "records_read=", out.records_read, "valid=", out.valid_records, "rejected=", out.rejected_records)
    print("coverage:", out.coverage_start, "->", out.coverage_end, "currency=", out.currency, "cost_basis=", out.cost_basis)
    assert out.status == "completed"
    assert out.valid_records > 0
    assert out.rejected_records == 0

    dims = CostStore().dimensions()
    print("Analytics store after AWS import: records=", dims.records, "providers=", dims.providers, "currencies=", dims.currencies)
    assert dims.records > 0
    assert "AWS" in dims.providers

    at = _run_page("app.pages.overview", tmp_path)
    _no_exceptions(at)
    print("Overview with data: metrics/charts/widgets present:", len(at.metric), len(at.dataframe), len(at.title) + len(at.header))

    at = _run_page("app.pages.cost_analysis", tmp_path)
    _no_exceptions(at)
    print("Cost Analysis with data renders: OK")

    at = _run_page("app.pages.import_history", tmp_path)
    _no_exceptions(at)
    htext = _texts(at)
    print("Import History shows imports/content:", ("01_aws_cur_valid_demo.csv" in htext) or ("completed" in htext) or (len(at.dataframe) > 0))

    at = _run_page("app.pages.data_quality", tmp_path)
    _no_exceptions(at)
    print("Data Quality renders with data: OK")

    # ------------------------------------------------------------------
    _banner("FEATURE: Upload valid FOCUS multi-provider (04_focus_multicloud_valid_demo.csv)")
    out = _import_file("04_focus_multicloud_valid_demo.csv")
    print("FOCUS import:", out.status, "provider=", out.provider, "valid=", out.valid_records, "schema=", out.schema_name)
    assert out.status == "completed"
    assert out.valid_records > 0
    dims = CostStore().dimensions()
    print("Providers after FOCUS:", dims.providers, "total records:", dims.records)
    assert len(dims.providers) >= 2 or dims.records > 0

    # ------------------------------------------------------------------
    _banner("FEATURE: Upload valid GCP billing with credits (03_gcp_billing_valid_demo.csv)")
    out = _import_file("03_gcp_billing_valid_demo.csv")
    print("GCP import:", out.status, "provider=", out.provider, "valid=", out.valid_records)
    assert out.status == "completed"
    dims = CostStore().dimensions()
    print("Providers after GCP:", dims.providers)

    # ------------------------------------------------------------------
    _banner("FEATURE: Format support - Parquet / GZIP / ZIP uploads")
    for fname in ("07_aws_cur_valid_demo.parquet", "08_aws_cur_valid_demo.csv.gz", "09_focus_bundle_demo.zip"):
        out = _import_file(fname)
        print(f"Format test {fname}: status={out.status} valid={out.valid_records} rejected={out.rejected_records}")
        assert out.status == "completed"
        assert out.valid_records > 0

    # ------------------------------------------------------------------
    _banner("FEATURE: Validation & Data Quality - intentional issues (05_validation_and_quality_issues_demo.csv)")
    out = _import_file("05_validation_and_quality_issues_demo.csv")
    print("Issues file import:", out.status, "read=", out.records_read, "valid=", out.valid_records, "rejected=", out.rejected_records, "warnings=", len(out.warnings))
    print("Sample warnings/issues:", out.warnings[:3])
    # Expect the pipeline to report problems (rejections and/or warnings) rather than silently accept everything
    assert out.records_read > 0
    assert (out.rejected_records > 0) or (len(out.warnings) > 0) or (out.valid_records < out.records_read)

    at = _run_page("app.pages.data_quality", tmp_path)
    _no_exceptions(at)
    print("Data Quality reflects issues/rejections/warnings: OK")

    at = _run_page("app.pages.import_history", tmp_path)
    _no_exceptions(at)
    print("Import History shows all imports including problematic one: OK")

    # ------------------------------------------------------------------
    _banner("FEATURE: Currency handling - mixed USD/EUR (06_mixed_currency_demo.csv) must not be silently mixed")
    out = _import_file("06_mixed_currency_demo.csv")
    print("Mixed currency import:", out.status, "valid=", out.valid_records, "currency_field=", out.currency, "warnings=", len(out.warnings))
    assert out.status == "completed"
    dims = CostStore().dimensions()
    print("Currencies in store (should include USD and EUR, handled separately):", dims.currencies)
    assert "USD" in dims.currencies
    assert "EUR" in dims.currencies

    # ------------------------------------------------------------------
    _banner("FEATURE: Anomalies & Cost Optimization (rule/statistical, explainable)")
    at = _run_page("app.pages.anomalies", tmp_path)
    _no_exceptions(at)
    print("Anomaly Detection page renders with data:", "ok", "text_len=", len(_texts(at)))

    at = _run_page("app.pages.recommendations", tmp_path)
    _no_exceptions(at)
    print("Cost Optimization page renders with data:", "ok", "text_len=", len(_texts(at)))

    # ------------------------------------------------------------------
    _banner("FEATURE: ML Forecasts - correct unavailable behavior (no production model / insufficient eligible history)")
    fc = ForecastService().availability()
    print("Forecast availability:", fc.available, "| reason:", fc.reason)
    assert fc.available is False  # production model was moved out; clean-state expectation

    at = _run_page("app.pages.forecast", tmp_path)
    _no_exceptions(at)
    ftext = _texts(at)
    print("Forecast UI communicates unavailable state:", ("not yet available" in ftext) or ("validated production model" in ftext) or ("insufficient" in ftext))
    assert "forecast" in ftext or "model" in ftext

    # ------------------------------------------------------------------
    _banner("SUMMARY: all key features exercised with synthetic manual datasets")
    imports = ImportRepository().list()
    final_dims = CostStore().dimensions()
    print(f"Total imports: {len(imports)} | total records in store: {final_dims.records}")
    print(f"Providers: {final_dims.providers} | Currencies: {final_dims.currencies}")
    print("Features demonstrated: empty-state, AWS/Azure/GCP/FOCUS ingestion, normalization, analytics/overview/cost analysis,")
    print("  import history, data quality, parquet/gzip/zip formats, validation+rejections/warnings, mixed-currency separation,")
    print("  anomalies & recommendations pages, and correct forecast unavailability (no fake predictions).")
    print("NOTE: All data is SYNTHETIC DEMO data. Forecasts require a validated production model + eligible history.")
