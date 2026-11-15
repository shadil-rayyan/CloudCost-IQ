# User Guide

CloudCost IQ is a local dashboard for importing multi-cloud billing exports, analyzing spend, and (when eligible) viewing validated 30-day forecasts. All processing is local to your environment unless you configure external services yourself.

## Getting started
1. Install dependencies and start the app:
   - UI: `make run-ui` or `streamlit run app/main.py` → http://localhost:8501
   - API (optional): `make run-api` → http://127.0.0.1:8000 (`/api/v1/health`)
2. The app starts in an **empty state** (no imported data, no production forecast model) unless you have existing `data/processed/` records and a promoted model artifact.
3. Import billing files via **Import Billing Data**, then explore analytics pages.

For a quick, repeatable walkthrough, use the **synthetic demo** files in `manual_test_datasets/` (see below) or run `make demo-test`.

## Pages

### Overview
- High-level KPIs, provider comparison, cost trends, top services/distribution, recent alerts/status, data freshness.
- **Empty state:** prompts you to import data.
- **Integrity:** does not present misleading combined totals across incompatible currencies/cost bases.

### Import Billing Data
- Select provider (or Auto-detect), upload supported exports (AWS/Azure/GCP/FOCUS; CSV/Parquet/GZ/ZIP), review validation/preview/warnings, and confirm import.
- Original files are retained; invalid records are reported (not silently dropped). Label demo/test uploads appropriately.

### Cost Analysis
- Filter by provider, account/subscription/project, service, region, date range, currency, and cost basis.
- View trends, breakdowns, comparisons, and detailed tables/exports. Results respect currency and cost-basis separation.

### ML Predictions
- **No production model / ineligible data:** shows a clear unavailable or insufficiency message (e.g. "Forecasting is not yet available. A validated production model has not been registered.") — never fabricated predictions.
- **With validated production model + eligible scope:** displays historical vs predicted costs, **P10/P50/P90** (30-day horizon), currency/cost basis, model version/training context, evaluation metrics/uncertainty, and errors when scope/history is incompatible.

### Anomaly Detection
- Surfaces unusual spend patterns using transparent statistical/rule logic (and forecast-aware context when a validated model is available). Entries include date/provider/account/service, expected vs observed, difference/severity/method/explanation.
- Statistical anomalies are **not** confirmed infrastructure failures.

### Cost Optimization
- Evidence-based recommendations (observation, evidence, suggested investigation, risk/limitations, priority, status such as New/Under Review/Accepted/Rejected/Completed). Estimated savings shown only when justified.
- **No automatic cloud resource changes** are performed.

### Import History
- Lists imports with provider/filename/format, timestamps, status, schema/coverage/currency/cost basis, record counts (read/valid/rejected), duplicates/warnings/errors, and lineage.

### Data Quality
- Coverage and completeness, duplicates, invalid/rejected records, missing/incomplete periods, warnings, and freshness signals (latest usage/invoice/upload). Delayed billing data is not presented as real-time.

### Settings
- Workspace and reporting preferences (e.g. display defaults). No cloud credentials are stored for MVP 1/2 file-upload workflows.

## Manual demo datasets (`manual_test_datasets/`)
Synthetic, labeled demo files for demonstrating each feature safely:

| File | Demonstrates |
|---|---|
| `01_aws_cur_valid_demo.csv` | Valid AWS import, normalization, analytics/history/quality |
| `02_azure_cost_valid_demo.csv` | Valid Azure import |
| `03_gcp_billing_valid_demo.csv` | Valid GCP import with credits/adjustments |
| `04_focus_multicloud_valid_demo.csv` | FOCUS multi-provider unified view |
| `05_validation_and_quality_issues_demo.csv` | Validation errors/warnings/rejections and quality reporting |
| `06_mixed_currency_demo.csv` | Mixed USD/EUR handled separately (no silent mixing) |
| `07_aws_cur_valid_demo.parquet` | Parquet import |
| `08_aws_cur_valid_demo.csv.gz` | Gzip CSV import |
| `09_focus_bundle_demo.zip` | ZIP/FOCUS bundle import |

Run the automated narrated demo: `make demo-test` (or `python -m pytest tests/ui/test_manual_feature_demo.py -v -s`). See `manual_test_datasets/README.md`.

## Tips & troubleshooting
- **`ModuleNotFoundError: No module named 'app'` when starting Streamlit:** run from the repo root; current `app/main.py` bootstraps the project root on `sys.path`. If needed set `PYTHONPATH` to the repo root.
- **API 404 on `/health`:** use `/api/v1/health`.
- **Forecast unavailable:** expected without a validated production model in `artifacts/models/production/` and/or without eligible historical data for the selected currency/cost basis/scope.
- **Large/sensitive files:** respect configured upload limits; do not commit private billing data or `.env`.
