# CloudCost IQ

**CloudCost IQ** is a local, Python-based multi-cloud FinOps platform for importing cloud billing exports, validating and normalizing them into a common schema, analyzing spend, and producing **validated** 30-day cost forecasts when an approved production model bundle is available.

> **Data integrity:** CloudCost IQ never fabricates real billing data, predictions, ML metrics, or savings. Synthetic/demo datasets must be explicitly labeled and are **not** evidence of production forecast accuracy. Forecasts are shown only from validated, production-promoted model artifacts with eligible data.

## Contents
- [Features and MVP status](#features-and-mvp-status)
- [Architecture](#architecture)
- [Technology stack](#technology-stack)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the application](#running-the-application)
- [Docker](#docker)
- [Billing import (AWS / Azure / GCP / FOCUS)](#billing-import-aws--azure--gcp--focus)
- [Normalized data model](#normalized-data-model)
- [Dashboard pages](#dashboard-pages)
- [API](#api)
- [Manual demo datasets and feature walkthrough](#manual-demo-datasets-and-feature-walkthrough)
- [ML training (MVP 2)](#ml-training-mvp-2)
- [Model artifacts, validation, and promotion](#model-artifacts-validation-and-promotion)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Security notes](#security-notes)
- [MVP roadmap / future enhancements](#mvp-roadmap--future-enhancements)
- [Additional documentation](#additional-documentation)

## Features and MVP status

### MVP 1 — Multi-Cloud Cost Intelligence Foundation (implemented)
- **Streamlit dashboard + FastAPI backend** (modular monolith, local-first).
- **Billing uploads & ingestion:** AWS Cost and Usage Report 2.0 / Data Exports, Azure Cost Management exports (actual/amortized/FOCUS), GCP Detailed Billing exports, and **FOCUS 1.x** files.
- **Formats:** CSV, Parquet, GZIP-compressed supported files, and ZIP bundles (with safe extraction/size/member limits). File contents are validated (not just extensions).
- **Validation & normalization:** required-field/type/date/currency/cost checks, schema/version detection, preserved raw provider fields and lineage, invalid rows reported (not silently discarded), original files retained.
- **Cost semantics:** currencies and cost bases are never mixed silently; actual/amortized/effective/billed/list contexts preserved as applicable; credits/refunds/adjustments preserved; missing/incomplete periods distinguished from zero-cost days.
- **Storage:** original uploads in `data/raw/`, normalized Parquet in `data/processed/`, rejections in `data/rejected/`, metadata in SQLite (`data/metadata/cloudcost_iq.db`), analytics via DuckDB/Parquet.
- **Analytics UI:** Overview KPIs/trends/provider-service breakdowns/freshness, Cost Analysis filters (provider/account/subscription/project/service/region/date/currency/cost basis) and breakdowns, Import History, Data Quality (coverage, duplicates, invalid/rejected, warnings, incomplete/missing periods).
- **Reporting:** tabular views/exports from the analytics UI; structured import summaries.
- **Recommendations & anomalies (transparent):** rule/statistical, evidence-based logic with explanations/limitations/statuses; **no** claim of an advanced deployed ML model when absent; **no** automatic cloud changes and no fabricated savings.

### MVP 2 — Advanced ML Cost Prediction (implemented architecture; gated by validated artifacts & data)
- **Model:** one **global multi-horizon XGBoost quantile regression** (`reg:quantileerror`) producing **P10/P50/P90** for a **30-day direct** horizon (`tree_method=hist`), trained per compatible currency/cost-basis cohort.
- **Data prep:** real normalized billing → daily panel, eligibility/sufficiency checks, leakage-safe lag/rolling/trend/calendar/horizon/categorical features, one documented target (`daily_billed_cost`, billed basis by default).
- **Evaluation:** chronological rolling-origin backtests + untouched final test; **MAE, RMSE, WAPE, pinball loss, P10–P90 interval coverage/width, quantile crossing rate**; baselines (trailing 7-day average, seasonal-naive weekly) for comparison (not additional ML models).
- **Artifacts & tracking:** versioned bundles in `artifacts/models/{candidates,staging,production}`, MLflow experiment tracking support, reproducible Colab notebook (`notebooks/colab_training.ipynb`) importing reusable `app.ml.*` modules.
- **Inference/UI:** Forecast API/page load only **validated production** artifacts, validate schema/currency/cost basis/quantile ordering, show history + P10/P50/P90 + uncertainty + model/version/metrics context, and show clear unavailable/insufficient-history messages (never fake forecasts). Forecast-based/transparent anomaly context where applicable.
- **Important:** promotion is explicit and gated (never automatic). Example candidate `cciq-xgb-q30-20261114-164550` (synthetic-demo trained) is provided in candidates for reference/demo only—not production evidence.

### MVP 3 — Production MLOps and Cloud Integration (planned / not implemented)
Model registry lifecycle automation, drift/performance monitoring, scheduled retraining, direct cloud billing/telemetry connectors (AWS/Azure/GCP/BigQuery), access hardening, CI/CD/monitoring rollout, rollback automation. **Not implemented in this MVP.**

## Architecture
Modular monolith (no unnecessary microservices):

- **Streamlit UI** (`app/main.py`, `app/pages/*`): dashboard presentation; calls reusable services/stores.
- **FastAPI** (`app/api/*`): versioned REST endpoints (`/api/v1/...`) for imports, costs, forecasts, anomalies, recommendations, health.
- **Ingestion** (`app/ingestion/*`): provider parsers (AWS/Azure/GCP/FOCUS), format detection, safe file reading, import orchestration.
- **Processing** (`app/processing/*`): validation, normalization, deduplication, aggregation, currency/cost-basis handling, data quality.
- **Analytics** (`app/analytics/*`): DuckDB + Parquet summaries, trends, breakdowns, comparisons.
- **ML** (`app/ml/*`): dataset builder, leakage-safe features, rolling-origin splits, XGBoost quantile train/eval, explainability, artifact manager, validated inference.
- **Recommendations** (`app/recommendations/*`): transparent rule/statistical suggestions.
- **Storage:** `data/raw` (originals), `data/processed` (normalized Parquet/panels), `data/rejected`, `data/metadata` (SQLite), `artifacts/models/{candidates,staging,production}`.

Colab is a **training environment only**; the app never trains on page load. See [`docs/architecture.md`](docs/architecture.md).

## Technology stack
- **Python 3.11+**, Streamlit (UI), FastAPI + Uvicorn (API)
- **Data:** Pandas, NumPy, PyArrow/Parquet, DuckDB (analytics), SQLite (metadata)
- **ML:** XGBoost (multi-quantile), scikit-learn, SHAP (where supported), Joblib, MLflow
- **Viz:** Plotly; **Config:** Pydantic Settings / YAML; **Tests:** Pytest; **Quality:** Ruff
- **Deployment:** Docker / Docker Compose (local-first)

## Repository layout
Key paths (see [`AGENTS.md`](AGENTS.md) for the full contract):
```
app/                # UI, API, ingestion, processing, analytics, ml, recommendations, core
configs/            # app_config.yaml, ml_config.yaml
data/               # raw, processed, rejected, demo, metadata (runtime; keep private data out of git)
artifacts/models/   # candidates, staging, production
notebooks/          # colab_training.ipynb
scripts/            # prepare_training_data.py, train_model.py, validate_model.py, inspect_dataset.py, ...
tests/              # unit/integration/ingestion/analytics/ml/ui/security
manual_test_datasets/ # labeled synthetic demo files for manual/UI feature demonstration
docs/               # technical and user documentation
doc/                # documentation copy/index (convenience)
Makefile            # common run/test commands
```

## Installation

### Prerequisites
- Python **3.11+** (tested with 3.12)
- `pip` and a virtual environment recommended
- (Optional) Docker & Docker Compose, `make`, Git

### Local setup
```bash
# clone/open the repository
cd CloudCost-IQ

python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt -r requirements-dev.txt
# optional ML training stack (XGBoost/MLflow/SHAP, etc.):
pip install -r requirements-ml.txt
```

Using the Makefile:
```bash
make install-dev   # app + dev/test deps
make install-ml    # optional ML deps
```

## Configuration
- **App config:** [`configs/app_config.yaml`](configs/app_config.yaml) (storage paths, upload limits/extensions, reporting defaults, quality thresholds/budgets). Paths are relative to project root unless absolute.
- **ML config:** [`configs/ml_config.yaml`](configs/ml_config.yaml) (model/objective/quantiles/horizon, hyperparameters, training cohort/eligibility, evaluation/promotion gates, artifact dirs, MLflow options).
- **Environment:** copy [`.env.example`](.env.example) to `.env` (never commit `.env`). Example overrides: `CLOUDCOST_LOG_LEVEL`, `CLOUDCOST_API_HOST`, `CLOUDCOST_API_PORT`, `CLOUDCOST_MAX_UPLOAD_MB`, `CLOUDCOST_MLFLOW_ENABLED`, `CLOUDCOST_MLFLOW_TRACKING_URI`.
- The application ensures required data/artifact directories exist at startup (`app.core.config`).

## Running the application

### Streamlit UI
```bash
streamlit run app/main.py
# or
make run-ui
```
Open `http://localhost:8501` (default). The entrypoint (`app/main.py`) adds the project root to `sys.path`, so `streamlit run app/main.py` works from the repo root without manually setting `PYTHONPATH`. If needed, you can still run with `$env:PYTHONPATH = $PWD.Path` (PowerShell) or `PYTHONPATH=$(pwd)`.

### FastAPI
```bash
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
# or
make run-api
```
- Health check: `GET http://127.0.0.1:8000/api/v1/health` (note: `/api/v1` prefix; `/health` is not routed).
- Interactive docs (FastAPI default): `/docs` and `/openapi.json` when enabled.

### Makefile quick reference
| Command | Purpose |
|---|---|
| `make help` | List targets |
| `make run-ui` | Streamlit dashboard |
| `make run-api` | FastAPI server (reload) |
| `make test` | `pytest -q` |
| `make demo-test` | Verbose manual-dataset feature demo test |
| `make lint` | `ruff check app scripts tests` |
| `make clean` | Remove caches/bytecode |

## Docker
Local containers are defined in [`docker-compose.yml`](docker-compose.yml):
- **dashboard:** `streamlit run app/main.py --server.address=0.0.0.0 --server.port=8501` (port `8501`)
- **api:** `uvicorn app.api.main:app --host 0.0.0.0 --port 8000` (port `8000`)
- Volumes: `./data`, `./artifacts`, `./configs`, `./mlruns` mounted into `/app/...`

```bash
docker compose up --build
```
Kubernetes is not part of the MVP.

## Billing import (AWS / Azure / GCP / FOCUS)

### Supported sources
| Provider / format | Examples | Notes |
|---|---|---|
| **AWS** | Cost and Usage Report 2.0 / Data Exports | CSV/Parquet (incl. gzip); preserves payer/linked, service/SKU/resource/region, usage/invoice periods, tags, credits/discounts/commitment-related fields where present |
| **Azure** | Cost Management exports | Actual/Amortized/FOCUS exports (CSV/Parquet); subscription/resource group/resource/service/meter/SKU/location/tags/currency/cost type preserved; actual vs amortized kept separate |
| **GCP** | Detailed Cloud Billing export | Billing account/project/service/SKU/resource/region, usage times, invoice month, credits, currency/adjustments preserved; upload exported CSV/Parquet (direct BigQuery integration is MVP 3) |
| **FOCUS** | FinOps Open Cost and Usage Specification files | Unified multi-provider input; record-level provider inferred when available |

**Supported FOCUS versions:** **FOCUS 1.x** (e.g. 1.0/1.1). Declared major versions outside the supported set are rejected with a clear error; schema/version metadata is recorded.

### Import workflow
1. Open **Import Billing Data**, choose provider (or Auto-detect) and cost-basis context as applicable.
2. Upload supported file (CSV/Parquet/GZ/ZIP). Contents, size, schema, and safety (archives/members/paths) are validated.
3. Review preview, validation results, warnings/errors, and coverage.
4. Confirm import to normalize and persist (original retained, processed Parquet written, metadata/lineage recorded).
5. Review outcomes in **Import History** and **Data Quality**.

### Safety & integrity rules
- Max file/uncompressed/archive limits enforced (configurable). Path traversal/unsafe archives rejected. Uploaded content is never executed.
- Duplicate/reimport/overlapping handling tracked; invalid records are reported, not silently dropped.
- Missing optional fields (e.g. some resource IDs/tags) remain `null`—not fabricated. Incomplete/missing periods are distinguished from true zero-cost days.

## Normalized data model
Normalized records include lineage/identity, provider/account/project attribution, service/SKU/resource/region, usage/charge/invoice periods, monetary fields, and preserved raw/tags. Unavailable values are `null`.

| Field | Description |
|---|---|
| `record_id`, `source_file_id`, `import_batch_id`, `schema_version` | Internal identity and lineage |
| `provider` | AWS, Azure, GCP, or FOCUS (record-level provider preserved when known) |
| `billing_account_id/name`, `linked_account_id`, `subscription_id`, `project_id/name` | Provider/account identifiers |
| `service_name`, `service_category`, `sku_id`, `sku_description`, `resource_id`, `region` | Attribution (resource may be null) |
| `usage_start/end`, `charge_period_start/end`, `invoice_month` | Time semantics preserved separately |
| `cost_amount`, `currency`, `cost_basis` | Normalized cost in original currency; bases kept separate |
| `quantity`, `unit`, `credits_amount`, `tax_amount` | Usage/adjustment details (nullable) |
| `charge_category`, `charge_type`, `pricing_model` | Normalized/provider charge context |
| `tags_json`, `raw_fields_json` | Preserved tags and original provider fields |

See [`docs/data_dictionary.md`](docs/data_dictionary.md) for details.

## Dashboard pages
| Page | Purpose |
|---|---|
| **Overview** | Spend KPIs, provider comparison, trends, top services/distribution, freshness, forecast/recommendation status; avoids misleading mixed-currency/cost-basis totals |
| **Import Billing Data** | Upload, validate/preview, confirm import |
| **Cost Analysis** | Filters (provider/account/subscription/project/service/region/date/currency/cost basis), trends, breakdowns, detailed tables |
| **ML Predictions** | Shows unavailable message until a validated production model exists; with eligible data+model shows historical vs P10/P50/P90, horizon/currency/cost basis, model version/date, metrics, uncertainty and clear errors |
| **Anomaly Detection** | Transparent statistical/rule (and forecast-aware when available) findings with evidence/severity/explanations |
| **Cost Optimization** | Evidence-based recommendations with observation, suggested investigation, risk/limitations, priority/status (no fabricated savings, no auto changes) |
| **Import History** | Upload/processing status, coverage, counts, warnings/errors, lineage |
| **Data Quality** | Coverage/completeness, duplicates, invalid/missing periods, warnings |
| **Settings** | Workspace/reporting preferences (no cloud credentials stored for MVP 1/2 file uploads) |

## API
Base path: `/api/v1`

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health/readiness + data/forecast availability |
| POST | `/imports` | Upload/import billing file (multipart) |
| GET | `/imports`, `/imports/{id}` | Import history/details |
| GET | `/costs/summary`, `/costs/timeseries`, `/costs/breakdown` | Cost analytics |
| POST | `/forecasts` | Generate forecast (validated production model + eligible scope) |
| GET | `/forecasts/{id}` | Retrieve forecast result |
| GET | `/anomalies` | List anomalies |
| GET | `/recommendations`, `PATCH /recommendations/{id}` | Recommendations and status updates |

Responses use Pydantic schemas and structured errors; sensitive paths/credentials/raw billing are not exposed.

## Manual demo datasets and feature walkthrough
Labeled **synthetic demo** files live in [`manual_test_datasets/`](manual_test_datasets/) with a manifest and [README](manual_test_datasets/README.md). Use them to demonstrate features (no real billing).

**Suggested flow (clean state):**
1. **Empty state:** Overview/History/Data Quality show no data; Forecast unavailable.
2. `01_aws_cur_valid_demo.csv` (or `07_*.parquet`, `08_*.csv.gz`): valid AWS import → KPIs, trends, breakdowns, history, data quality.
3. `04_focus_multicloud_valid_demo.csv` (or `09_focus_bundle_demo.zip`): multi-provider FOCUS view.
4. `02_azure_cost_valid_demo.csv`, `03_gcp_billing_valid_demo.csv`: Azure/GCP (credits/adjustments preserved).
5. `05_validation_and_quality_issues_demo.csv`: rejections/warnings and quality issues surfaced.
6. `06_mixed_currency_demo.csv`: USD/EUR separated (no silent mixing).
7. Review **Anomaly Detection** and **Cost Optimization** for explainable outputs.
8. **ML Predictions** remains unavailable without a promoted validated production model and eligible history.

**Automated demo (recommended for presentations):**
```bash
make demo-test
# or
python -m pytest tests/ui/test_manual_feature_demo.py -v -s
```
This runs an end-to-end, step-by-step demonstration from an isolated empty workspace using `manual_test_datasets/`.

## ML training (MVP 2)

### Requirements & principles
- Train/evaluate on **real normalized billing exports** (or approved FOCUS inputs for preparation experiments). Synthetic/demo data must not be used for production-accuracy claims.
- **Target:** daily billed cost (default), one compatible **currency** and **cost basis** per cohort/model artifact.
- **Eligibility (configurable):** e.g. 180+ days history (365 preferred), sufficient observed coverage/series, 3 rolling-origin validation folds + untouched 30-day final test. Missing days are **not** converted to zero.
- **Leakage prevention:** all lags/rolling/encodings derived from data available at forecast origin; final test never used for selection/tuning.

### Feature engineering (leakage-safe)
- **Lags:** 1, 7, 14, 28 days
- **Rolling (7d/28d):** mean, median, std, sum
- **Trend/change:** recent vs historical/short-long changes (as implemented)
- **Calendar:** day-of-week/weekend/day-of-month/week/month/quarter, target-date features
- **Forecast context:** horizon (1..30), provider/service/account/category/region context as available (no unjustified high-cardinality IDs)

### Model
- **Single global XGBoost multi-quantile** model: `objective=reg:quantileerror`, `quantile_alpha=[0.10,0.50,0.90]`, `tree_method=hist`, direct 30-horizon forecasting (no recursive lag feedback). Quantile ordering/crossing checked and handled per artifact contract.

### Evaluation
- Chronological rolling-origin backtests + untouched final test.
- **Metrics:** MAE, RMSE, WAPE, pinball loss (P10/P50/P90), interval coverage/width, quantile crossing rate; reported overall and by relevant slices where available.
- **Baselines (benchmarks):** trailing 7-day average, seasonal-naive weekly. Do not claim improvement without measured out-of-sample results.

### Local scripts
```bash
python scripts/prepare_training_data.py --currency USD --cost-basis billed
python scripts/inspect_dataset.py --panel data/processed/ml/daily_panel.parquet
python scripts/train_model.py --panel data/processed/ml/daily_panel.parquet
python scripts/validate_model.py --version <version>            # gate checks
python scripts/validate_model.py --version <version> --promote # explicit promotion to production
```

### Google Colab
Use [`notebooks/colab_training.ipynb`](notebooks/colab_training.ipynb). It checks out the repo/ref, installs deps, loads data (Drive/upload/synthetic_demo option for notebook runs only—**synthetic is demo-only**), runs the reusable `app.ml.*` pipeline, tracks experiments, evaluates, and exports a candidate bundle. Promote only after gates pass via `scripts/validate_model.py --promote`. See [`docs/ml_training_guide.md`](docs/ml_training_guide.md) and [`docs/ml_pipeline.md`](docs/ml_pipeline.md).

## Model artifacts, validation, and promotion
Artifacts are stored under `artifacts/models/{candidates,staging,production}/<version>/` and must include the required bundle (e.g. native model, feature schema/order, model config, metrics, training manifest, dataset quality, model card; checksums as produced). Inference requires a **compatible, validated production** bundle matching currency/cost basis and schema.

- Validate: `python scripts/validate_model.py --version <version>`
- Promote (explicit): `python scripts/validate_model.py --version <version> --promote`
- No auto-promotion; failed gates or invalid artifacts must not be served.

Forecast UI/API report availability, model/version context, and clear reasons when unavailable. See [`docs/model_card.md`](docs/model_card.md) and the example candidate bundle metadata.

## Testing
```bash
make test
# or
python -m pytest -q

# lint
make lint
# or
python -m ruff check app scripts tests

# focused/manual demo
python -m pytest tests/ui/test_manual_feature_demo.py -v -s
```
Tests cover ingestion/parsing, normalization/analytics behavior, ML dataset/inference contracts, API/UI smoke flows, and security/safety cases as included in `tests/` (CI should not depend on private billing data).

## Known limitations
- **Forecasts** depend on real, sufficiently complete, and recent billing history; delayed/revised exports affect quality. Intervals are estimates and may be miscalibrated under regime shifts.
- No CPU/memory telemetry-driven rightsizing, no automatic cloud resource changes, no undocumented/silent FX conversion or mixed-currency totals.
- GCP direct BigQuery connector, automated cloud pulls, drift monitoring, retraining automation, and full model-registry lifecycle are **MVP 3** (not implemented).
- The included candidate model bundle in this repo context may be **synthetic-demo** trained; treat accordingly (not for real-world/production accuracy claims).

## Security notes
- Billing exports may be sensitive: enforce upload limits, safe parsing/extraction, isolated storage, and avoid logging secrets or raw sensitive records.
- Never commit `.env`, credentials, private billing data, or large/private artifacts. MVP 1/2 file-upload flows do not require cloud credentials.
- MVP 3 cloud integrations (future) must use least-privilege, preferably short-lived/read-only access.

## MVP roadmap / future enhancements
- **MVP 1 (complete):** end-to-end multi-cloud ingest/normalize/analytics/quality/history/reporting with transparent recommendations.
- **MVP 2 (complete as gated ML system):** global XGBoost quantile 30-day forecasting, reproducible training/eval, artifacted & validated inference, explainability hooks.
- **MVP 3 (future):** registry/version lifecycle + rollback automation, drift/perf monitoring, scheduled retraining, direct AWS/Azure/GCP BigQuery ingestion, telemetry joins, access/CI/CD/observability hardening.

## Additional documentation
- [`docs/architecture.md`](docs/architecture.md)
- [`docs/data_dictionary.md`](docs/data_dictionary.md)
- [`docs/dataset_sources.md`](docs/dataset_sources.md)
- [`docs/ingestion_guide.md`](docs/ingestion_guide.md)
- [`docs/ml_training_guide.md`](docs/ml_training_guide.md)
- [`docs/ml_pipeline.md`](docs/ml_pipeline.md)
- [`docs/model_card.md`](docs/model_card.md)
- [`docs/user_guide.md`](docs/user_guide.md)
- Convenience copy: [`doc/`](doc/)
- Project contract: [`AGENTS.md`](AGENTS.md)
