# Architecture

CloudCost IQ is a **modular monolith** (Python) designed to run locally first, with clear separation between ingestion, processing, analytics, ML, and presentation. No unnecessary microservices.

## High-level flow
```
User -> Streamlit UI (app/main.py, app/pages/*)
            |
            v
         FastAPI (/api/v1, app/api/*)
            |
   +--------+---------+------------------+
   |        |         |                  |
Ingestion Processing Analytics            ML
(parsers,  (validate, (DuckDB+Parquet    (dataset/features,
 detect,   normalize,  summaries/trends/   train/eval/infer,
 import)   dedupe,     breakdowns)         artifacts)
           quality)         |                  |
               \           |                  /
                \          v                 v
                 +--> Normalized data <--- Model artifacts
                                |
                                v
                        Recommendations
                                |
                                v
                            Dashboard/API
```

## Components

### UI — `app/`
- `app/main.py`: Streamlit entrypoint, navigation, and app bootstrap (ensures project root on `sys.path`, config dirs, logging, page wiring via `st.navigation`).
- `app/pages/*`: Overview, Import Billing Data, Cost Analysis, ML Predictions, Anomaly Detection, Cost Optimization, Import History, Data Quality, Settings. Pages render application state or explicit empty/unavailable states (no fabricated predictions).

### API — `app/api/`
- FastAPI app with versioned routes under `/api/v1` (`health`, `imports`, `costs`, `forecasts`, `anomalies`, `recommendations`).
- Pydantic request/response models; structured errors; no exposure of internal paths/credentials/raw sensitive billing.

### Ingestion — `app/ingestion/`
- Provider/format detection and parsers: AWS, Azure, GCP, FOCUS 1.x (`*_parser.py`, `focus_parser.py`, `file_detector.py`, `base_parser.py`).
- Safe reading for CSV/Parquet/GZIP/ZIP, schema/format validation, upload safety (size, members, paths, content checks), import orchestration/lineage (`import_manager.py`).

### Processing — `app/processing/`
- Validation, normalization to common schema, deduplication/overlap awareness, aggregation, currency and cost-basis handling, data-quality assessment. Preserves raw provider fields, lineage, and cost semantics (credits/adjustments, periods).

### Analytics — `app/analytics/`
- DuckDB queries over normalized Parquet for summaries, timeseries, breakdowns, provider comparison, anomaly inputs, and quality reporting.

### ML — `app/ml/`
- `dataset_builder.py`, `feature_engineering.py`, `time_series_split.py`, `train.py`, `evaluate.py`, `predict.py`, `explain.py`, `artifact_manager.py`, `model_contract.py`.
- Reusable, leakage-safe pipeline shared by scripts, Colab notebook, and inference. Training never runs in dashboard requests.

### Recommendations — `app/recommendations/`
- Transparent rule/statistical engine with evidence, priority, status, and limitations (no automatic cloud changes).

### Storage & artifacts
- `data/raw/` original imports
- `data/processed/` normalized Parquet and ML panels
- `data/rejected/` rejected files/records (subject to retention controls)
- `data/metadata/` SQLite app metadata (imports, lineage, app state)
- `artifacts/models/{candidates,staging,production}/` versioned model bundles
- `mlruns/` local MLflow tracking (as configured)

## Key design rules
- Provider-specific parsing isolated from normalization/UI; analytics independent of Streamlit.
- Inference uses the same feature definitions/schema as training; artifacts versioned and validated before use.
- Currencies and cost bases never mixed silently; missing days not fabricated as zero.
- Google Colab is for training only (not a runtime dependency).
