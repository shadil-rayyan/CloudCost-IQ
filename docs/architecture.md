# Architecture

CloudCost IQ is a modular monolith:

- **Streamlit UI** (`app/main.py`, `app/pages/*`) - dashboard only; calls reusable services/stores.
- **FastAPI** (`app/api/*`) - versioned REST endpoints for imports, costs, forecasts, anomalies, recommendations.
- **Ingestion** (`app/ingestion/*`) - provider parsers (AWS/Azure/GCP/FOCUS), detection, safe file reading, import manager.
- **Processing** (`app/processing/*`) - validation, normalization, deduplication, aggregation, currency/cost-basis handling, quality.
- **Analytics** (`app/analytics/*`, DuckDB+Parquet) - summaries, trends, breakdowns, provider comparison.
- **ML** (`app/ml/*`) - dataset builder, leakage-safe features, rolling-origin splits, XGBoost quantile training/evaluation, explainability, artifact manager, validated inference.
- **Recommendations** (`app/recommendations/*`) - transparent rule/statistical suggestions; no automatic cloud changes.

Storage: `data/raw` (original imports), `data/processed` (normalized Parquet), `data/rejected`, `data/metadata` (SQLite), `artifacts/models/{candidates,staging,production}`. Colab is for training only; the app never trains on page load.
