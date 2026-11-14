# CloudCost IQ

CloudCost IQ is a local, Python multi-cloud FinOps dashboard for importing billing exports, normalizing them into a common schema, analyzing spend, and producing **validated** 30-day forecasts when a production model bundle exists.

## What works (MVP 1 + MVP 2)

- **Imports:** AWS Cost and Usage Report 2.0/Data Exports, Azure Cost Management exports (actual/amortized/FOCUS), Google Cloud detailed billing exports, and FOCUS 1.x files (CSV/Parquet/gzip/ZIP with safety checks).
- **Validation & normalization:** required-field checks, schema/version detection, preserved raw fields, currencies/cost bases never mixed silently, invalid rows reported (not dropped).
- **Analytics:** overview KPIs, provider/service/account/region/date filtering, trends/breakdowns, import history, data quality, CSV exports.
- **Recommendations/anomalies (transparent):** statistical/rule-based; no claims of an advanced deployed model when absent.
- **ML (MVP 2):** global multi-horizon XGBoost quantile regression (P10/P50/P90), direct horizons 1–30, leakage-safe lags/rolling/calendar features, chronological rolling-origin backtests + untouched final test, baselines (trailing 7-day average, seasonal-naive weekly), MAE/RMSE/WAPE/pinball/interval coverage/crossing checks, MLflow tracking, versioned artifact bundles, and inference only from a **validated production** bundle.

**No fake data:** the app never fabricates billing data, predictions, metrics, or savings. The forecast page shows an unavailable message until a validated production model is registered.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt -r requirements-dev.txt
# optional ML training stack:
pip install -r requirements-ml.txt

streamlit run app/main.py       # dashboard
uvicorn app.api.main:app --reload --port 8000
```

Docker (local): `docker compose up --build` (set ports/volumes in `docker-compose.yml`).

## Training (real data only)

1. Import real normalized billing exports through the app or supply normalized Parquet.
2. Prepare dataset: `python scripts/prepare_training_data.py --currency USD --cost-basis billed`
3. Inspect: `python scripts/inspect_dataset.py`
4. Train candidate: `python scripts/train_model.py --panel data/processed/ml/daily_panel.parquet`
5. Validate/promote: `python scripts/validate_model.py --version <version> --promote`
6. Colab: `notebooks/colab_training.ipynb` (imports reusable `app.ml.*` modules).

Eligibility gates (configurable): 180+ days history (365 preferred), enough observed days/coverage, one currency + one cost basis per cohort, 3 rolling-origin folds + untouched 30-day test. Missing days remain missing (never zero). Synthetic `scripts/create_demo_billing.py` output is **demo/test only** and is clearly labelled.

## Data model / FOCUS

Normalized schema includes provider, billing/subscription/project, service/SKU/resource/region, usage/charge/invoice periods, `cost_amount`, `currency`, `cost_basis`, quantity/unit, credits/tax, tags, preserved raw fields, and lineage. FOCUS 1.x is detected and mapped; unsupported incompatible versions are rejected with a clear error.

## Testing

```bash
pytest -q
ruff check app scripts tests
```

## Known limitations

- Forecasts depend on delayed/revised billing exports and validated model quality; intervals are estimates.
- No CPU/memory telemetry, no automatic cloud resource changes, no FX conversion that silently mixes currencies.
- MVP 3 items (model registry lifecycle automation, drift monitoring, direct cloud connectors, CI/CD/monitoring hardening) are not implemented.
