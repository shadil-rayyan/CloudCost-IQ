# ML Pipeline (MVP 2)

## Objective
Predict **next 30 days** of daily cloud spend (default target: `daily_billed_cost`) per eligible scope with a **global multi-horizon XGBoost quantile** model outputting **P10, P50, P90**.

## End-to-end stages
1. **Normalized billing data** (`data/processed/*.parquet`, schema/lineage preserved)
2. **Dataset preparation** (`scripts/prepare_training_data.py`, `app/ml/dataset_builder.py`)
   - Select cohort: currency + cost basis
   - Aggregate to daily panel by series (e.g. provider/billing account/service/...)
   - Record coverage/quality; keep missing days as missing (`observed=False`, `cost=NaN`)
   - Enforce eligibility gates (history, observed coverage, fold/test feasibility)
3. **Feature engineering** (`app/ml/feature_engineering.py`, leakage-safe)
   - Lags (1/7/14/28), rolling stats (7/28d mean/median/std/sum), trend/change, calendar & target-date features, horizon (1..30), categorical context
4. **Chronological splitting** (`app/ml/time_series_split.py`)
   - Rolling-origin validation folds (each simulating 30-day forecast)
   - **Untouched final test** window (never used for selection/tuning)
5. **Training** (`scripts/train_model.py`, `app/ml/train.py`)
   - XGBoost `reg:quantileerror`, quantiles [0.10,0.50,0.90], `tree_method=hist`, direct horizons
   - Early stopping on chronological validation; configuration captured
6. **Evaluation** (`app/ml/evaluate.py`)
   - MAE, RMSE, WAPE; pinball loss (P10/P50/P90); interval coverage/width; quantile crossing rate
   - Baselines: trailing 7-day average, seasonal-naive weekly (benchmarks)
   - Slice reporting where applicable (overall/provider/horizon buckets, etc.)
7. **Explainability** (`app/ml/explain.py`)
   - Global feature importance and SHAP-based analysis where supported (interpretation ≠ causation)
8. **Artifact bundle** (`app/ml/artifact_manager.py`, `artifacts/models/candidates/<version>/`)
   - Trained model (native), feature schema/order & mappings, model config, metrics (incl. baselines/folds/final), training manifest (dataset fingerprint/range, code/env versions, seed), dataset quality, model card, checksums (as produced)
9. **Validation & promotion** (`scripts/validate_model.py`)
   - Contract/schema/metric/gate checks; **explicit** promotion to `production/` (no auto-promotion)
10. **Inference** (`app/ml/predict.py`, Forecast API/UI)
   - Load approved production artifact, validate cohort/schema/horizon/quantiles, generate 30-day features, predict, enforce/record ordering, return results or clear unavailable/insufficient-history reason

## Governance rules
- One compatible currency + one cost basis per artifact/cohort.
- No training during UI/API requests; no silent model switching.
- Quantile crossing monitored and corrected/handled per contract before display.
- Private billing data and credentials never committed or exposed. MLflow tracking stored in approved locations.
- Colab used for training experiments only; notebook imports reusable `app.ml.*` for parity.
