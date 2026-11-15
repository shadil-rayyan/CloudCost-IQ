# ML Training Guide (MVP 2)

## Overview
CloudCost IQ uses **one global multi-horizon XGBoost quantile regression** model to forecast **daily billed cost** for the next **30 days**, producing **P10/P50/P90** estimates. Training is reproducible (scripts + Colab notebook) and inference is allowed only from **validated, explicitly promoted** production artifacts with compatible, eligible data.

**Key constraints:** no fake predictions/metrics; no leakage; no mixing of currencies/cost bases; missing days are not zero-filled; final test is untouched during selection.

## Prerequisites
- Python 3.11+ and dependencies: `pip install -r requirements.txt -r requirements-ml.txt`
- Real normalized billing data (Parquet) with sufficient eligible history for the target cohort (currency + cost basis). Synthetic/demo data is **not** for production-accuracy claims.

## 1) Prepare and inspect data
```bash
python scripts/prepare_training_data.py --currency USD --cost-basis billed
python scripts/inspect_dataset.py --panel data/processed/ml/daily_panel.parquet
```
- Builds a daily panel per eligible series (e.g. provider/account/service/currency/cost basis), aggregates target cost, and emits quality/sufficiency information.
- **Eligibility (configurable, `configs/ml_config.yaml`):** ~180+ days history (365 preferred), sufficient observed days/coverage, series quality, and support for 3 rolling-origin folds + untouched 30-day final test. Monthly-only data is not eligible for daily forecasting.

## 2) Feature engineering (leakage-safe, reusable in `app/ml/`)
- Lags: `lag_1`, `lag_7`, `lag_14`, `lag_28`
- Rolling (7d/28d): mean, median, std, sum (and related trend/change features as implemented)
- Calendar/target-date: day-of-week, weekend, day-of-month, week, month, quarter
- Horizon/context: direct horizons 1..30, provider/service/account/category/region context as available
- All features computed from information available at/before forecast origin; no future leakage.

## 3) Train candidate (never auto-promoted)
```bash
python scripts/train_model.py --panel data/processed/ml/daily_panel.parquet [--track-mlflow]
```
- Trains global quantile model (`reg:quantileerror`, `[0.1,0.5,0.9]`, `tree_method=hist`), direct multi-horizon.
- Uses chronological rolling-origin validation folds and preserves an **untouched final test** period.
- Writes a versioned candidate bundle under `artifacts/models/candidates/<version>/` with metrics, config, schema, manifest, quality, and model card.

## 4) Validate and (explicitly) promote
```bash
python scripts/validate_model.py --version <version>            # checks/gates
python scripts/validate_model.py --version <version> --promote # explicit promotion
```
Promotion gates (configurable) include artifact/schema integrity, metric/baseline comparison, interval coverage/crossing bounds, and documented evaluation. A candidate must **not** be promoted automatically.

## 5) Evaluation methodology
- **Splits:** rolling-origin (expanding/rolling chronological) backtests simulating 30-day forecasts; final 30-day (or configured) test untouched.
- **Point metrics:** MAE, RMSE, WAPE (and MASE where defined). Avoid over-reliance on MAPE for near-zero costs.
- **Probabilistic metrics:** pinball loss (P10/P50/P90), P10–P90 interval coverage and width, quantile crossing rate (P10 ≤ P50 ≤ P90 enforced/checked).
- **Baselines (benchmarks only):** trailing 7-day average, seasonal-naive weekly. Report comparisons honestly; no fabricated improvement.

## 6) Google Colab
Notebook: [`notebooks/colab_training.ipynb`](../notebooks/colab_training.ipynb) (single-cell pipeline)
- Set repo/ref and data source (`CLOUDCOST_DATA_SOURCE=drive|upload|synthetic_demo`), cohort (`CLOUDCOST_CURRENCY`, `CLOUDCOST_COST_BASIS`).
- Runs: environment capture → checkout/install → load/validate → panel/eligibility → features → splits → train/evaluate/explain → bundle + validation.
- Records versions, git ref, dataset fingerprint; imports `app.ml.*` for parity with inference.
- **`synthetic_demo` is for demo-only notebook runs, not production-accuracy claims.** Copy/download artifacts out of the runtime and promote only via validation gates.

## 7) MLflow
- Optional tracking (`configs/ml_config.yaml` / env). Log params, metrics, folds/final test, dataset fingerprint, and artifacts to local/approved tracking (`mlruns/` by default). Do not upload private raw billing data to public servers.

## 8) Inference integration
- App loads **production** bundles only, validates feature schema/currency/cost basis/horizon/quantiles, and returns P10/P50/P90 or a clear unavailable/insufficient-history reason. Training never occurs on UI/API request paths.

See [`docs/ml_pipeline.md`](ml_pipeline.md) and [`docs/model_card.md`](model_card.md).
