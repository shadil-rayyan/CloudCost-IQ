# ML training guide (MVP 2)

**Model:** one global multi-horizon XGBoost quantile regressor (`reg:quantileerror`, P10/P50/P90, `tree_method=hist`). Direct horizons 1–30; no recursive lag substitution.

## Prepare data
```bash
python scripts/prepare_training_data.py --currency USD --cost-basis billed
python scripts/inspect_dataset.py --panel data/processed/ml/daily_panel.parquet
```
Eligibility: 180+ days (365 preferred), observed coverage/series gates, one currency + one cost basis. Missing days are not zero-filled.

## Train candidate (never auto-promoted)
```bash
python scripts/train_model.py --panel data/processed/ml/daily_panel.parquet [--track-mlflow]
python scripts/validate_model.py --version <version>           # gates only
python scripts/validate_model.py --version <version> --promote # explicit promotion
```

## Evaluation
Chronological rolling-origin folds + untouched final test. Metrics: MAE, RMSE, WAPE, pinball loss (P10/P50/P90), P10–P90 interval coverage/width, quantile crossing rate. Baselines: trailing 7-day average and seasonal-naive weekly (benchmarks only). Final test is never used for selection.

## Colab
Use `notebooks/colab_training.ipynb`. It is designed to run on Google Colab with:
- reproducible project checkout (set `CLOUDCOST_REPO_URL`/`CLOUDCOST_REPO_REF`) and dependency install,
- data source options: `CLOUDCOST_DATA_SOURCE=drive|upload|synthetic_demo` (synthetic is **demo-only**, never for production claims),
- cohort selection via `CLOUDCOST_CURRENCY`/`CLOUDCOST_COST_BASIS`,
- end-to-end pipeline: validation/eligibility → panel → leakage-safe features → rolling-origin + untouched test → train/evaluate (baselines, pinball, coverage, crossing) → explainability → candidate bundle save/validation → zip + optional copy to Drive.

The notebook records package versions, Git ref, dataset fingerprint, and imports `app.ml.*` for identical features. Promote only with `scripts/validate_model.py --version <version> --promote` after all gates pass.
