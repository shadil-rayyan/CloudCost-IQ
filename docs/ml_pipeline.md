# ML pipeline

1. **Normalized records** (Parquet) → dataset builder: daily panel per series (`provider+billing_account+service+currency+cost_basis`), coverage/missing/eligibility report, dataset fingerprint.
2. **Feature engineering** (reused in scripts/notebook/inference): lag 1/7/14/28, rolling 7/28 stats/sums, EWM/trends, days since last observation, observed ratio, horizon and target-date calendar/cyclical features, categorical provider/service/account/category. All historical features use data ≤ origin.
3. **Supervised direct multi-horizon rows** for horizons 1..30; target = future observed daily cost.
4. **Chronological splits:** rolling-origin validation folds + untouched final test.
5. **Training:** XGBoost quantile model, early stopping on validation folds only; selected rounds applied to final test fit and deployable refit.
6. **Evaluation:** point/quantile/calibration metrics and baselines; promotion gates.
7. **Artifact bundle:** `model.ubj`, `feature_schema.json`, `model_config.json`, `metrics.json`, `training_manifest.json`, `dataset_quality.json`, `model_card.md`, checksums.
8. **Inference:** only validated production bundle; currency/cost-basis/scope/history checks; monotonic P10≤P50≤P90 correction recorded; results persisted to metadata/forecasts.
