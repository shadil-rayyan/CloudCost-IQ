# Model Card (Template / Guidance)

Each trained candidate/production model **must** ship with a bundle-local `model_card.md` (and companion JSON artifacts) describing its purpose, data, evaluation, and limitations. Do not promote or serve a model without complete metadata.

## Required bundle contents (per version)
Stored under `artifacts/models/{candidates,staging,production}/<version>/` (names may include checksums/helpers produced by the pipeline):
- `model.ubj` (or native XGBoost model file)
- `preprocessor.joblib` (if used; trusted artifacts only)
- `feature_schema.json` (names, types, order, categorical mappings, schema version)
- `model_config.json` (model name/objective/quantiles/horizon/params, target, currency, cost basis)
- `metrics.json` (fold/final metrics, baselines, pinball, coverage/width, crossing, slices as available)
- `training_manifest.json` (version, dataset fingerprint/range, timestamp, git commit, package versions, seed, cohort coverage, checksums)
- `dataset_quality.json` (eligibility/coverage/series/rejection/missing-period summary)
- `model_card.md` (this document, filled for the specific model)

## Model card sections (to complete per artifact)
1. **Purpose & intended use:** 30-day daily cost forecast (P10/P50/P90) for FinOps planning; not a bill, not autonomous resource control.
2. **Architecture:** global XGBoost multi-quantile (`reg:quantileerror`, [0.1,0.5,0.9], `tree_method=hist`), direct horizons 1–30.
3. **Training data:** currency/cost basis/target, date range/history, eligible series/observed days, sources (real billing only for production claims), handling of missing days/incomplete periods.
4. **Features:** leakage-safe lags/rolling/trend/calendar/horizon/categorical context; feature schema version.
5. **Evaluation:** chronological rolling-origin folds + untouched final test; MAE/RMSE/WAPE, pinball, interval coverage/width, crossing; baseline comparisons; per-slice notes.
6. **Promotion/gate results:** pass/fail against configured thresholds with values.
7. **Limitations & risks:** billing delays/revisions, cohort specificity (currency/cost basis), sparse/non-recurring spend, interval miscalibration under regime change, no causal explanation/telemetry, not for sole basis of financial/contractual commitments.
8. **Ethics & safety:** no automatic cloud mutations; sensitive data handling; no fabricated metrics/savings.

## Example (candidate, synthetic-demo trained — **not for production accuracy claims**)
See: `artifacts/models/candidates/cciq-xgb-q30-20261114-164550/model_card.md`

Key facts from that example bundle (for reference only): USD/billed daily target, 2025-09-21–2026-11-14 history (420 days, 9 series), 3 rolling folds + untouched final test, direct 30-day quantiles. Final-test reported metrics include WAPE ~0.0667 and interval coverage ~0.656 with **interval_calibration gate FAIL** (coverage below configured minimum) — therefore it must **not** be treated as production-approved; keep in candidates unless/until gates pass and explicit promotion is performed. Being synthetic-demo trained, it must not be cited as real-world performance.

Always rely on the specific artifact's own `metrics.json`, `training_manifest.json`, and `dataset_quality.json` for authoritative values.
