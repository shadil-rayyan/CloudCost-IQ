# Model card template

Generated bundles include `artifacts/models/*/<version>/model_card.md` with:
- purpose, architecture (global XGBoost multi-quantile, P10/P50/P90, horizon 30), and configuration;
- training data cohort (currency, cost basis, date range, eligible series, missing/zero-day handling);
- chronological rolling-origin evaluation + untouched final test metrics and baseline comparison;
- promotion gate results;
- limitations (billing delays/revisions, cohort specificity, sparse/non-recurring spend, empirical interval calibration, no causality/telemetry, not a sole basis for financial commitments).

Do not publish or serve a bundle that fails validation gates or lacks documented limitations.
