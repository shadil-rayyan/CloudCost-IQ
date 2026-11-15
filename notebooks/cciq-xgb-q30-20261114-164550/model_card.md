# Model card - cloudcost_iq_global_xgboost_quantile (cciq-xgb-q30-20261114-164550)

## Purpose

Forecast daily cloud spending for the next 30 days with an
estimated P10/P50/P90 interval. The model supports FinOps planning; it does not
modify cloud resources and is not a billing statement.

## Model architecture

- One global XGBoost multi-quantile regressor (`reg:quantileerror`) trained
  across eligible provider/account/service series.
- Quantiles: P10, P50, P90.
- Tree method: `hist`; 132 boosting rounds selected from
  the validation folds.
- Direct multi-horizon design: horizons 1..30 are predicted
  independently, without recursive lag substitution.

## Training data

- Currency: USD; cost basis: billed; target:
  `daily_billed_cost`.
- History: 2025-09-21 to 2026-11-14
  (420 calendar days).
- Eligible series: 9 of
  9; observed days:
  3780.
- Missing days remain missing and are not treated as zero-cost days.

## Evaluation methodology

Chronological rolling-origin backtesting: 3 validation
folds, each simulating a 30-day forecast, plus an untouched
final test window. Metrics are out-of-sample only.

| Fold | WAPE | MAE | P10-P90 coverage | Rounds |
|---|---|---|---|---|
| fold_1 | 0.0645 | 3.02 | 0.670 | 135 |
| fold_2 | 0.0644 | 3.12 | 0.708 | 132 |
| fold_3 | 0.0624 | 3.03 | 0.696 | 108 |

Final test WAPE: 0.0667; MAE: 3.24;
RMSE: 5.05; pinball loss (mean):
1.0692; interval coverage:
0.656; interval width:
7.67; quantile crossing rate:
0.00000.

Baseline comparison (final test):
- Trailing 7-day average WAPE: 0.1333
- Seasonal-naive weekly WAPE: 0.1598
- Improvement versus trailing average: 0.499
- Pinball ratio versus best baseline: 0.331

## Promotion gates

| Gate | Result | Detail |
|---|---|---|
| wape_within_limit | PASS | wape=0.06673518277615442 limit=0.6 |
| pinball_beats_baseline | PASS | ratio=0.33088273154757497 limit=1.1 |
| interval_calibration | FAIL | coverage=0.6561529271206691 expected=[0.7, 0.95] |
| quantile_crossing_within_limit | PASS | rate=0.0 limit=0.02 |
| baseline_comparison_present | PASS | baseline comparison recorded |
| final_test_untouched | PASS | final test period excluded from model selection |

## Known limitations

- Billing exports are delayed and revised; recency of the latest complete day
  directly affects forecast quality.
- The model is trained per currency and cost basis and must not be applied to a
  different cohort.
- Sparse series, missing days and non-recurring purchases increase uncertainty.
- Quantile intervals are empirically calibrated on historical data and can be
  wrong during unprecedented events.
- No CPU/memory telemetry is used; forecasts cannot explain *why* a cost
  changed, only that it changed.
- Predictions must not be used as the sole basis for contractual or financial
  commitments.
