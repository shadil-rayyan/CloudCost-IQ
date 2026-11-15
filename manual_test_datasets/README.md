# Manual test datasets (SYNTHETIC)

**SYNTHETIC DEMONSTRATION DATA ONLY.** These files are not real cloud billing exports and must not be used to claim production or real-world accuracy.

Use these files to manually test CloudCost IQ features via **Import Billing Data** (Streamlit UI) or `POST /api/v1/imports` (mark dataset kind as demo/test where applicable).

## Generate / refresh
From the project root:
```powershell
python manual_test_datasets/generate_datasets.py
```

## Files and what to test

| File | Format / Provider | What to test |
|---|---|---|
| `01_aws_cur_valid_demo.csv` | AWS CUR-style CSV | Valid AWS import, normalization, provider/service/region/date parsing, daily trends. |
| `02_azure_cost_valid_demo.csv` | Azure Cost Management CSV | Valid Azure import, subscription/service mapping, daily costs. |
| `03_gcp_billing_valid_demo.csv` | GCP detailed billing CSV | Valid GCP import, project/service mapping, credits/net-cost handling. |
| `04_focus_multicloud_valid_demo.csv` | FOCUS 1.x CSV (multi-provider rows) | FOCUS ingestion, multi-provider records, unified schema mapping. |
| `05_validation_and_quality_issues_demo.csv` | AWS-style CSV with intentional issues | Validation errors/warnings, invalid dates/missing/negative values, mixed currencies in file, Data Quality visibility (no silent drops/fabrication). |
| `06_mixed_currency_demo.csv` | AWS-style CSV, USD + EUR | Currency handling (currencies must not be incorrectly combined/summed). |
| `07_aws_cur_valid_demo.parquet` | Parquet | Parquet ingestion path. |
| `08_aws_cur_valid_demo.csv.gz` | GZIP CSV | Compressed upload support. |
| `09_focus_bundle_demo.zip` | ZIP (contains FOCUS CSV) | ZIP/archive upload handling. |
| `manifest.json` | Metadata | Generated synthetic file list and warning. |

## Suggested manual test flow
1. **Upload/import:** import `01`–`04` and `07`–`09`; verify detection, preview, warnings, and successful commit in Import Billing Data / Import History.
2. **Analytics:** check Overview and Cost Analysis (filters by provider/account/service/region/date; trends/breakdowns/tables).
3. **Data Quality & History:** review Import History and Data Quality (counts, coverage, freshness, issues).
4. **Validation behavior:** import `05_validation_and_quality_issues_demo.csv`; confirm issues are clearly reported and invalid data is not silently accepted/fabricated.
5. **Currency behavior:** import `06_mixed_currency_demo.csv`; confirm incompatible currencies are handled correctly (no misleading combined totals).
6. **Anomalies & Recommendations:** review rule/statistical, explainable outputs on imported demo data.
7. **Forecasts (ML Predictions):** require a **validated production model** and eligible history for the selected scope/currency/cost basis. These 90-day files may be insufficient for 30-day forecast eligibility—expect a clear unavailable/insufficient-history state rather than fake results. (Synthetic-trained demo models are demo-only, not real-accuracy evidence.)

## Notes
- Re-uploading the same file can be used to test duplicate/overlap detection.
- These datasets are synthetic; do not treat results as real billing or production-accurate forecasts.
