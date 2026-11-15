# Dataset Sources

CloudCost IQ distinguishes billing data used for **ingestion testing**, **dashboard validation**, **ML training**, and **synthetic/demo** use. These are not interchangeable.

## Source matrix
| Source | Ingestion testing | Dashboard testing | ML training / production claims |
|---|---:|---:|---|
| **AWS CUR 2.0 / AWS Data Exports (real)** | Yes | Yes | Yes (eligible real history) |
| **Azure Cost Management exports (real, actual/amortized/FOCUS)** | Yes | Yes | Yes (eligible real history) |
| **GCP Detailed Billing export (real)** | Yes | Yes | Yes (eligible real history) |
| **FinOps FOCUS Sandbox / FOCUS files** | Yes | Yes | **Limited/research only** (documented coverage may be insufficient for 30-day daily forecasting; do not treat as sole training source) |
| **Microsoft sample/schema exports** | Schema/format testing | Limited | No |
| **Synthetic/demo data** (e.g. `manual_test_datasets/`, `scripts/create_demo_billing.py` output) | Yes (labeled) | Demo/test only | **No production-accuracy claims** |

## ML training data requirements (initial gates, configurable)
- Real normalized billing exports with meaningful time and cost information.
- **Target:** daily billed cost (default) aggregated from normalized records.
- **Cohort:** one compatible **currency** and one compatible **cost basis** per model artifact; no undocumented FX conversion.
- **History/coverage:** 180+ calendar days minimum (365+ preferred), sufficient observed days per series (target ≥120, configurable), and enough structure for **3 rolling-origin folds + untouched 30-day final test**.
- **Quality:** incomplete/missing periods handled explicitly; missing days are not fabricated as zero; monthly-only data is not eligible for daily 30-day forecasting (do not manufacture daily values).

## Important principles
- Do not increase row count with irrelevant/pricing-only/synthetic data as a substitute for historical billing behavior.
- Preserve invoice month separately from usage/charge timing; use a consistent, documented aggregation timezone.
- A dataset that fails eligibility must produce a clear insufficiency message while the rest of the app continues to function.

See [`configs/ml_config.yaml`](../configs/ml_config.yaml), [`docs/ml_training_guide.md`](ml_training_guide.md), and [`docs/ingestion_guide.md`](ingestion_guide.md).
