# Dataset sources

| Source | Ingestion | Dashboard | ML training |
|---|---:|---:|---:|
| AWS CUR 2.0 / Data Exports (real) | Yes | Yes | Yes |
| Azure Cost Management exports (real, actual/amortized/FOCUS) | Yes | Yes | Yes |
| GCP Detailed Billing export (real) | Yes | Yes | Yes |
| FinOps FOCUS Sandbox / FOCUS files | Yes | Yes | Limited/research only (documented coverage may be insufficient for 30-day daily forecasting) |
| Microsoft/sample schema exports | Schema testing | Limited | No production claims |
| `scripts/create_demo_billing.py` synthetic output | Yes (labelled) | Demo/test only | No production claims |

Use at least 180 days of real daily history (365+ preferred), one compatible currency and cost basis per training cohort. Do not convert currencies silently or manufacture daily values from monthly totals.
