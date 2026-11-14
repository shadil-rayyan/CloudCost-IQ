# Ingestion guide

1. Open **Import Billing Data** in the dashboard or POST `/api/v1/imports`.
2. Choose provider (Auto/AWS/Azure/GCP/FOCUS), cost basis, and dataset kind (`real`/`demo`/`test`).
3. Upload CSV, Parquet, gzip-compressed supported files, or ZIP (safe extraction; size/member limits enforced).
4. The app detects schema, validates records, shows preview/warnings/errors, and stores import lineage.
5. Confirm to normalize and persist processed Parquet. Rejected records are reported; invalid records are never silently discarded.

FOCUS 1.x is supported; incompatible/unsupported schema versions are rejected. Azure actual vs amortized is determined by export/cost basis selection and surfaced clearly. GCP nested credits are preserved and netted into cost with `credits_amount` retained.
