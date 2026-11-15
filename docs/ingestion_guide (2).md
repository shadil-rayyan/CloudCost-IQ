# Ingestion Guide

This guide describes how CloudCost IQ imports, validates, and normalizes billing exports.

## Supported inputs
- **Providers/formats:** AWS CUR 2.0 / Data Exports, Azure Cost Management exports (actual/amortized/FOCUS), GCP Detailed Billing exports, and **FOCUS 1.x** unified exports.
- **File types:** `.csv`, `.parquet`, `.csv.gz`/supported `.gz`, `.zip` (safe extraction). Content is validated (not just extensions).
- **Provider selection:** Auto-detect (default) or explicit provider where offered.

## Import flow
1. **Upload** via Streamlit **Import Billing Data** page or `POST /api/v1/imports` (multipart).
2. **Inspection & detection:** identify format/provider/schema hints and columns.
3. **Validation:** required fields, schema compatibility, date/cost/currency checks, row-level issues; preview and warnings/errors surfaced.
4. **Confirmation:** user confirms import.
5. **Normalization & persistence:** map to common schema, preserve raw fields/lineage, write normalized Parquet (`data/processed/`), store original (`data/raw/`), record metadata/quality in SQLite; rejected/problem details tracked (and `data/rejected/` as applicable).
6. **Review:** **Import History** and **Data Quality** pages show status, coverage, counts, warnings, duplicates/overlaps, and issues.

## FOCUS support
- **Supported:** FOCUS **1.x**. The parser detects/records schema version metadata, validates required columns, maps standard fields, preserves extensions/raw fields, and stores record-level provider when available.
- **Unsupported:** incompatible declared major versions are rejected with a clear, actionable error. Do not assume all providers emit identical FOCUS versions.

## Validation & data quality
- Checks include: missing required fields, invalid costs/dates/currencies, unsupported schemas, corrupt/oversized/unsafe files, duplicate/overlapping imports, incomplete/missing periods, negative credits/adjustments.
- Missing optional fields (e.g. resource IDs/tags) are not automatically errors; unavailable values remain `null`.
- **Missing vs zero:** missing/incomplete billing periods are not converted to zero spend. Only represent zero when coverage confirms a complete period with no billable records.
- Invalid records are reported (counts/warnings/details where applicable), not silently discarded.

## Cost semantics
- Keep **cost basis** distinct (billed/actual/amortized/effective/list as supported). Do not silently remove adjustments/credits/refunds/taxes.
- Keep **original currency**; never sum incompatible currencies or perform undocumented FX conversion.
- Preserve usage, charge, and invoice timing separately.

## Import tracking & reimports
Each import records identifiers, provider/format/schema, timestamps, coverage, currency/cost basis, records read/valid/rejected, duplicates/warnings/errors, and lineage. File fingerprinting and controlled handling of revised/overlapping exports are supported to avoid silent corruption or naive row-level deduplication.

## Safety controls
- File size, uncompressed size, archive member, and row limits (configurable in `configs/app_config.yaml`).
- Safe ZIP handling (path traversal/member checks), content-based validation, no execution of uploaded content.
- Local-first processing; no cloud credentials required for file-upload MVP flows.

## Manual demo (synthetic)
Use labeled files in `manual_test_datasets/` (see its `README.md`) to demonstrate valid imports, multi-provider/FOCUS, formats (parquet/gzip/zip), validation/rejections/warnings, and mixed-currency separation. Synthetic data must not be treated as real billing.
