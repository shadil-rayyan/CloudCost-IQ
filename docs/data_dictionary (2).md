# Data Dictionary — Normalized Billing

Normalized records are produced by ingestion/processing and stored as Parquet in `data/processed/` (with metadata in SQLite). Unavailable fields are `null`; missing values are never invented. Original provider fields are preserved.

## Normalized columns (schema v1.0)
Defined in `app/core/constants.py` (`NORMALIZED_COLUMNS`):

| Field | Type (logical) | Description / notes |
|---|---|---|
| `record_id` | string | Internal unique record identifier |
| `provider` | string | `AWS`, `Azure`, `GCP`, or `FOCUS` (record-level provider preserved when known) |
| `billing_account_id` | string | Billing/payer account identifier |
| `billing_account_name` | string | Billing account name |
| `linked_account_id` | string | AWS linked/member account |
| `subscription_id` | string | Azure subscription |
| `project_id` | string | GCP project id |
| `project_name` | string | Project name |
| `service_name` | string | Cloud service/product |
| `service_category` | string | Normalized service grouping (where derivable) |
| `sku_id` | string | SKU/product identifier |
| `sku_description` | string | SKU description |
| `resource_id` | string | Resource identifier when available (may be null) |
| `region` | string | Region/location |
| `usage_start` | datetime (UTC-aware where applicable) | Usage start |
| `usage_end` | datetime | Usage end |
| `charge_period_start` | datetime | Charge period start |
| `charge_period_end` | datetime | Charge period end |
| `invoice_month` | string/date key | Invoice/billing month (preserved separately from usage) |
| `cost_amount` | numeric | Normalized cost value in original currency for the selected/available basis; credits/refunds may be negative (e.g. GCP credits netting preserved with `credits_amount`) |
| `currency` | string | Original currency code (e.g. USD, EUR) |
| `cost_basis` | string | `billed`, `actual`, `amortized`, `effective`, `list` (as applicable/supported) — bases kept separate |
| `quantity` | numeric | Usage quantity |
| `unit` | string | Unit of measure |
| `credits_amount` | numeric | Credits amount (nullable) |
| `tax_amount` | numeric | Tax amount when separately available (nullable) |
| `charge_category` | string | Normalized: usage, purchase, tax, credit, refund, adjustment, discount, fee, commitment, other |
| `charge_type` | string | Provider-specific charge type |
| `pricing_model` | string | On-demand, reservation, savings plan, etc. (when available) |
| `tags_json` | string (JSON) | Billing/resource tags/labels |
| `raw_fields_json` | string (JSON) | Original provider fields preserved for traceability (synthetic demo rows may include `synthetic: true` metadata) |
| `source_file_id` | string | Source file identifier |
| `import_batch_id` | string | Import batch identifier |
| `schema_version` | string | Normalized schema version |

## Semantics & integrity
- **No invented values:** null means unavailable/unknown.
- **Currencies/cost bases:** never summed or converted silently across incompatible groups.
- **Adjustments:** credits, refunds, taxes, and adjustments preserved according to source semantics.
- **Periods:** usage, charge, and invoice timing preserved distinctly; incomplete/missing periods are tracked and not treated as zero spend.
- **ML panels:** missing days represented as `observed=False` with `cost=NaN` (never zero-filled to imply spend).

See also [`docs/ingestion_guide.md`](ingestion_guide.md) and [`docs/dataset_sources.md`](dataset_sources.md).
