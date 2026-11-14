# Data dictionary (normalized billing)

| Field | Notes |
|---|---|
| `record_id`, `source_file_id`, `import_batch_id`, `schema_version` | Lineage; never fabricated. |
| `provider` | AWS/Azure/GCP/FOCUS (record-level provider preserved when known). |
| `billing_account_id/name`, `linked_account_id`, `subscription_id`, `project_id/name` | Provider identity fields; null when unavailable. |
| `service_name`, `service_category`, `sku_id/description`, `resource_id`, `region` | Attribution; resource IDs may be missing. |
| `usage_start/end`, `charge_period_start/end`, `invoice_month` | Usage/charge/invoice semantics preserved separately. |
| `cost_amount`, `currency`, `cost_basis` | Monetary value in original currency; cost bases (billed/actual/amortized/effective/list where applicable) kept separate. Credits/refunds may be negative; GCP credits are netted into `cost_amount` with `credits_amount` preserved. |
| `quantity`, `unit`, `credits_amount`, `tax_amount` | Null when unavailable. |
| `charge_category`, `charge_type`, `pricing_model` | Normalized where possible. |
| `tags_json`, `raw_fields_json` | Preserved provider data; synthetic demo rows mark `synthetic: true` in raw metadata. |

Missing days in ML panels are represented as `observed=False` with `cost=NaN` (never zero).
