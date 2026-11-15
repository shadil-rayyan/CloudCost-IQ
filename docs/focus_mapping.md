# FOCUS Mapping

CloudCost IQ uses FOCUS 1.4 as its canonical billing schema.

## Supported versions
- Target output: FOCUS 1.4.
- Import: FOCUS 1.2 (version-aware, converted only when needed).

## Provider to FOCUS mapping (core)
| FOCUS column | AWS | Azure | GCP |
|---|---|---|---|
| ServiceProviderName | provider (AWS) | provider (Azure) | provider (GCP) |
| BillingAccountId | line_item_usage_account_id | SubscriptionId | billing_account_id |
| BillingCurrency | line_item_currency_code | BillingCurrency | currency |
| BilledCost | line_item_unblended_cost | CostInBillingCurrency | cost (+ credits) |
| ChargePeriodStart | line_item_usage_start_date | Date/UsageStart | usage_start_time |
| ChargePeriodEnd | line_item_usage_end_date | Date/UsageEnd | usage_end_time |
| ServiceName | product_servicecode | MeterCategory | service.description |
| RegionName | product_region | ResourceLocation | location.region |

Unmapped provider fields are preserved as x_-prefixed custom columns where useful.
Application metadata (import IDs, file hashes, timestamps) stays in SQLite.
