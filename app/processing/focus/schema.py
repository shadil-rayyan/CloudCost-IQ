"""FOCUS 1.4 canonical schema definition (supported subset)."""

from __future__ import annotations

from typing import Final

# Core FOCUS 1.4 Cost and Usage columns supported by CloudCost IQ.
FOCUS_COLUMNS: Final[tuple[str, ...]] = (
    "ServiceProviderName",
    "BillingAccountId",
    "BillingAccountName",
    "SubAccountId",
    "BillingCurrency",
    "ChargePeriodStart",
    "ChargePeriodEnd",
    "BilledCost",
    "EffectiveCost",
    "ListCost",
    "ContractedCost",
    "ChargeCategory",
    "ServiceCategory",
    "ServiceName",
    "ServiceSubcategory",
    "ResourceId",
    "RegionName",
    "ConsumedQuantity",
    "ConsumedUnit",
    "SkuId",
    "SkuDescription",
    "Tags",
    "PricingCategory",
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "ServiceProviderName",
    "BillingCurrency",
    "BilledCost",
    "ChargePeriodStart",
)

RECOMMENDED_COLUMNS: Final[tuple[str, ...]] = (
    "BillingAccountId",
    "ServiceName",
    "ChargeCategory",
)

CUSTOM_PREFIX: Final[str] = "x_"
