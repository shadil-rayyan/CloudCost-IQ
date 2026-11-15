"""Shared constants and small lookups used across the application."""

from __future__ import annotations

import re
from typing import Final

NORMALIZED_SCHEMA_VERSION: Final[str] = "1.4"

#: Supported provider discriminators in the normalized dataset.
PROVIDERS: Final[tuple[str, ...]] = ("AWS", "Azure", "GCP", "FOCUS")

#: Supported cost bases. ``billed`` is the default ML target basis.
COST_BASES: Final[tuple[str, ...]] = (
    "billed",
    "actual",
    "amortized",
    "effective",
    "list",
)

#: Normalized charge categories.
CHARGE_CATEGORIES: Final[tuple[str, ...]] = (
    "usage",
    "purchase",
    "tax",
    "credit",
    "refund",
    "adjustment",
    "discount",
    "fee",
    "commitment",
    "other",
)

#: Normalized column order for processed billing records.
NORMALIZED_COLUMNS: Final[tuple[str, ...]] = (
    "record_id",
    "ServiceProviderName",
    "BillingAccountId",
    "BillingAccountName",
    "SubAccountId",
    "linked_account_id",
    "subscription_id",
    "project_id",
    "project_name",
    "ServiceName",
    "ServiceCategory",
    "SkuId",
    "SkuDescription",
    "ResourceId",
    "RegionName",
    "usage_start",
    "usage_end",
    "ChargePeriodStart",
    "ChargePeriodEnd",
    "invoice_month",
    "BilledCost",
    "EffectiveCost",
    "ListCost",
    "ContractedCost",
    "BillingCurrency",
    "cost_basis",
    "ConsumedQuantity",
    "ConsumedUnit",
    "credits_amount",
    "tax_amount",
    "ChargeCategory",
    "charge_type",
    "PricingCategory",
    "Tags",
    "raw_fields_json",
    "source_file_id",
    "import_batch_id",
    "schema_version",
)

#: Columns used by the analytics layer; kept in sync with ``NORMALIZED_COLUMNS``.
DATETIME_COLUMNS: Final[tuple[str, ...]] = (
    "usage_start",
    "usage_end",
    "ChargePeriodStart",
    "ChargePeriodEnd",
)

NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "BilledCost",
    "EffectiveCost",
    "ListCost",
    "ContractedCost",
    "EffectiveCost",
    "ListCost",
    "ContractedCost",
    "ConsumedQuantity",
    "credits_amount",
    "tax_amount",
)

TEXT_COLUMNS: Final[tuple[str, ...]] = tuple(
    column
    for column in NORMALIZED_COLUMNS
    if column not in DATETIME_COLUMNS and column not in NUMERIC_COLUMNS
)

#: Canonical service categories derived from provider service names.
SERVICE_CATEGORY_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "compute": (
        "ec2",
        "compute engine",
        "virtual machine",
        "virtual machines",
        "vm",
        "kubernetes",
        "eks",
        "aks",
        "gke",
        "container",
        "lambda",
        "functions",
        "app service",
        "cloud run",
        "batch",
        "fargate",
    ),
    "storage": (
        "s3",
        "simple storage",
        "storage",
        "blob",
        "disk",
        "ebs",
        "efs",
        "fsx",
        "cloud storage",
        "persistent disk",
        "file storage",
        "backup",
        "snapshot",
    ),
    "database": (
        "rds",
        "database",
        "sql",
        "dynamodb",
        "cosmos",
        "spanner",
        "bigtable",
        "redis",
        "cache",
        "elasticache",
        "memorystore",
        "bigquery",
        "redshift",
        "synapse",
    ),
    "network": (
        "vpc",
        "network",
        "networking",
        "data transfer",
        "cloudfront",
        "load balancer",
        "load balancing",
        "nat gateway",
        "route",
        "dns",
        "cdn",
        "bandwidth",
        "vpn",
        "expressroute",
        "interconnect",
    ),
    "analytics": (
        "analytics",
        "athena",
        "emr",
        "glue",
        "kinesis",
        "dataflow",
        "dataproc",
        "data factory",
        "stream analytics",
        "pub/sub",
        "event hub",
        "event grid",
        "data catalog",
    ),
    "ai_ml": (
        "machine learning",
        "sagemaker",
        "bedrock",
        "vertex",
        "ai platform",
        "cognitive",
        "openai",
        "translation",
        "vision",
        "speech",
    ),
    "security": (
        "iam",
        "kms",
        "key vault",
        "security",
        "guardduty",
        "waf",
        "shield",
        "secret",
        "identity",
        "entra",
    ),
    "monitoring": (
        "cloudwatch",
        "monitoring",
        "log",
        "logging",
        "observability",
        "trace",
        "insights",
    ),
    "developer_tools": (
        "codebuild",
        "codepipeline",
        "codecommit",
        "devops",
        "artifact",
        "repository",
        "build",
        "developer",
    ),
    "support": ("support", "premium support", "developer support"),
}

OTHER_SERVICE_CATEGORY: Final[str] = "other"


def classify_service(service_name: str | None) -> str:
    """Map a provider service name to a normalized category.

    The mapping is deliberately keyword based and documented; it never invents
    information, it only groups services for display.
    """
    if not service_name:
        return OTHER_SERVICE_CATEGORY
    needle = str(service_name).lower()
    for category, keywords in SERVICE_CATEGORY_KEYWORDS.items():
        if any(keyword in needle for keyword in keywords):
            return category
    return OTHER_SERVICE_CATEGORY


CURRENCY_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{3}$")


def normalize_currency(value: object) -> str | None:
    """Return an upper-case ISO-4217 style currency code or ``None``."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL"}:
        return None
    return text[:3] if len(text) >= 3 else text
