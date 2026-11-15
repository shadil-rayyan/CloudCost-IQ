"""Generate synthetic, clearly-labeled billing datasets for manual UI/API testing.

Run:
    python manual_test_datasets/generate_datasets.py

SYNTHETIC DEMONSTRATION DATA ONLY (not real cloud bills). Do not use to claim
production or real-world accuracy.
"""

from __future__ import annotations

import gzip
import json
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(42)
START = date(2026, 1, 1)
DAYS = 90
DATES = [START + timedelta(days=i) for i in range(DAYS)]


def make_aws_cur_csv() -> Path:
    rows = []
    for d in DATES:
        for service, code, base in [
            ("Amazon Elastic Compute Cloud", "AmazonEC2", 55.0),
            ("Amazon Simple Storage Service", "AmazonS3", 12.0),
            ("Amazon Relational Database Service", "AmazonRDS", 28.0),
        ]:
            weekday = 1.1 if d.weekday() < 5 else 0.85
            cost = round(max(0.5, base * float(RNG.normal(1.0, 0.08)) * weekday), 4)
            rows.append(
                {
                    "identity/LineItemId": f"aws-{code}-{d.isoformat()}-{int(RNG.integers(1000, 9999))}",
                    "bill/PayerAccountId": "111122223333",
                    "bill/BillingPeriodStartDate": f"{d.replace(day=1).isoformat()}T00:00:00Z",
                    "lineItem/UsageAccountId": "111122223333",
                    "lineItem/LineItemType": "Usage",
                    "lineItem/UsageStartDate": f"{d.isoformat()}T00:00:00Z",
                    "lineItem/UsageEndDate": f"{(d + timedelta(days=1)).isoformat()}T00:00:00Z",
                    "lineItem/ProductCode": code,
                    "product/ServiceCode": code,
                    "product/ProductName": service,
                    "product/regionCode": "us-east-1",
                    "lineItem/UsageAmount": round(24 * float(RNG.uniform(0.8, 1.2)), 4),
                    "lineItem/UsageType": "BoxUsage:t3.medium",
                    "pricing/unit": "Hrs",
                    "lineItem/UnblendedCost": cost,
                    "lineItem/CurrencyCode": "USD",
                    "resourceTags/user:Environment": "demo",
                    "resourceTags/user:Synthetic": "true",
                }
            )
    p = ROOT / "01_aws_cur_valid_demo.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def make_azure_csv() -> Path:
    rows = []
    for d in DATES:
        for service, meter_cat, base in [
            ("Virtual Machines", "Virtual Machines", 48.0),
            ("Azure Storage", "Storage", 10.0),
            ("Azure SQL Database", "SQL Database", 24.0),
        ]:
            cost = round(max(0.5, base * float(RNG.normal(1.0, 0.09))), 4)
            rows.append(
                {
                    "Date": d.isoformat(),
                    "SubscriptionId": "0000aaaa-bbbb-cccc-dddd-eeeeffff0000",
                    "SubscriptionName": "Demo Azure Subscription",
                    "BillingAccountId": "ba-azure-demo",
                    "ServiceName": service,
                    "MeterCategory": meter_cat,
                    "MeterId": f"meter-{meter_cat.replace(' ', '-').lower()}",
                    "MeterName": f"{service} Usage",
                    "ResourceLocation": "eastus",
                    "Quantity": round(24 * float(RNG.uniform(0.7, 1.3)), 4),
                    "UnitOfMeasure": "1 Hour",
                    "CostInBillingCurrency": cost,
                    "BillingCurrency": "USD",
                    "ChargeType": "Usage",
                    "Tags": json.dumps({"Environment": "demo", "Synthetic": "true"}),
                }
            )
    p = ROOT / "02_azure_cost_valid_demo.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def make_gcp_csv() -> Path:
    rows = []
    for d in DATES:
        for service, base in [("Compute Engine", 44.0), ("Cloud Storage", 9.0), ("BigQuery", 18.0)]:
            gross = round(max(0.5, base * float(RNG.normal(1.0, 0.08))), 4)
            credit = round(-gross * 0.05, 4) if RNG.random() < 0.2 else 0.0
            rows.append(
                {
                    "billing_account_id": "012345-6789AB-CDEF01",
                    "project.id": "demo-gcp-project",
                    "project.name": "Demo GCP Project",
                    "service.description": service,
                    "sku.id": f"sku-{service.replace(' ', '-').lower()}",
                    "sku.description": f"{service} synthetic usage",
                    "usage_start_time": f"{d.isoformat()}T00:00:00Z",
                    "usage_end_time": f"{(d + timedelta(days=1)).isoformat()}T00:00:00Z",
                    "usage.amount": round(24 * float(RNG.uniform(0.8, 1.2)), 4),
                    "usage.unit": "hour",
                    "cost": gross,
                    "currency": "USD",
                    "credits": json.dumps([{"name": "Sustained use", "amount": credit}]) if credit else "[]",
                    "location.region": "us-central1",
                    "invoice.month": d.strftime("%Y%m"),
                    "labels": json.dumps([{"key": "environment", "value": "demo"}, {"key": "synthetic", "value": "true"}]),
                    "cost_type": "regular",
                }
            )
    p = ROOT / "03_gcp_billing_valid_demo.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def make_focus_csv() -> Path:
    rows = []
    for d in DATES:
        for provider, service, account, base in [
            ("AWS", "AmazonEC2", "111122223333", 40.0),
            ("Microsoft", "Virtual Machines", "sub-azure-demo", 38.0),
            ("Google Cloud", "Compute Engine", "demo-gcp-project", 36.0),
        ]:
            cost = round(max(0.5, base * float(RNG.normal(1.0, 0.10))), 4)
            rows.append(
                {
                    "BilledCost": cost,
                    "EffectiveCost": cost,
                    "BillingCurrency": "USD",
                    "ProviderName": provider,
                    "BillingAccountId": account,
                    "BillingAccountName": f"{provider} Demo",
                    "ServiceName": service,
                    "ServiceCategory": "Compute",
                    "ChargeCategory": "Usage",
                    "ChargeSubcategory": "On-Demand",
                    "PricingCategory": "On-Demand",
                    "ChargePeriodStart": f"{d.isoformat()}T00:00:00Z",
                    "ChargePeriodEnd": f"{(d + timedelta(days=1)).isoformat()}T00:00:00Z",
                    "BillingPeriodStart": f"{d.replace(day=1).isoformat()}T00:00:00Z",
                    "RegionId": "global-demo",
                    "ConsumedQuantity": 24.0,
                    "ConsumedUnit": "Hours",
                    "Tags": json.dumps({"environment": "demo", "synthetic": "true"}),
                }
            )
    p = ROOT / "04_focus_multicloud_valid_demo.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def make_validation_issues_csv() -> Path:
    # Intentionally mixed/invalid rows to test validation, warnings, and data-quality reporting.
    rows = []
    for i, d in enumerate(DATES[:30]):
        rows.append(
            {
                "identity/LineItemId": f"bad-{i}",
                "bill/PayerAccountId": "111122223333",
                "lineItem/UsageAccountId": "111122223333",
                "lineItem/LineItemType": "Usage",
                "lineItem/UsageStartDate": "not-a-date" if i in (3, 7) else f"{d.isoformat()}T00:00:00Z",
                "lineItem/UsageEndDate": f"{(d + timedelta(days=1)).isoformat()}T00:00:00Z",
                "product/ServiceCode": "AmazonEC2",
                "product/ProductName": "Amazon EC2",
                "product/regionCode": "us-east-1",
                "lineItem/UnblendedCost": "-5.00" if i == 10 else ("" if i == 12 else f"{20 + i}.00"),
                "lineItem/CurrencyCode": "" if i == 5 else ("EUR" if i % 11 == 0 else "USD"),
                "resourceTags/user:Synthetic": "true",
            }
        )
    p = ROOT / "05_validation_and_quality_issues_demo.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def make_mixed_currency_aws_csv() -> Path:
    # Valid rows but mixed currencies (USD/EUR) to test currency handling (no silent mixing).
    rows = []
    for i, d in enumerate(DATES[:45]):
        currency = "EUR" if i % 2 == 0 else "USD"
        rows.append(
            {
                "identity/LineItemId": f"mix-{i}",
                "bill/PayerAccountId": "111122223333",
                "lineItem/UsageAccountId": "111122223333",
                "lineItem/LineItemType": "Usage",
                "lineItem/UsageStartDate": f"{d.isoformat()}T00:00:00Z",
                "lineItem/UsageEndDate": f"{(d + timedelta(days=1)).isoformat()}T00:00:00Z",
                "product/ServiceCode": "AmazonEC2",
                "product/ProductName": "Amazon EC2",
                "product/regionCode": "eu-west-1" if currency == "EUR" else "us-east-1",
                "lineItem/UsageAmount": 24.0,
                "lineItem/UnblendedCost": round(30 + i * 0.2, 4),
                "lineItem/CurrencyCode": currency,
                "bill/BillingPeriodStartDate": f"{d.replace(day=1).isoformat()}T00:00:00Z",
                "resourceTags/user:Synthetic": "true",
            }
        )
    p = ROOT / "06_mixed_currency_demo.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def make_parquet_gzip_zip(aws_csv: Path, focus_csv: Path) -> list[Path]:
    out: list[Path] = []
    df = pd.read_csv(aws_csv)
    p_parquet = ROOT / "07_aws_cur_valid_demo.parquet"
    df.to_parquet(p_parquet, index=False)
    out.append(p_parquet)

    p_gz = ROOT / "08_aws_cur_valid_demo.csv.gz"
    with open(aws_csv, "rb") as f_in, gzip.open(p_gz, "wb") as f_out:
        f_out.writelines(f_in)
    out.append(p_gz)

    p_zip = ROOT / "09_focus_bundle_demo.zip"
    with zipfile.ZipFile(p_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(focus_csv, arcname=focus_csv.name)
    out.append(p_zip)
    return out


def main() -> None:
    aws = make_aws_cur_csv()
    azure = make_azure_csv()
    gcp = make_gcp_csv()
    focus = make_focus_csv()
    issues = make_validation_issues_csv()
    mixed = make_mixed_currency_aws_csv()
    extras = make_parquet_gzip_zip(aws, focus)
    files = [aws, azure, gcp, focus, issues, mixed, *extras]
    manifest = {
        "synthetic": True,
        "warning": "SYNTHETIC DEMO DATA - not real cloud bills. For CloudCost IQ manual testing only.",
        "generated_for": "Manual UI/API feature testing (upload, validation, normalization, analytics, data quality, import history, filters, currency handling).",
        "files": [p.name for p in files],
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Generated in", ROOT)
    for p in files:
        print(f" - {p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
