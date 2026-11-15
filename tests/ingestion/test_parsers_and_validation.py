from __future__ import annotations

import pandas as pd

from app.ingestion.aws_parser import AWSCURParser
from app.ingestion.azure_parser import AzureCostParser
from app.ingestion.file_detector import detect_schema
from app.ingestion.gcp_parser import GCPBillingParser
from app.processing.normalization import finalize_records
from app.processing.validation import validate_batch


def test_detect_and_parse_aws():
    raw = pd.DataFrame([
        {
            "lineItem/UsageStartDate": "2026-01-01T00:00:00Z",
            "lineItem/UsageEndDate": "2026-01-01T01:00:00Z",
            "lineItem/UnblendedCost": "12.34",
            "lineItem/CurrencyCode": "USD",
            "product/servicecode": "AmazonEC2",
            "bill/PayerAccountId": "123456789012",
            "lineItem/LineItemType": "Usage",
        }
    ])
    detected = detect_schema(raw)
    assert detected.provider == "AWS"
    batch = AWSCURParser().parse(raw, cost_basis="billed")
    normalized = finalize_records(
        batch.frame, provider=batch.provider, import_batch_id="i", source_file_id="f", cost_basis="billed", schema_version=batch.schema_version, raw_frame=batch.raw_frame
    )
    report = validate_batch(normalized)
    assert report.valid_records == 1
    assert report.rejected_records == 0
    assert normalized.loc[0, "BillingCurrency"] == "USD"
    assert normalized.loc[0, "ServiceProviderName"] == "AWS"


def test_detect_and_parse_azure():
    raw = pd.DataFrame([
        {
            "Date": "2026-01-01",
            "CostInBillingCurrency": "5.50",
            "BillingCurrency": "USD",
            "SubscriptionId": "sub-1",
            "MeterId": "meter-1",
            "ServiceName": "Virtual Machines",
            "ChargeType": "Usage",
        }
    ])
    # Detection is heuristic; parsing/validation is the contract we care about.
    batch = AzureCostParser().parse(raw, cost_basis="actual")
    normalized = finalize_records(
        batch.frame, provider=batch.provider, import_batch_id="i", source_file_id="f", cost_basis="actual", schema_version=batch.schema_version, raw_frame=batch.raw_frame
    )
    report = validate_batch(normalized)
    assert report.valid_records == 1


def test_detect_and_parse_gcp_nets_credits():
    raw = pd.DataFrame([
        {
            "BillingAccountId": "acct-1",
            "service.description": "Compute Engine",
            "usage_start_time": "2026-01-01T00:00:00Z",
            "cost": "10.00",
            "credits": '[{"amount": -2.00}]',
            "BillingCurrency": "USD",
        }
    ])
    detected = detect_schema(raw)
    assert detected.provider == "GCP"
    batch = GCPBillingParser().parse(raw, cost_basis="billed")
    normalized = finalize_records(
        batch.frame, provider=batch.provider, import_batch_id="i", source_file_id="f", cost_basis="billed", schema_version=batch.schema_version, raw_frame=batch.raw_frame
    )
    assert abs(float(normalized.loc[0, "BilledCost"]) - 8.0) < 1e-9
    credits = normalized.loc[0, "credits_amount"]
    if pd.notna(credits):
        assert abs(float(credits) - (-2.0)) < 1e-9
    assert validate_batch(normalized).valid_records == 1


def test_validation_rejects_missing_required_fields():
    from app.core.constants import NORMALIZED_COLUMNS

    frame = pd.DataFrame([{column: None for column in NORMALIZED_COLUMNS}])
    frame.loc[0, "ServiceProviderName"] = "AWS"
    frame.loc[0, "BilledCost"] = 1.0
    # missing usage_start and currency -> rejected
    report = validate_batch(frame)
    assert report.rejected_records == 1
    codes = {issue.code for issue in report.issues}
    assert "missing_usage_start" in codes
    assert "missing_currency" in codes
