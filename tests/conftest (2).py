"""Pytest configuration: isolate all storage under tmp_path and clear config caches."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts" / "models"
    monkeypatch.setenv("CLOUDCOST_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CLOUDCOST_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("CLOUDCOST_DATABASE_PATH", str(data_dir / "metadata" / "cloudcost_iq.db"))
    from app.core.config import get_app_config, reset_config_cache

    reset_config_cache()
    config = get_app_config()
    config.ensure_directories()
    yield tmp_path
    reset_config_cache()


@pytest.fixture
def sample_normalized_frame() -> pd.DataFrame:
    from app.core.constants import NORMALIZED_COLUMNS, classify_service

    rows = []
    start = dt.date(2025, 1, 1)
    for day in range(220):
        date = start + dt.timedelta(days=day)
        for provider, service, base in (
            ("AWS", "Amazon EC2", 40.0),
            ("Azure", "Virtual Machines", 35.0),
            ("GCP", "Compute Engine", 30.0),
        ):
            rows.append(
                {
                    "record_id": f"{provider}-{service}-{date}",
                    "provider": provider,
                    "billing_account_id": f"{provider.lower()}-acct",
                    "billing_account_name": f"{provider} Demo",
                    "linked_account_id": None,
                    "subscription_id": None,
                    "project_id": None,
                    "project_name": None,
                    "service_name": service,
                    "service_category": classify_service(service),
                    "sku_id": service,
                    "sku_description": service,
                    "resource_id": None,
                    "region": "us-east-1",
                    "usage_start": pd.Timestamp(date),
                    "usage_end": pd.Timestamp(date + dt.timedelta(days=1)),
                    "charge_period_start": pd.Timestamp(date),
                    "charge_period_end": pd.Timestamp(date + dt.timedelta(days=1)),
                    "invoice_month": date.strftime("%Y-%m"),
                    "cost_amount": base + (day * 0.02),
                    "currency": "USD",
                    "cost_basis": "billed",
                    "quantity": 24.0,
                    "unit": "Hrs",
                    "credits_amount": 0.0,
                    "tax_amount": None,
                    "charge_category": "usage",
                    "charge_type": "Usage",
                    "pricing_model": "on-demand",
                    "tags_json": "{}",
                    "raw_fields_json": '{"synthetic_test": true}',
                    "source_file_id": "test",
                    "import_batch_id": "test",
                    "schema_version": "1.0",
                }
            )
    return pd.DataFrame(rows, columns=list(NORMALIZED_COLUMNS))
