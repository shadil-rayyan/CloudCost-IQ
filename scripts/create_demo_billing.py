"""Generate clearly labelled synthetic billing data for demos and tests.

This is **not** real billing data and must never be used to claim production
accuracy. The output is a normalized-schema Parquet file whose ``raw_fields_json``
column records ``"synthetic": true``, with a sidecar manifest that repeats the
warning. Use it to exercise the dashboard or the training pipeline end to end
when real exports are unavailable.

Usage::

    python scripts/create_demo_billing.py --days 420
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.constants import NORMALIZED_COLUMNS, classify_service  # noqa: E402
from app.utils.files import write_parquet_atomic  # noqa: E402

PROVIDERS = ("AWS", "Azure", "GCP")
SERVICES = {
    "AWS": (("Amazon Elastic Compute Cloud", 82.0), ("Amazon Simple Storage Service", 21.0), ("Amazon RDS", 34.0)),
    "Azure": (("Virtual Machines", 74.0), ("Azure Storage", 18.0), ("Azure SQL Database", 29.0)),
    "GCP": (("Compute Engine", 68.0), ("Cloud Storage", 15.0), ("BigQuery", 26.0)),
}
ACCOUNTS = {
    "AWS": ("111122223333", "demo-aws-production"),
    "Azure": ("0000aaaa-bbbb-cccc-dddd-eeeeffff0000", "demo-azure-subscription"),
    "GCP": ("demo-gcp-project", "Demo GCP Project"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=420, help="Calendar days of synthetic history")
    parser.add_argument("--end", default=dt.date.today().isoformat(), help="Last synthetic date (ISO)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "demo" / "demo_synthetic_normalized.parquet",
    )
    return parser.parse_args()


def build_frame(days: int, end: dt.date, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=end, periods=days, freq="D")
    rows: list[dict] = []
    for provider in PROVIDERS:
        account_id, account_name = ACCOUNTS[provider]
        for service_name, base in SERVICES[provider]:
            level = base * rng.uniform(0.7, 1.3)
            for index, date in enumerate(dates):
                weekday = date.dayofweek
                weekly = 1.12 if weekday < 5 else 0.82
                trend = 1.0 + 0.00035 * index
                noise = rng.normal(1.0, 0.07)
                cost = max(0.0, round(level * weekly * trend * noise, 2))
                rows.append(
                    {
                        "record_id": f"demo-{uuid.uuid5(uuid.NAMESPACE_URL, f'{provider}-{service_name}-{date.date()}').hex[:24]}",
                        "provider": provider,
                        "billing_account_id": account_id,
                        "billing_account_name": account_name,
                        "linked_account_id": account_id if provider == "AWS" else None,
                        "subscription_id": account_id if provider == "Azure" else None,
                        "project_id": account_id if provider == "GCP" else None,
                        "project_name": account_name if provider == "GCP" else None,
                        "service_name": service_name,
                        "service_category": classify_service(service_name),
                        "sku_id": f"{provider.lower()}-{service_name.lower().replace(' ', '-')}",
                        "sku_description": f"Synthetic demo SKU for {service_name}",
                        "resource_id": None,
                        "region": {"AWS": "us-east-1", "Azure": "eastus", "GCP": "us-central1"}[provider],
                        "usage_start": date,
                        "usage_end": date + pd.Timedelta(days=1),
                        "charge_period_start": date,
                        "charge_period_end": date + pd.Timedelta(days=1),
                        "invoice_month": date.strftime("%Y-%m"),
                        "cost_amount": cost,
                        "currency": "USD",
                        "cost_basis": "billed",
                        "quantity": float(24 * 30),
                        "unit": "Hrs",
                        "credits_amount": 0.0,
                        "tax_amount": None,
                        "charge_category": "usage",
                        "charge_type": "Usage",
                        "pricing_model": "on-demand",
                        "tags_json": json.dumps({"environment": "demo", "synthetic": True}),
                        "raw_fields_json": json.dumps(
                            {
                                "synthetic": True,
                                "generator": "scripts/create_demo_billing.py",
                                "warning": "Synthetic demonstration data - not a real bill.",
                            }
                        ),
                        "source_file_id": "demo-synthetic",
                        "import_batch_id": "demo-synthetic",
                        "schema_version": "1.0",
                    }
                )
    frame = pd.DataFrame(rows)
    return frame.loc[:, list(NORMALIZED_COLUMNS)]


def main() -> int:
    args = parse_args()
    end = dt.date.fromisoformat(args.end)
    frame = build_frame(args.days, end, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(frame, args.output)
    manifest = {
        "synthetic": True,
        "warning": (
            "SYNTHETIC DEMONSTRATION DATA. Generated by scripts/create_demo_billing.py. "
            "It is not a real cloud bill and must not be used to claim production accuracy."
        ),
        "rows": int(len(frame)),
        "days": int(args.days),
        "start": frame["usage_start"].min().date().isoformat(),
        "end": frame["usage_start"].max().date().isoformat(),
        "providers": list(PROVIDERS),
        "currency": "USD",
        "cost_basis": "billed",
        "seed": args.seed,
        "file": args.output.name,
    }
    (args.output.parent / "demo_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(frame):,} synthetic rows to {args.output}")
    print("Reminder: this file is for demonstrations and automated tests only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
