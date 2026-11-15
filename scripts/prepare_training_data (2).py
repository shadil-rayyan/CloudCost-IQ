"""Prepare the daily multi-series training panel from normalized billing records.

Usage::

    python scripts/prepare_training_data.py --currency USD --cost-basis billed

The panel and its dataset-quality report are written under ``data/processed/ml``
by default. No model is trained and no forecast is produced here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logging import configure_logging  # noqa: E402
from app.ml.pipeline import (  # noqa: E402
    load_normalized_records,
    prepare_training_panel,
    save_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=None, help="Normalized Parquet file or directory")
    parser.add_argument("--currency", default=None, help="Training currency cohort (default: ml_config)")
    parser.add_argument("--cost-basis", default=None, help="Training cost basis (default: ml_config)")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ml",
        help="Directory for daily_panel.parquet and dataset_quality.json",
    )
    parser.add_argument(
        "--allow-insufficient",
        action="store_true",
        help="Write the panel even when eligibility gates fail (never implies a model is trained)",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    records = load_normalized_records(args.records)
    if records.empty:
        print("No normalized billing records were found. Import billing data first.")
        return 1

    result = prepare_training_panel(
        records, currency=args.currency, cost_basis=args.cost_basis
    )
    quality = result.quality
    print(json.dumps(_digest(quality), indent=2, default=str))

    if result.panel.empty:
        print("No daily panel could be built.")
        return 1
    if not result.is_sufficient and not args.allow_insufficient:
        print(
            "Dataset does not meet the minimum eligibility requirements. "
            "Re-run with --allow-insufficient to persist it for inspection only."
        )
        return 2

    paths = save_panel(result, args.output)
    print(f"Panel written to {paths['panel']}")
    print(f"Quality report written to {paths['quality']}")
    return 0


def _digest(quality: dict) -> dict:
    keys = (
        "currency",
        "cost_basis",
        "series_count",
        "eligible_series_count",
        "history_start",
        "history_end",
        "history_days",
        "eligible_records",
        "missing_periods",
        "coverage",
        "providers",
        "sufficient",
        "reason",
        "dataset_fingerprint",
    )
    return {key: quality.get(key) for key in keys}


if __name__ == "__main__":
    raise SystemExit(main())
