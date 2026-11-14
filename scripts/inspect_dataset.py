"""Inspect the daily training panel and print its quality report.

Usage::

    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --panel data/processed/ml/daily_panel.parquet

The command is read-only. It reports coverage, eligible series, missing days,
zero-cost days and the dataset fingerprint so a training run can be audited
against the data it actually used.
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
from app.ml.pipeline import load_panel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ml" / "daily_panel.parquet",
    )
    parser.add_argument(
        "--quality",
        type=Path,
        default=None,
        help="Quality report path (default: dataset_quality.json next to the panel)",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    panel_path: Path = args.panel
    if not panel_path.exists():
        print(f"Panel file not found: {panel_path}")
        print("Run scripts/prepare_training_data.py first.")
        return 1

    panel = load_panel(panel_path)
    quality_path = args.quality or panel_path.parent / "dataset_quality.json"
    quality: dict = {}
    if quality_path.exists():
        quality = json.loads(quality_path.read_text("utf-8"))

    observed = panel.loc[panel["observed"]]
    print(f"Panel: {panel_path}")
    print(f"Rows: {len(panel):,} | series: {panel['series_id'].nunique():,}")
    print(f"Observed rows: {len(observed):,} | missing days: {len(panel) - len(observed):,}")
    print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")

    if quality:
        keys = (
            "currency",
            "cost_basis",
            "history_days",
            "eligible_series_count",
            "series_count",
            "missing_periods",
            "zero_cost_days",
            "sufficient",
            "reason",
            "dataset_fingerprint",
        )
        print("\nDataset quality:")
        for key in keys:
            print(f"  {key}: {quality.get(key)}")
        providers = quality.get("providers") or []
        if providers:
            print(f"  providers: {', '.join(providers)}")
    else:
        print(f"\nNo quality report at {quality_path}; showing panel statistics only.")

    print("\nObserved cost by provider:")
    if not observed.empty:
        summary = (
            observed.groupby("provider")["cost"]
            .agg(["count", "sum", "mean", "max"])
            .sort_values("sum", ascending=False)
        )
        print(summary.to_string(float_format=lambda value: f"{value:,.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
