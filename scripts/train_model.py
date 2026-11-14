"""Train the global multi-horizon XGBoost quantile model.

Usage::

    python scripts/train_model.py --panel data/processed/ml/daily_panel.parquet

The command:
1. builds leakage-safe direct multi-horizon features,
2. creates chronological rolling-origin folds plus an untouched final test,
3. trains and evaluates the candidate,
4. writes a versioned bundle under ``artifacts/models/candidates``.

It never promotes the model; use ``scripts/validate_model.py`` for that. A
candidate that fails the promotion gates is still saved for inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logging import configure_logging  # noqa: E402
from app.ml.pipeline import (  # noqa: E402
    build_supervised,
    load_normalized_records,
    load_panel,
    make_splits,
    prepare_training_panel,
)
from app.ml.train import TrainingConfig, train_forecast_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=None, help="Daily panel Parquet file")
    parser.add_argument("--records", type=Path, default=None, help="Normalized records if no panel is supplied")
    parser.add_argument("--currency", default=None)
    parser.add_argument("--cost-basis", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--track-mlflow", action="store_true", help="Log the run to MLflow")
    parser.add_argument("--no-save", action="store_true", help="Evaluate without writing a bundle")
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--early-stopping-rounds", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    dataset_quality: dict = {}
    if args.panel:
        panel = load_panel(args.panel)
        quality_path = args.panel.parent / "dataset_quality.json"
        if quality_path.exists():
            dataset_quality = json.loads(quality_path.read_text("utf-8"))
    else:
        records = load_normalized_records(args.records)
        if records.empty:
            print("No normalized billing records were found.")
            return 1
        result = prepare_training_panel(
            records, currency=args.currency, cost_basis=args.cost_basis
        )
        panel = result.eligible_panel
        dataset_quality = result.quality
        if not result.is_sufficient:
            print("Dataset eligibility failed:")
            print(result.insufficient_reason())
            return 2

    supervised = build_supervised(panel)
    if supervised.frame.empty:
        print("No supervised training rows could be built.")
        return 2
    splits = make_splits(supervised)
    print(json.dumps([*splits.messages], indent=2, default=str))
    if not splits.is_valid:
        print("The chronological split plan is incomplete; training was not started.")
        return 2

    config = TrainingConfig.from_ml_config(
        currency=args.currency, cost_basis=args.cost_basis
    )
    hyperparameters = dict(config.hyperparameters)
    if args.max_depth is not None:
        hyperparameters["max_depth"] = args.max_depth
    if args.n_estimators is not None:
        hyperparameters["n_estimators"] = args.n_estimators
    if args.early_stopping_rounds is not None:
        hyperparameters["early_stopping_rounds"] = args.early_stopping_rounds
    config = replace(config, hyperparameters=hyperparameters)

    result = train_forecast_model(
        supervised,
        splits,
        dataset_quality=dataset_quality,
        config=config,
        version=args.version,
        save_bundle=not args.no_save,
        track_mlflow=args.track_mlflow or None,
    )

    final = result.metrics.get("final_test", {})
    print("Final test metrics (out-of-sample):")
    for key in ("mae", "rmse", "wape", "pinball_mean", "interval_coverage", "quantile_crossing_rate"):
        print(f"  {key}: {final.get(key)}")
    print("Promotion gates:")
    for gate in result.gates:
        print(f"  [{'PASS' if gate['passed'] else 'FAIL'}] {gate['name']}: {gate['detail']}")
    print(f"Candidate version: {result.version}")
    if result.bundle_path:
        print(f"Bundle: {result.bundle_path}")
    for message in result.messages:
        print(message)
    return 0 if result.passes_gates else 2


if __name__ == "__main__":
    raise SystemExit(main())
