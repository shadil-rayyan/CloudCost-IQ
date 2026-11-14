"""Validate a candidate model bundle and optionally promote it.

Usage::

    python scripts/validate_model.py --version cciq-xgb-q30-20260101-120000
    python scripts/validate_model.py --version <version> --promote

The command always runs every artifact-contract gate. Promotion only happens
when ``--promote`` is supplied **and** all gates pass. Training never promotes a
model automatically.
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
from app.ml.artifact_manager import ArtifactManager  # noqa: E402
from app.ml.model_contract import validate_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Model version or bundle directory")
    parser.add_argument(
        "--stage",
        default="auto",
        choices=("auto", "candidates", "staging", "production"),
        help="Stage to resolve the version in",
    )
    parser.add_argument("--promote", action="store_true", help="Promote to production after passing gates")
    parser.add_argument(
        "--to-stage",
        default="production",
        choices=("staging", "production"),
        help="Destination stage when --promote is used",
    )
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        help="Move the current production model back to candidates before promotion",
    )
    parser.add_argument("--json", action="store_true", help="Print the full validation report as JSON")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    manager = ArtifactManager()

    try:
        path, stage = manager.find_bundle_path(args.version, stage=args.stage)
    except Exception as exc:  # ModelNotAvailableError and friends
        print(f"Model bundle not found: {exc}")
        return 1

    print(f"Validating {path} (stage={stage})")
    validation = validate_bundle(path)
    if args.json:
        print(json.dumps(validation.to_dict(), indent=2, default=str))
    else:
        for check in validation.checks:
            print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
        for error in validation.errors:
            print(f"  error: {error}")

    if not validation.passed:
        print("The bundle did not pass every validation gate. It will not be promoted.")
        return 2

    if not args.promote:
        print("Validation passed. Promotion was not requested.")
        return 0

    try:
        destination = manager.promote(
            path.name,
            from_stage=stage if stage != "unknown" else "candidates",
            to_stage=args.to_stage,
            archive_existing=args.archive_existing,
        )
    except Exception as exc:
        print(f"Promotion failed: {exc}")
        return 3

    print(f"Promoted {path.name} to {destination}")
    print("The forecast dashboard will now serve this bundle after the next cache refresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
