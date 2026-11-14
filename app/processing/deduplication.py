"""Duplicate and re-import handling.

Billing exports are frequently re-delivered and revised. CloudCost IQ treats
duplicates cautiously:

* exact duplicate *rows* are reported but never deleted automatically;
* an identical file fingerprint is reported as a duplicate import;
* overlapping date coverage is reported, and replacement only happens when the
  caller explicitly asks for it and the previous import is fully covered.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.core.constants import NORMALIZED_COLUMNS
from app.database.models import ImportRecord


@dataclass(slots=True)
class DuplicateReport:
    exact_duplicate_rows: int = 0
    records_considered: int = 0
    duplicate_import_ids: list[str] = field(default_factory=list)
    replaceable_import_ids: list[str] = field(default_factory=list)
    partial_overlap_import_ids: list[str] = field(default_factory=list)
    overlap_details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def exact_duplicate_ratio(self) -> float:
        if not self.records_considered:
            return 0.0
        return self.exact_duplicate_rows / self.records_considered

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_duplicate_rows": int(self.exact_duplicate_rows),
            "records_considered": int(self.records_considered),
            "exact_duplicate_ratio": round(self.exact_duplicate_ratio, 6),
            "duplicate_import_ids": self.duplicate_import_ids,
            "replaceable_import_ids": self.replaceable_import_ids,
            "partial_overlap_import_ids": self.partial_overlap_import_ids,
            "overlap_details": self.overlap_details,
        }


def exact_duplicate_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify rows that are identical across every normalized column."""
    if frame is None or frame.empty:
        return pd.Series(dtype=bool)
    comparable = [column for column in NORMALIZED_COLUMNS if column in frame.columns]
    comparable = [column for column in comparable if column not in {"record_id", "raw_fields_json"}]
    if not comparable:
        return pd.Series(False, index=frame.index)
    return frame.duplicated(subset=comparable, keep="first")


def detect_within_batch(frame: pd.DataFrame) -> DuplicateReport:
    mask = exact_duplicate_mask(frame)
    return DuplicateReport(
        exact_duplicate_rows=int(mask.sum()) if len(mask) else 0,
        records_considered=int(len(frame)) if frame is not None else 0,
    )


def find_duplicate_imports(
    fingerprint: str | None, imports: list[ImportRecord]
) -> list[str]:
    if not fingerprint:
        return []
    return [record.import_id for record in imports if record.file_fingerprint == fingerprint]


def _date_range(record: ImportRecord) -> tuple[dt.date, dt.date] | None:
    if record.date_coverage_start and record.date_coverage_end:
        return record.date_coverage_start, record.date_coverage_end
    return None


def analyse_coverage_overlap(
    *,
    provider: str,
    start: dt.date | None,
    end: dt.date | None,
    existing_imports: list[ImportRecord],
) -> DuplicateReport:
    """Classify overlapping imports as fully replaceable or partial conflicts."""
    report = DuplicateReport()
    if start is None or end is None:
        return report
    for record in existing_imports:
        if record.provider != provider or record.status != "completed":
            continue
        bounds = _date_range(record)
        if bounds is None:
            continue
        other_start, other_end = bounds
        if other_end < start or other_start > end:
            continue
        fully_covered = other_start >= start and other_end <= end
        detail = {
            "import_id": record.import_id,
            "filename": record.filename,
            "coverage_start": other_start.isoformat(),
            "coverage_end": other_end.isoformat(),
            "fully_covered": fully_covered,
        }
        report.overlap_details.append(detail)
        if fully_covered:
            report.replaceable_import_ids.append(record.import_id)
        else:
            report.partial_overlap_import_ids.append(record.import_id)
    return report


def merge_reports(base: DuplicateReport, *others: DuplicateReport) -> DuplicateReport:
    merged = DuplicateReport(
        exact_duplicate_rows=base.exact_duplicate_rows,
        records_considered=base.records_considered,
    )
    for report in (base, *others):
        merged.duplicate_import_ids = list(dict.fromkeys(merged.duplicate_import_ids + report.duplicate_import_ids))
        merged.replaceable_import_ids = list(
            dict.fromkeys(merged.replaceable_import_ids + report.replaceable_import_ids)
        )
        merged.partial_overlap_import_ids = list(
            dict.fromkeys(merged.partial_overlap_import_ids + report.partial_overlap_import_ids)
        )
        merged.overlap_details.extend(report.overlap_details)
        merged.exact_duplicate_rows = max(merged.exact_duplicate_rows, report.exact_duplicate_rows)
        merged.records_considered = max(merged.records_considered, report.records_considered)
    return merged
