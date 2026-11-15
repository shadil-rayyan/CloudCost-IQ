"""Import orchestration: stage, analyze, validate, normalize and persist.

The dashboard uses the two-phase flow (``stage`` -> ``analyze`` -> user
confirmation -> ``commit``). The API/CLI can use :meth:`ImportManager.import_file`
which performs all phases in one call.
"""

from __future__ import annotations

import datetime as dt
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import AppConfig, get_app_config
from app.core.exceptions import CloudCostError, EmptyDataError, IngestionError
from app.database.models import ImportRecord, ImportStatus
from app.database.repositories import ImportRepository
from app.ingestion.base_parser import ReadResult, read_tabular_file
from app.ingestion.file_detector import DetectedSchema, detect_schema
from app.processing.deduplication import (
    DuplicateReport,
    analyse_coverage_overlap,
    detect_within_batch,
    find_duplicate_imports,
    merge_reports,
)
from app.processing.normalization import empty_normalized_frame, finalize_records
from app.processing.storage import ProcessedStore
from app.processing.validation import ValidationReport, validate_batch
from app.utils.dates import utc_now
from app.utils.files import human_size, sanitize_filename, sha256_file
from app.utils.formatting import safe_json_dumps


@dataclass(slots=True)
class ImportOptions:
    provider: str = "Auto"
    cost_basis: str = "billed"
    dataset_kind: str = "real"
    replace_overlapping: bool = False
    allow_duplicate_import: bool = True


@dataclass(slots=True)
class StagedUpload:
    import_id: str
    path: Path
    filename: str
    fingerprint: str
    size_bytes: int
    staging_directory: Path

    def cleanup(self) -> None:
        shutil.rmtree(self.staging_directory, ignore_errors=True)


@dataclass(slots=True)
class ImportAnalysis:
    staged: StagedUpload
    detected: DetectedSchema
    cost_basis: str
    dataset_kind: str
    report: ValidationReport
    duplicates: DuplicateReport
    notes: list[str] = field(default_factory=list)
    preview: pd.DataFrame = field(default_factory=pd.DataFrame)
    normalized_preview: pd.DataFrame = field(default_factory=pd.DataFrame)
    file_format: str = "csv"
    schema_version: str = "1.0"

    @property
    def is_importable(self) -> bool:
        return self.report.is_importable

    @property
    def warning_issues(self) -> list[dict[str, Any]]:
        return [issue.to_dict() for issue in self.report.issues if issue.severity != "error"]

    def summary(self) -> dict[str, Any]:
        valid = self.report.valid_frame
        return {
            "import_id": self.staged.import_id,
            "filename": self.staged.filename,
            "file_size": human_size(self.staged.size_bytes),
            "provider": self.detected.provider,
            "provider_score": round(self.detected.score, 3),
            "schema_name": self.detected.parser.schema_name,
            "cost_basis": self.cost_basis,
            "dataset_kind": self.dataset_kind,
            "records_read": self.report.records_read,
            "valid_records": self.report.valid_records,
            "rejected_records": self.report.rejected_records,
            "duplicate_rows": self.duplicates.exact_duplicate_rows,
            "exact_duplicate_imports": self.duplicates.duplicate_import_ids,
            "replaceable_imports": self.duplicates.replaceable_import_ids,
            "partial_overlap_imports": self.duplicates.partial_overlap_import_ids,
            "currencies": sorted(valid["currency"].dropna().unique().tolist()) if not valid.empty else [],
            "coverage_start": _coverage(valid)[0],
            "coverage_end": _coverage(valid)[1],
            "notes": self.notes,
        }


@dataclass(slots=True)
class ImportOutcome:
    import_id: str
    status: str
    provider: str
    filename: str
    records_read: int = 0
    valid_records: int = 0
    rejected_records: int = 0
    duplicate_records: int = 0
    currency: str | None = None
    cost_basis: str | None = None
    coverage_start: str | None = None
    coverage_end: str | None = None
    schema_name: str | None = None
    schema_version: str | None = None
    warnings: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None
    replaced_import_ids: list[str] = field(default_factory=list)
    processing_seconds: float = 0.0
    results_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "import_id": self.import_id,
            "status": self.status,
            "provider": self.provider,
            "filename": self.filename,
            "records_read": self.records_read,
            "valid_records": self.valid_records,
            "rejected_records": self.rejected_records,
            "duplicate_records": self.duplicate_records,
            "currency": self.currency,
            "cost_basis": self.cost_basis,
            "coverage_start": self.coverage_start,
            "coverage_end": self.coverage_end,
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "warnings": self.warnings,
            "warning_count": sum(int(item.get("count", 0)) for item in self.warnings),
            "error_message": self.error_message,
            "replaced_import_ids": self.replaced_import_ids,
            "processing_seconds": round(self.processing_seconds, 3),
            "results_path": self.results_path,
        }


def _coverage(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    if frame is None or frame.empty:
        return None, None
    dates = pd.to_datetime(frame["usage_start"], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return dates.min().date().isoformat(), dates.max().date().isoformat()


class ImportManager:
    """Coordinates the complete billing ingestion pipeline."""

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        repository: ImportRepository | None = None,
        store: ProcessedStore | None = None,
    ) -> None:
        self.config = config or get_app_config()
        self.config.ensure_directories()
        self.repository = repository or ImportRepository(self.config.storage.database_path)
        self.store = store or ProcessedStore(self.config.storage.processed_dir)

    # ------------------------------------------------------------------ staging
    def stage_upload(self, data: bytes | Path | str, filename: str) -> StagedUpload:
        """Persist an upload in an isolated staging directory."""
        safe_name = sanitize_filename(filename)
        if isinstance(data, (str, Path)):
            source = Path(data)
            payload_path = source
            size = source.stat().st_size
        else:
            size = len(data)
            payload_path = None
        if size == 0:
            raise EmptyDataError("The uploaded file is empty")
        if size > self.config.uploads.max_file_size_bytes:
            raise IngestionError(
                f"File is {human_size(size)}, above the {self.config.uploads.max_file_size_mb} MB limit"
            )

        import_id = f"imp-{utc_now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        staging_directory = self.config.storage.raw_dir / "_staging" / import_id
        staging_directory.mkdir(parents=True, exist_ok=True)
        destination = staging_directory / safe_name
        if payload_path is not None:
            shutil.copy2(payload_path, destination)
        else:
            destination.write_bytes(data)  # type: ignore[arg-type]
        ensure_inside = self.config.storage.raw_dir.resolve()
        if ensure_inside not in destination.resolve().parents:
            raise IngestionError("Staged upload escaped the configured storage directory")
        return StagedUpload(
            import_id=import_id,
            path=destination,
            filename=safe_name,
            fingerprint=sha256_file(destination),
            size_bytes=destination.stat().st_size,
            staging_directory=staging_directory,
        )

    # ----------------------------------------------------------------- analysis
    def analyze(self, staged: StagedUpload, options: ImportOptions | None = None) -> ImportAnalysis:
        options = options or ImportOptions()
        read_result: ReadResult | None = None
        try:
            read_result = read_tabular_file(
                staged.path,
                upload_config=self.config.uploads,
                max_rows=self.config.uploads.max_rows,
            )
            detected = detect_schema(read_result.frame, options.provider)
            notes = list(read_result.notes)
            parsed = detected.parser.parse(
                read_result.frame,
                cost_basis=options.cost_basis,
                notes=notes,
            )
            normalized = finalize_records(
                parsed.frame,
                provider=parsed.provider,
                import_batch_id=staged.import_id,
                source_file_id=staged.fingerprint,
                cost_basis=options.cost_basis,
                raw_frame=parsed.raw_frame,
            )
            report = validate_batch(normalized)

            existing = self.repository.list()
            duplicates = detect_within_batch(report.valid_frame)
            duplicates.duplicate_import_ids = find_duplicate_imports(
                staged.fingerprint, existing
            )
            if report.valid_records:
                valid = report.valid_frame
                coverage_start = pd.to_datetime(valid["usage_start"], errors="coerce").min()
                coverage_end = pd.to_datetime(valid["usage_start"], errors="coerce").max()
                overlap = analyse_coverage_overlap(
                    provider=detected.provider,
                    start=coverage_start.date() if pd.notna(coverage_start) else None,
                    end=coverage_end.date() if pd.notna(coverage_end) else None,
                    existing_imports=existing,
                )
                duplicates = merge_reports(duplicates, overlap)

            preview = read_result.frame.head(self.config.uploads.preview_rows)
            normalized_preview = report.valid_frame.head(self.config.uploads.preview_rows)
            return ImportAnalysis(
                staged=staged,
                detected=detected,
                cost_basis=options.cost_basis,
                dataset_kind=options.dataset_kind,
                report=report,
                duplicates=duplicates,
                notes=notes,
                preview=preview,
                normalized_preview=normalized_preview,
                file_format=read_result.file_format,
                schema_version=parsed.schema_version,
            )
        finally:
            if read_result is not None:
                read_result.cleanup()

    # ------------------------------------------------------------------- commit
    def commit(self, analysis: ImportAnalysis, options: ImportOptions | None = None) -> ImportOutcome:
        options = options or ImportOptions(
            cost_basis=analysis.cost_basis, dataset_kind=analysis.dataset_kind
        )
        started = time.perf_counter()
        staged = analysis.staged
        report = analysis.report
        provider = analysis.detected.provider

        warnings: list[dict[str, Any]] = []
        for issue in report.issues:
            if issue.severity != "error":
                warnings.append(issue.to_dict())
        for note in analysis.notes:
            warnings.append(
                {"code": "parser_note", "severity": "info", "message": note, "count": 1}
            )
        if analysis.duplicates.duplicate_import_ids:
            warnings.append(
                {
                    "code": "duplicate_import",
                    "severity": "warning",
                    "message": (
                        "This exact file fingerprint was imported before "
                        f"({', '.join(analysis.duplicates.duplicate_import_ids[:3])})."
                    ),
                    "count": len(analysis.duplicates.duplicate_import_ids),
                }
            )
        if analysis.duplicates.partial_overlap_import_ids:
            warnings.append(
                {
                    "code": "partial_coverage_overlap",
                    "severity": "warning",
                    "message": (
                        "This file partially overlaps existing imports that were NOT replaced: "
                        + ", ".join(analysis.duplicates.partial_overlap_import_ids[:5])
                    ),
                    "count": len(analysis.duplicates.partial_overlap_import_ids),
                }
            )

        outcome = ImportOutcome(
            import_id=staged.import_id,
            status=ImportStatus.COMPLETED,
            provider=provider,
            filename=staged.filename,
            records_read=report.records_read,
            valid_records=report.valid_records,
            rejected_records=report.rejected_records,
            duplicate_records=analysis.duplicates.exact_duplicate_rows,
            cost_basis=analysis.cost_basis,
            schema_name=analysis.detected.parser.schema_name,
            schema_version=analysis.schema_version,
            warnings=warnings,
        )

        if not report.is_importable:
            outcome.status = ImportStatus.FAILED
            outcome.error_message = (
                "No valid billing records remained after validation. "
                "Review the validation issues before retrying."
            )
            self._record_failure(analysis, options, outcome)
            staged.cleanup()
            outcome.processing_seconds = time.perf_counter() - started
            return outcome

        valid = report.valid_frame
        currencies = sorted(valid["currency"].dropna().astype(str).unique().tolist())
        outcome.currency = currencies[0] if len(currencies) == 1 else None
        coverage_start, coverage_end = _coverage(valid)
        outcome.coverage_start = coverage_start
        outcome.coverage_end = coverage_end

        final_raw = self.config.storage.raw_dir / f"{staged.import_id}__{staged.filename}"
        try:
            self.store.write(valid, staged.import_id)
            outcome.results_path = str(self.store.path_for(staged.import_id))
            if not report.rejected_frame.empty:
                rejected_path = self.config.storage.rejected_dir / f"{staged.import_id}__rejected.parquet"
                report.rejected_frame.to_parquet(rejected_path, index=False)
            if staged.path.exists():
                shutil.move(str(staged.path), final_raw)
        except Exception as exc:  # pragma: no cover - protective path
            self.store.delete(staged.import_id)
            outcome.status = ImportStatus.FAILED
            outcome.error_message = "Failed to persist normalized billing records"
            self._record_failure(analysis, options, outcome)
            staged.cleanup()
            outcome.processing_seconds = time.perf_counter() - started
            raise IngestionError(outcome.error_message) from exc

        replaced: list[str] = []
        if options.replace_overlapping and analysis.duplicates.replaceable_import_ids:
            replaced = analysis.duplicates.replaceable_import_ids
            for import_id in replaced:
                self.store.delete(import_id)
            self.repository.mark_superseded(replaced)

        record = ImportRecord(
            import_id=staged.import_id,
            provider=provider,
            filename=staged.filename,
            status=ImportStatus.COMPLETED,
            stored_filename=final_raw.name,
            file_size_bytes=staged.size_bytes,
            file_fingerprint=staged.fingerprint,
            file_format=analysis.file_format,
            schema_name=analysis.detected.parser.schema_name,
            schema_version=outcome.schema_version,
            dataset_kind=options.dataset_kind,
            cost_basis=analysis.cost_basis,
            currency=outcome.currency,
            date_coverage_start=dt.date.fromisoformat(coverage_start) if coverage_start else None,
            date_coverage_end=dt.date.fromisoformat(coverage_end) if coverage_end else None,
            invoice_month_min=_month_bound(valid, "min"),
            invoice_month_max=_month_bound(valid, "max"),
            records_read=report.records_read,
            valid_records=report.valid_records,
            rejected_records=report.rejected_records,
            duplicate_records=analysis.duplicates.exact_duplicate_rows,
            warning_count=int(sum(item.get("count", 0) for item in warnings)),
            error_count=report.rejected_records,
            warnings=warnings,
            import_config={
                "requested_provider": options.provider,
                "detected_provider": provider,
                "detection_score": analysis.detected.score,
                "detection_candidates": analysis.detected.candidates,
                "cost_basis": analysis.cost_basis,
                "replace_overlapping": options.replace_overlapping,
                "duplicate_row_report": analysis.duplicates.to_dict(),
            },
            uploaded_at=utc_now(),
            completed_at=utc_now(),
            processing_seconds=time.perf_counter() - started,
            replaced_import_ids=replaced,
        )
        self.repository.save(record)
        staged.cleanup()
        outcome.replaced_import_ids = replaced
        outcome.processing_seconds = record.processing_seconds or 0.0
        return outcome

    def import_file(
        self,
        data: bytes | Path | str,
        filename: str,
        options: ImportOptions | None = None,
    ) -> ImportOutcome:
        """One-shot import used by the API and scripts."""
        options = options or ImportOptions()
        staged = self.stage_upload(data, filename)
        try:
            analysis = self.analyze(staged, options)
        except CloudCostError as exc:
            outcome = ImportOutcome(
                import_id=staged.import_id,
                status=ImportStatus.FAILED,
                provider=options.provider,
                filename=staged.filename,
                cost_basis=options.cost_basis,
                error_message=str(exc),
            )
            self._record_stage_failure(staged, options, outcome)
            staged.cleanup()
            raise
        return self.commit(analysis, options)

    def cancel(self, staged: StagedUpload) -> None:
        staged.cleanup()

    # ------------------------------------------------------------------ helpers
    def _record_failure(
        self, analysis: ImportAnalysis, options: ImportOptions, outcome: ImportOutcome
    ) -> None:
        record = ImportRecord(
            import_id=analysis.staged.import_id,
            provider=analysis.detected.provider,
            filename=analysis.staged.filename,
            status=ImportStatus.FAILED,
            stored_filename=None,
            file_size_bytes=analysis.staged.size_bytes,
            file_fingerprint=analysis.staged.fingerprint,
            file_format=analysis.detected.parser.schema_name,
            schema_name=analysis.detected.parser.schema_name,
            schema_version=outcome.schema_version,
            dataset_kind=options.dataset_kind,
            cost_basis=analysis.cost_basis,
            records_read=analysis.report.records_read,
            valid_records=analysis.report.valid_records,
            rejected_records=analysis.report.rejected_records,
            warning_count=len(outcome.warnings),
            error_count=analysis.report.rejected_records,
            warnings=outcome.warnings,
            error_message=outcome.error_message,
            import_config={"requested_provider": options.provider},
            uploaded_at=utc_now(),
            completed_at=utc_now(),
        )
        self.repository.save(record)

    def _record_stage_failure(
        self, staged: StagedUpload, options: ImportOptions, outcome: ImportOutcome
    ) -> None:
        record = ImportRecord(
            import_id=staged.import_id,
            provider=options.provider,
            filename=staged.filename,
            status=ImportStatus.FAILED,
            stored_filename=None,
            file_size_bytes=staged.size_bytes,
            file_fingerprint=staged.fingerprint,
            dataset_kind=options.dataset_kind,
            cost_basis=options.cost_basis,
            error_message=outcome.error_message,
            import_config={"requested_provider": options.provider},
            uploaded_at=utc_now(),
            completed_at=utc_now(),
        )
        self.repository.save(record)


def _month_bound(frame: pd.DataFrame, func: str) -> str | None:
    if frame is None or frame.empty or "invoice_month" not in frame:
        return None
    values = frame["invoice_month"].dropna().astype(str)
    if values.empty:
        return None
    return values.min() if func == "min" else values.max()
