"""Billing import endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.schemas import ImportDetail, ImportResponse
from app.core.config import get_app_config
from app.core.exceptions import CloudCostError
from app.database.repositories import ImportRepository
from app.ingestion.import_manager import ImportManager, ImportOptions
from app.utils.files import human_size

router = APIRouter()


@router.post("/imports", response_model=ImportResponse, status_code=201)
async def upload_import(
    file: UploadFile = File(...),
    provider: str = Form("Auto"),
    cost_basis: str = Form("billed"),
    dataset_kind: str = Form("real"),
    replace_overlapping: bool = Form(False),
) -> ImportResponse:
    config = get_app_config()
    payload = await file.read()
    if len(payload) > config.uploads.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File is {human_size(len(payload))}, above the "
                f"{config.uploads.max_file_size_mb} MB limit."
            ),
        )
    manager = ImportManager(config=config)
    options = ImportOptions(
        provider=provider,
        cost_basis=cost_basis,
        dataset_kind=dataset_kind,
        replace_overlapping=replace_overlapping,
    )
    try:
        outcome = manager.import_file(payload, file.filename or "upload", options)
    except CloudCostError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result = outcome.to_dict()
    return ImportResponse(**{key: result.get(key) for key in ImportResponse.model_fields})


@router.get("/imports", response_model=list[ImportDetail])
def list_imports(
    provider: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[ImportDetail]:
    repository = ImportRepository()
    items = repository.list(provider=provider, status=status, limit=max(1, min(limit, 500)))
    return [_detail(record) for record in items]


@router.get("/imports/{import_id}", response_model=ImportDetail)
def get_import(import_id: str) -> ImportDetail:
    record = ImportRepository().get(import_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Import not found")
    return _detail(record)


def _detail(record: Any) -> ImportDetail:
    return ImportDetail(
        import_id=record.import_id,
        status=str(record.status),
        provider=record.provider,
        filename=record.filename,
        records_read=record.records_read,
        valid_records=record.valid_records,
        rejected_records=record.rejected_records,
        duplicate_records=record.duplicate_records,
        currency=record.currency,
        cost_basis=record.cost_basis,
        coverage_start=record.date_coverage_start.isoformat() if record.date_coverage_start else None,
        coverage_end=record.date_coverage_end.isoformat() if record.date_coverage_end else None,
        schema_name=record.schema_name,
        schema_version=record.schema_version,
        warning_count=record.warning_count,
        error_message=record.error_message,
        processing_seconds=record.processing_seconds or 0.0,
        file_format=record.file_format,
        dataset_kind=str(record.dataset_kind or "real"),
        warnings=record.warnings,
        uploaded_at=record.uploaded_at.isoformat() if record.uploaded_at else None,
        completed_at=record.completed_at.isoformat() if record.completed_at else None,
    )
