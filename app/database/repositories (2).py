"""Repositories for import, forecast, recommendation and settings metadata."""

from __future__ import annotations

import datetime as dt
from typing import Any

from app.database.connection import init_db, session
from app.database.models import (
    ForecastRecord,
    ImportRecord,
    RecommendationRecord,
)
from app.utils.dates import utc_now
from app.utils.formatting import safe_json_dumps


def _json_loads(text: str | None, default: Any) -> Any:
    import json

    if not text:
        return default
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return default


def _iso(value: dt.date | dt.datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def _datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


class ImportRepository:
    """Persistence for billing import metadata."""

    def __init__(self, database_path=None) -> None:
        self.database_path = database_path
        init_db(database_path)

    def save(self, record: ImportRecord) -> None:
        payload = {
            "import_id": record.import_id,
            "provider": record.provider,
            "filename": record.filename,
            "stored_filename": record.stored_filename,
            "file_size_bytes": record.file_size_bytes,
            "file_fingerprint": record.file_fingerprint,
            "file_format": record.file_format,
            "schema_name": record.schema_name,
            "schema_version": record.schema_version,
            "status": str(record.status),
            "dataset_kind": str(record.dataset_kind),
            "cost_basis": record.cost_basis,
            "currency": record.currency,
            "date_coverage_start": _iso(record.date_coverage_start),
            "date_coverage_end": _iso(record.date_coverage_end),
            "invoice_month_min": record.invoice_month_min,
            "invoice_month_max": record.invoice_month_max,
            "records_read": int(record.records_read or 0),
            "valid_records": int(record.valid_records or 0),
            "rejected_records": int(record.rejected_records or 0),
            "duplicate_records": int(record.duplicate_records or 0),
            "warning_count": int(record.warning_count or 0),
            "error_count": int(record.error_count or 0),
            "warnings_json": safe_json_dumps(record.warnings),
            "error_message": record.error_message,
            "import_config_json": safe_json_dumps(record.import_config),
            "uploaded_at": _iso(record.uploaded_at),
            "completed_at": _iso(record.completed_at),
            "processing_seconds": record.processing_seconds,
            "replaced_import_ids": safe_json_dumps(record.replaced_import_ids),
        }
        columns = ", ".join(payload)
        placeholders = ", ".join(f":{key}" for key in payload)
        updates = ", ".join(f"{key}=excluded.{key}" for key in payload if key != "import_id")
        with session(self.database_path) as connection:
            connection.execute(
                f"INSERT INTO imports ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(import_id) DO UPDATE SET {updates}",
                payload,
            )

    def get(self, import_id: str) -> ImportRecord | None:
        with session(self.database_path) as connection:
            row = connection.execute(
                "SELECT * FROM imports WHERE import_id = ?", (import_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list(
        self,
        *,
        provider: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[ImportRecord]:
        query = "SELECT * FROM imports WHERE 1=1"
        params: list[Any] = []
        if provider:
            query += " AND provider = ?"
            params.append(provider)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY uploaded_at DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        with session(self.database_path) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows if row is not None]

    def find_by_fingerprint(self, fingerprint: str) -> list[ImportRecord]:
        with session(self.database_path) as connection:
            rows = connection.execute(
                "SELECT * FROM imports WHERE file_fingerprint = ? AND status = 'completed'",
                (fingerprint,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows if row is not None]

    def completed_import_ids(self) -> list[str]:
        with session(self.database_path) as connection:
            rows = connection.execute(
                "SELECT import_id FROM imports WHERE status = 'completed' ORDER BY uploaded_at"
            ).fetchall()
        return [row["import_id"] for row in rows]

    def mark_superseded(self, import_ids: list[str]) -> None:
        if not import_ids:
            return
        placeholders = ", ".join("?" for _ in import_ids)
        with session(self.database_path) as connection:
            connection.execute(
                f"UPDATE imports SET status = 'superseded' WHERE import_id IN ({placeholders})",
                import_ids,
            )

    def delete(self, import_id: str) -> bool:
        with session(self.database_path) as connection:
            cursor = connection.execute("DELETE FROM imports WHERE import_id = ?", (import_id,))
        return cursor.rowcount > 0

    def latest_upload(self) -> dt.datetime | None:
        with session(self.database_path) as connection:
            row = connection.execute(
                "SELECT MAX(uploaded_at) AS latest FROM imports WHERE status = 'completed'"
            ).fetchone()
        return _datetime(row["latest"]) if row else None

    def _row_to_record(self, row) -> ImportRecord:
        return ImportRecord(
            import_id=row["import_id"],
            provider=row["provider"],
            filename=row["filename"],
            stored_filename=row["stored_filename"],
            file_size_bytes=row["file_size_bytes"],
            file_fingerprint=row["file_fingerprint"],
            file_format=row["file_format"],
            schema_name=row["schema_name"],
            schema_version=row["schema_version"],
            status=row["status"],
            dataset_kind=row["dataset_kind"] or "real",
            cost_basis=row["cost_basis"],
            currency=row["currency"],
            date_coverage_start=_date(row["date_coverage_start"]),
            date_coverage_end=_date(row["date_coverage_end"]),
            invoice_month_min=row["invoice_month_min"],
            invoice_month_max=row["invoice_month_max"],
            records_read=row["records_read"] or 0,
            valid_records=row["valid_records"] or 0,
            rejected_records=row["rejected_records"] or 0,
            duplicate_records=row["duplicate_records"] or 0,
            warning_count=row["warning_count"] or 0,
            error_count=row["error_count"] or 0,
            warnings=_json_loads(row["warnings_json"], []),
            error_message=row["error_message"],
            import_config=_json_loads(row["import_config_json"], {}),
            uploaded_at=_datetime(row["uploaded_at"]),
            completed_at=_datetime(row["completed_at"]),
            processing_seconds=row["processing_seconds"],
            replaced_import_ids=_json_loads(row["replaced_import_ids"], []),
        )


class ForecastRepository:
    """Persistence for generated forecast metadata."""

    def __init__(self, database_path=None) -> None:
        self.database_path = database_path
        init_db(database_path)

    def save(self, record: ForecastRecord) -> None:
        payload = {
            "forecast_id": record.forecast_id,
            "model_version": record.model_version,
            "provider": record.provider,
            "billing_account_id": record.billing_account_id,
            "service_name": record.service_name,
            "currency": record.currency,
            "cost_basis": record.cost_basis,
            "horizon_days": int(record.horizon_days),
            "generated_at": _iso(record.generated_at),
            "as_of_date": _iso(record.as_of_date),
            "history_start": _iso(record.history_start),
            "history_end": _iso(record.history_end),
            "status": str(record.status),
            "notes": record.notes,
            "results_path": record.results_path,
            "metrics_json": safe_json_dumps(record.metrics),
        }
        columns = ", ".join(payload)
        placeholders = ", ".join(f":{key}" for key in payload)
        updates = ", ".join(f"{key}=excluded.{key}" for key in payload if key != "forecast_id")
        with session(self.database_path) as connection:
            connection.execute(
                f"INSERT INTO forecasts ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(forecast_id) DO UPDATE SET {updates}",
                payload,
            )

    def get(self, forecast_id: str) -> ForecastRecord | None:
        with session(self.database_path) as connection:
            row = connection.execute(
                "SELECT * FROM forecasts WHERE forecast_id = ?", (forecast_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list(self, limit: int = 50) -> list[ForecastRecord]:
        with session(self.database_path) as connection:
            rows = connection.execute(
                "SELECT * FROM forecasts ORDER BY generated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row) -> ForecastRecord:
        return ForecastRecord(
            forecast_id=row["forecast_id"],
            model_version=row["model_version"],
            provider=row["provider"],
            billing_account_id=row["billing_account_id"],
            service_name=row["service_name"],
            currency=row["currency"],
            cost_basis=row["cost_basis"],
            horizon_days=row["horizon_days"],
            generated_at=_datetime(row["generated_at"]) or utc_now(),
            as_of_date=_date(row["as_of_date"]),
            history_start=_date(row["history_start"]),
            history_end=_date(row["history_end"]),
            status=row["status"],
            notes=row["notes"],
            results_path=row["results_path"],
            metrics=_json_loads(row["metrics_json"], {}),
        )


class RecommendationRepository:
    """Persistence for generated recommendations and their review status."""

    def __init__(self, database_path=None) -> None:
        self.database_path = database_path
        init_db(database_path)

    def upsert_many(self, records: list[RecommendationRecord]) -> int:
        """Insert recommendations, preserving the status of existing ones."""
        if not records:
            return 0
        now = utc_now().isoformat()
        inserted = 0
        with session(self.database_path) as connection:
            for record in records:
                existing = connection.execute(
                    "SELECT status FROM recommendations WHERE recommendation_id = ?",
                    (record.recommendation_id,),
                ).fetchone()
                status = existing["status"] if existing else str(record.status)
                connection.execute(
                    """
                    INSERT INTO recommendations (
                        recommendation_id, rule_id, title, provider, account, service,
                        observation, evidence, suggested_action, risk, estimated_savings,
                        currency, priority, status, period_start, period_end,
                        created_at, updated_at, metadata_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(recommendation_id) DO UPDATE SET
                        title=excluded.title,
                        observation=excluded.observation,
                        evidence=excluded.evidence,
                        suggested_action=excluded.suggested_action,
                        risk=excluded.risk,
                        estimated_savings=excluded.estimated_savings,
                        currency=excluded.currency,
                        priority=excluded.priority,
                        period_start=excluded.period_start,
                        period_end=excluded.period_end,
                        metadata_json=excluded.metadata_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        record.recommendation_id,
                        record.rule_id,
                        record.title,
                        record.provider,
                        record.account,
                        record.service,
                        record.observation,
                        record.evidence,
                        record.suggested_action,
                        record.risk,
                        record.estimated_savings,
                        record.currency,
                        record.priority,
                        status,
                        _iso(record.period_start),
                        _iso(record.period_end),
                        _iso(record.created_at) or now,
                        now,
                        safe_json_dumps(record.metadata),
                    ),
                )
                inserted += 1
        return inserted

    def list(self, status: str | None = None) -> list[RecommendationRecord]:
        query = "SELECT * FROM recommendations"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += (
            " ORDER BY CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2"
            " WHEN 'low' THEN 3 ELSE 4 END, updated_at DESC"
        )
        with session(self.database_path) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def update_status(self, recommendation_id: str, status: str) -> bool:
        with session(self.database_path) as connection:
            cursor = connection.execute(
                "UPDATE recommendations SET status = ?, updated_at = ? WHERE recommendation_id = ?",
                (status, utc_now().isoformat(), recommendation_id),
            )
        return cursor.rowcount > 0

    def _row_to_record(self, row) -> RecommendationRecord:
        return RecommendationRecord(
            recommendation_id=row["recommendation_id"],
            rule_id=row["rule_id"],
            title=row["title"],
            provider=row["provider"],
            account=row["account"],
            service=row["service"],
            observation=row["observation"],
            evidence=row["evidence"],
            suggested_action=row["suggested_action"],
            risk=row["risk"],
            estimated_savings=row["estimated_savings"],
            currency=row["currency"],
            priority=row["priority"],
            status=row["status"],
            period_start=_date(row["period_start"]),
            period_end=_date(row["period_end"]),
            created_at=_datetime(row["created_at"]),
            updated_at=_datetime(row["updated_at"]),
            metadata=_json_loads(row["metadata_json"], {}),
        )


class SettingsRepository:
    """Key/value workspace settings stored in SQLite."""

    DEFAULTS = {
        "reporting_currency": "USD",
        "default_cost_basis": "billed",
        "monthly_budget": "",
        "workspace_name": "My workspace",
    }

    def __init__(self, database_path=None) -> None:
        self.database_path = database_path
        init_db(database_path)

    def get_all(self) -> dict[str, str]:
        values = dict(self.DEFAULTS)
        with session(self.database_path) as connection:
            rows = connection.execute("SELECT key, value FROM workspace_settings").fetchall()
        for row in rows:
            values[row["key"]] = row["value"]
        return values

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.get_all().get(key, default)

    def set_many(self, values: dict[str, Any]) -> None:
        now = utc_now().isoformat()
        with session(self.database_path) as connection:
            for key, value in values.items():
                connection.execute(
                    """
                    INSERT INTO workspace_settings (key, value, updated_at) VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                    """,
                    (key, "" if value is None else str(value), now),
                )
