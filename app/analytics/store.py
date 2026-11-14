"""DuckDB-backed access to normalized Parquet billing data.

The store keeps all monetary dimensions intact: callers receive records, not
pre-summed totals, so every aggregation can group by currency and cost basis.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from app.core.config import get_app_config
from app.processing.normalization import empty_normalized_frame

_DATE_EXPRESSION = "COALESCE(charge_period_start, usage_start)"


@dataclass(slots=True)
class CostFilter:
    """Filters applied to normalized billing records."""

    providers: tuple[str, ...] = ()
    billing_accounts: tuple[str, ...] = ()
    services: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    currencies: tuple[str, ...] = ()
    cost_basis: str | None = None
    start_date: dt.date | None = None
    end_date: dt.date | None = None

    def is_empty(self) -> bool:
        return not any(
            [
                self.providers,
                self.billing_accounts,
                self.services,
                self.regions,
                self.currencies,
                self.cost_basis,
                self.start_date,
                self.end_date,
            ]
        )


@dataclass(slots=True)
class StoreDimensions:
    providers: list[str] = field(default_factory=list)
    billing_accounts: list[str] = field(default_factory=list)
    services: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    currencies: list[str] = field(default_factory=list)
    cost_bases: list[str] = field(default_factory=list)
    date_min: dt.date | None = None
    date_max: dt.date | None = None
    records: int = 0


class CostStore:
    """Query normalized billing Parquet files with DuckDB."""

    def __init__(
        self,
        processed_dir: Path | str | None = None,
        import_ids: list[str] | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir or get_app_config().storage.processed_dir)
        self.import_ids = import_ids

    # ------------------------------------------------------------------ helpers
    def files(self) -> list[Path]:
        if self.import_ids is not None:
            files = [self.processed_dir / f"{import_id}.parquet" for import_id in self.import_ids]
            return [path for path in files if path.exists()]
        return sorted(self.processed_dir.glob("*.parquet"))

    def has_data(self) -> bool:
        return bool(self.files())

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(database=":memory:")

    def _from_clause(self, files: list[Path]) -> str:
        quoted = ", ".join(
            "'" + str(path).replace("'", "''") + "'" for path in files
        )
        return f"read_parquet([{quoted}], union_by_name=true)"

    @staticmethod
    def _where(filters: CostFilter | None) -> tuple[str, list[Any]]:
        if filters is None:
            return "", []
        clauses: list[str] = []
        params: list[Any] = []
        for column, values in (
            ("provider", filters.providers),
            ("billing_account_id", filters.billing_accounts),
            ("service_name", filters.services),
            ("region", filters.regions),
            ("currency", filters.currencies),
        ):
            if values:
                placeholders = ", ".join("?" for _ in values)
                clauses.append(f"{column} IN ({placeholders})")
                params.extend(values)
        if filters.cost_basis:
            clauses.append("cost_basis = ?")
            params.append(filters.cost_basis)
        if filters.start_date:
            clauses.append(f"{_DATE_EXPRESSION} >= ?")
            params.append(dt.datetime.combine(filters.start_date, dt.time.min))
        if filters.end_date:
            clauses.append(f"{_DATE_EXPRESSION} < ?")
            params.append(dt.datetime.combine(filters.end_date + dt.timedelta(days=1), dt.time.min))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return where, params

    # -------------------------------------------------------------------- reads
    def load(
        self,
        filters: CostFilter | None = None,
        *,
        columns: list[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        files = self.files()
        if not files:
            return empty_normalized_frame()
        select = ", ".join(columns) if columns else "*"
        where, params = self._where(filters)
        sql = f"SELECT {select} FROM {self._from_clause(files)}{where} ORDER BY {_DATE_EXPRESSION}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        connection = self._connect()
        try:
            return connection.execute(sql, params).fetch_df()
        finally:
            connection.close()

    def dimensions(self) -> StoreDimensions:
        files = self.files()
        if not files:
            return StoreDimensions()
        source = self._from_clause(files)
        connection = self._connect()
        try:
            def distinct(column: str) -> list[str]:
                rows = connection.execute(
                    f"SELECT DISTINCT {column} FROM {source} WHERE {column} IS NOT NULL ORDER BY {column}"
                ).fetchall()
                return [str(row[0]) for row in rows if row[0] is not None]

            bounds = connection.execute(
                f"SELECT MIN({_DATE_EXPRESSION}), MAX({_DATE_EXPRESSION}), COUNT(*) FROM {source}"
            ).fetchone()
            return StoreDimensions(
                providers=distinct("provider"),
                billing_accounts=distinct("billing_account_id"),
                services=distinct("service_name"),
                regions=distinct("region"),
                currencies=distinct("currency"),
                cost_bases=distinct("cost_basis"),
                date_min=bounds[0].date() if bounds and bounds[0] is not None else None,
                date_max=bounds[1].date() if bounds and bounds[1] is not None else None,
                records=int(bounds[2]) if bounds else 0,
            )
        finally:
            connection.close()
