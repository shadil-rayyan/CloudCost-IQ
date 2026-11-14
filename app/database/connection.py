"""SQLite metadata storage.

SQLite holds *metadata only* (imports, forecasts, recommendations, workspace
settings). Billing records live in Parquet and are queried with DuckDB.
Every call opens a short-lived connection so the module is safe to use from
both Streamlit (threaded) and FastAPI (async workers).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.core.config import get_app_config

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS imports (
    import_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    filename TEXT NOT NULL,
    stored_filename TEXT,
    file_size_bytes INTEGER,
    file_fingerprint TEXT,
    file_format TEXT,
    schema_name TEXT,
    schema_version TEXT,
    status TEXT NOT NULL,
    dataset_kind TEXT NOT NULL DEFAULT 'real',
    cost_basis TEXT,
    currency TEXT,
    date_coverage_start TEXT,
    date_coverage_end TEXT,
    invoice_month_min TEXT,
    invoice_month_max TEXT,
    records_read INTEGER DEFAULT 0,
    valid_records INTEGER DEFAULT 0,
    rejected_records INTEGER DEFAULT 0,
    duplicate_records INTEGER DEFAULT 0,
    warning_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    warnings_json TEXT,
    error_message TEXT,
    import_config_json TEXT,
    uploaded_at TEXT,
    completed_at TEXT,
    processing_seconds REAL,
    replaced_import_ids TEXT
);
CREATE INDEX IF NOT EXISTS idx_imports_provider ON imports(provider);
CREATE INDEX IF NOT EXISTS idx_imports_fingerprint ON imports(file_fingerprint);
CREATE INDEX IF NOT EXISTS idx_imports_status ON imports(status);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    provider TEXT,
    billing_account_id TEXT,
    service_name TEXT,
    currency TEXT NOT NULL,
    cost_basis TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    generated_at TEXT NOT NULL,
    as_of_date TEXT,
    history_start TEXT,
    history_end TEXT,
    status TEXT NOT NULL,
    notes TEXT,
    results_path TEXT,
    metrics_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_forecasts_model ON forecasts(model_version);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id TEXT PRIMARY KEY,
    rule_id TEXT NOT NULL,
    title TEXT NOT NULL,
    provider TEXT,
    account TEXT,
    service TEXT,
    observation TEXT,
    evidence TEXT,
    suggested_action TEXT,
    risk TEXT,
    estimated_savings REAL,
    currency TEXT,
    priority TEXT NOT NULL DEFAULT 'medium',
    status TEXT NOT NULL DEFAULT 'new',
    period_start TEXT,
    period_end TEXT,
    created_at TEXT,
    updated_at TEXT,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_recommendations_status ON recommendations(status);

CREATE TABLE IF NOT EXISTS workspace_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT
);
"""


def default_database_path() -> Path:
    return Path(get_app_config().storage.database_path)


def connect(database_path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection with sanity pragmas and dictionary-like rows."""
    path = Path(database_path) if database_path else default_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


@contextmanager
def session(database_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    connection = connect(database_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(database_path: Path | str | None = None) -> Path:
    """Create tables if they do not exist and return the database path."""
    path = Path(database_path) if database_path else default_database_path()
    with session(path) as connection:
        connection.executescript(SCHEMA_SQL)
    return path
