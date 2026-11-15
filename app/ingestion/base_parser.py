"""Base classes and safe file reading for provider-specific parsers.

Parsers only *read and map* provider columns into the normalized shape. Shared
finalization (types, record IDs, raw preservation) happens once in
:mod:`app.processing.normalization`.
"""

from __future__ import annotations

import abc
import gzip
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import UploadConfig
from app.core.exceptions import (
    EmptyDataError,
    FileSafetyError,
    IngestionError,
    SchemaValidationError,
    UnsupportedFormatError,
)
from app.utils.dates import to_datetime_utc
from app.utils.files import ensure_within_directory, file_extension, safe_extract_zip


def normalize_key(name: object) -> str:
    """Normalize provider column names for alias matching.

    ``lineItem/UsageStartDate``, ``Line Item Usage Start Date`` and
    ``usage_start_date`` all resolve to ``line_item_usage_start_date`` /
    ``usage_start_date`` style keys, and camelCase (``ChargePeriodStart``)
    is split before lowercasing.
    """
    text = str(name).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower()


@dataclass(slots=True)
class ReadResult:
    frame: pd.DataFrame
    file_format: str
    source_path: Path
    rows: int
    notes: list[str] = field(default_factory=list)
    temporary_directory: Path | None = None

    def cleanup(self) -> None:
        if self.temporary_directory and self.temporary_directory.exists():
            shutil.rmtree(self.temporary_directory, ignore_errors=True)


def _read_csv_smart(path: Path) -> tuple[pd.DataFrame, str]:
    sample = ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            sample = handle.read(8192)
    except OSError as exc:
        raise IngestionError("Unable to read the uploaded CSV file") from exc

    delimiters = {",": sample.count(","), "\t": sample.count("\t"), ";": sample.count(";"), "|": sample.count("|")}
    delimiter = max(delimiters, key=delimiters.get) if any(delimiters.values()) else ","
    attempts = [
        {"encoding": "utf-8"},
        {"encoding": "utf-8-sig"},
        {"encoding": "latin-1"},
    ]
    last_error: Exception | None = None
    for options in attempts:
        try:
            frame = pd.read_csv(
                path,
                sep=delimiter,
                low_memory=False,
                **options,
            )
            return frame, options["encoding"]
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise EmptyDataError("The uploaded CSV file is empty") from exc
        except pd.errors.ParserError as exc:
            raise IngestionError("The uploaded CSV file could not be parsed") from exc
    raise IngestionError("The uploaded CSV file could not be decoded") from last_error


def read_tabular_file(
    path: Path | str,
    *,
    upload_config: UploadConfig | None = None,
    max_rows: int | None = None,
) -> ReadResult:
    """Safely read a CSV/Parquet file (optionally gzip or zip compressed).

    Enforces size limits, archive safety and row limits. The returned
    ``ReadResult`` owns any temporary directory and must be cleaned up.
    """
    config = upload_config or UploadConfig()
    path = Path(path)
    if not path.exists():
        raise IngestionError("Uploaded file was not found")
    if path.stat().st_size == 0:
        raise EmptyDataError("The uploaded file is empty")
    if path.stat().st_size > config.max_file_size_bytes:
        raise FileSafetyError(
            f"File exceeds the {config.max_file_size_mb} MB upload limit"
        )

    extension = file_extension(path)
    temporary_directory: Path | None = None
    notes: list[str] = []
    source = path

    try:
        if extension == "zip":
            temporary_directory = Path(tempfile.mkdtemp(prefix="cloudcost-zip-"))
            extracted = safe_extract_zip(
                path,
                temporary_directory,
                max_members=config.max_archive_members,
                max_uncompressed_bytes=config.max_uncompressed_size_bytes,
            )
            ordered = sorted(
                extracted,
                key=lambda item: (file_extension(item) != "csv", item.name),
            )
            source = ordered[0]
            if len(extracted) > 1:
                notes.append(
                    f"Archive contained {len(extracted)} supported files; imported '{source.name}' only."
                )
            extension = file_extension(source)
        elif path.suffix.lower() == ".gz":
            if path.with_suffix("").suffix.lower() == ".parquet":
                temporary_directory = Path(tempfile.mkdtemp(prefix="cloudcost-gz-"))
                decompressed = temporary_directory / path.with_suffix("").name
                with gzip.open(path, "rb") as source_handle, decompressed.open("wb") as sink:
                    copied = 0
                    for chunk in iter(lambda: source_handle.read(1024 * 256), b""):
                        copied += len(chunk)
                        if copied > config.max_uncompressed_size_bytes:
                            raise FileSafetyError("Compressed file exceeds the uncompressed size limit")
                        sink.write(chunk)
                source = decompressed
                extension = "parquet"
            else:
                temporary_directory = Path(tempfile.mkdtemp(prefix="cloudcost-gz-"))
                decompressed = temporary_directory / path.with_suffix("").name
                with gzip.open(path, "rb") as source_handle, decompressed.open("wb") as sink:
                    copied = 0
                    for chunk in iter(lambda: source_handle.read(1024 * 256), b""):
                        copied += len(chunk)
                        if copied > config.max_uncompressed_size_bytes:
                            raise FileSafetyError("Compressed file exceeds the uncompressed size limit")
                        sink.write(chunk)
                source = decompressed
                extension = "csv"

        if extension == "parquet":
            frame = pd.read_parquet(source)
            file_format = "parquet"
        elif extension == "csv":
            frame, encoding = _read_csv_smart(source)
            file_format = "csv"
            if encoding != "utf-8":
                notes.append(f"CSV decoded using {encoding}.")
        else:
            raise UnsupportedFormatError(
                "Unsupported file type. Upload CSV, Parquet, GZIP-compressed CSV/Parquet or ZIP."
            )

        if frame is None or frame.empty:
            raise EmptyDataError("The uploaded file contains no rows")
        if frame.shape[1] == 1 and frame.columns[0].startswith("Unnamed"):
            raise UnsupportedFormatError(
                "The file does not look like a delimited billing export (no header detected)"
            )
        if max_rows and len(frame) > max_rows:
            raise IngestionError(
                f"File contains {len(frame):,} rows, above the configured limit of {max_rows:,}"
            )
        if temporary_directory is not None:
            ensure_within_directory(source, temporary_directory)

        return ReadResult(
            frame=frame,
            file_format=file_format,
            source_path=source,
            rows=len(frame),
            notes=notes,
            temporary_directory=temporary_directory,
        )
    except Exception:
        if temporary_directory and temporary_directory.exists():
            shutil.rmtree(temporary_directory, ignore_errors=True)
        raise


class ColumnResolver:
    """Resolve normalized field names against provider-specific columns."""

    def __init__(self, columns: list[str] | pd.Index) -> None:
        self.columns = [str(column) for column in columns]
        self.mapping: dict[str, str] = {}
        for column in self.columns:
            self.mapping.setdefault(normalize_key(column), column)

    def find(self, *aliases: str) -> str | None:
        for alias in aliases:
            key = normalize_key(alias)
            if key in self.mapping:
                return self.mapping[key]
            collapsed = key.replace("_", "")
            for candidate_key, original in self.mapping.items():
                if candidate_key.replace("_", "") == collapsed:
                    return original
        return None

    def has(self, *aliases: str) -> bool:
        return self.find(*aliases) is not None

    def series(self, frame: pd.DataFrame, *aliases: str) -> pd.Series:
        column = self.find(*aliases)
        if column is None:
            return pd.Series([None] * len(frame), index=frame.index, dtype="object")
        return frame[column]

    def text(self, frame: pd.DataFrame, *aliases: str) -> pd.Series:
        values = self.series(frame, *aliases).astype("object")
        return values.map(lambda value: None if value is None or (isinstance(value, float) and pd.isna(value)) else str(value).strip() or None)

    def numeric(self, frame: pd.DataFrame, *aliases: str) -> pd.Series:
        return pd.to_numeric(self.series(frame, *aliases), errors="coerce")

    def datetime(self, frame: pd.DataFrame, *aliases: str) -> pd.Series:
        return to_datetime_utc(self.series(frame, *aliases))

    def first_text(self, frame: pd.DataFrame, groups: list[tuple[str, ...]]) -> pd.Series:
        result = pd.Series([None] * len(frame), index=frame.index, dtype="object")
        for aliases in groups:
            candidate = self.text(frame, *aliases)
            result = result.where(result.notna(), candidate)
        return result

    def first_numeric(self, frame: pd.DataFrame, groups: list[tuple[str, ...]]) -> pd.Series:
        result = pd.Series([float("nan")] * len(frame), index=frame.index)
        for aliases in groups:
            candidate = self.numeric(frame, *aliases)
            result = result.where(result.notna(), candidate)
        return result

    def columns_matching(self, prefix: str) -> list[str]:
        key_prefix = normalize_key(prefix)
        return [
            column
            for column in self.columns
            if normalize_key(column).startswith(key_prefix)
        ]


@dataclass(slots=True)
class ParsedBatch:
    """Provider-mapped records before shared finalization."""

    frame: pd.DataFrame
    raw_frame: pd.DataFrame
    provider: str
    schema_name: str
    schema_version: str
    notes: list[str] = field(default_factory=list)


class BaseBillingParser(abc.ABC):
    """Interface implemented by every provider parser."""

    provider: str = ""
    schema_name: str = ""
    description: str = ""
    #: Normalized column keys that strongly indicate this provider/schema.
    signature_columns: tuple[str, ...] = ()

    @abc.abstractmethod
    def detect_score(self, resolver: ColumnResolver) -> float:
        """Return a 0..1 confidence that this parser handles the columns."""

    @abc.abstractmethod
    def required_fields(self) -> dict[str, tuple[str, ...]]:
        """Normalized field -> accepted aliases required in the source file."""

    @abc.abstractmethod
    def map_frame(
        self,
        raw: pd.DataFrame,
        resolver: ColumnResolver,
        *,
        cost_basis: str,
        notes: list[str],
    ) -> pd.DataFrame:
        """Map provider columns into normalized field names (no finalization)."""

    def validate_schema(self, resolver: ColumnResolver) -> None:
        missing: list[str] = []
        for field_name, aliases in self.required_fields().items():
            if not resolver.has(*aliases):
                missing.append(f"{field_name} ({', '.join(aliases[:3])})")
        if missing:
            raise SchemaValidationError(
                f"{self.provider} billing schema is missing required columns: "
                + "; ".join(missing)
            )

    def parse(
        self,
        raw: pd.DataFrame,
        *,
        cost_basis: str,
        notes: list[str] | None = None,
    ) -> ParsedBatch:
        resolver = ColumnResolver(raw.columns)
        self.validate_schema(resolver)
        collected_notes = list(notes or [])
        mapped = self.map_frame(raw, resolver, cost_basis=cost_basis, notes=collected_notes)
        if mapped.empty:
            raise EmptyDataError("No billable rows were mapped from the file")
        return ParsedBatch(
            frame=mapped,
            raw_frame=raw,
            provider=self.provider,
            schema_name=self.schema_name,
            schema_version=self.detect_schema_version(resolver),
            notes=collected_notes,
        )

    def detect_schema_version(self, resolver: ColumnResolver) -> str:
        return "1.0"

    def score_from_signatures(self, resolver: ColumnResolver) -> float:
        if not self.signature_columns:
            return 0.0
        matches = sum(1 for column in self.signature_columns if resolver.has(column))
        return matches / len(self.signature_columns)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{type(self).__name__} provider={self.provider}>"


def safe_column(frame: pd.DataFrame, column: str | None) -> pd.Series:
    """Return a column or an all-null series, preserving the frame index."""
    if column is not None and column in frame.columns:
        return frame[column]
    return pd.Series([None] * len(frame), index=frame.index, dtype="object")


def jsonify_credits(value: Any) -> float | None:
    """Sum GCP-style nested credit amounts from a JSON string/list/dict."""
    import json

    if value is None:
        return None
    payload = value
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return None
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, (list, tuple)):
        return None
    total = 0.0
    found = False
    for entry in payload:
        if isinstance(entry, dict):
            amount = entry.get("amount", entry.get("value"))
            if amount is None:
                continue
            try:
                total += float(amount)
                found = True
            except (TypeError, ValueError):
                continue
    return total if found else None
