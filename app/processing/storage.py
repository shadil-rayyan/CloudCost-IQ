"""Parquet storage for normalized billing records.

Each import is stored as one immutable parquet file named after its import ID.
Replacing a billing period means superseding the old import metadata and
deleting its parquet file - never editing data in place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.core.config import get_app_config
from app.core.constants import DATETIME_COLUMNS, NORMALIZED_COLUMNS
from app.processing.normalization import empty_normalized_frame
from app.utils.files import write_parquet_atomic


class ProcessedStore:
    """Read/write access to normalized billing parquet files."""

    def __init__(self, processed_dir: Path | str | None = None) -> None:
        if processed_dir is None:
            processed_dir = get_app_config().storage.processed_dir
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, import_id: str) -> Path:
        return self.processed_dir / f"{import_id}.parquet"

    def write(self, frame: pd.DataFrame, import_id: str) -> Path:
        destination = self.path_for(import_id)
        write_parquet_atomic(frame, destination)
        return destination

    def delete(self, import_id: str) -> bool:
        path = self.path_for(import_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def exists(self, import_id: str) -> bool:
        return self.path_for(import_id).exists()

    def list_import_ids(self) -> list[str]:
        return sorted(path.stem for path in self.processed_dir.glob("*.parquet"))

    def parquet_files(self, import_ids: list[str] | None = None) -> list[Path]:
        if import_ids is None:
            return sorted(self.processed_dir.glob("*.parquet"))
        return [self.path_for(import_id) for import_id in import_ids if self.exists(import_id)]

    def read(self, import_ids: list[str] | None = None) -> pd.DataFrame:
        files = self.parquet_files(import_ids)
        if not files:
            return empty_normalized_frame()
        frames = [pd.read_parquet(path) for path in files]
        frame = pd.concat(frames, ignore_index=True)
        for column in DATETIME_COLUMNS:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        missing = [column for column in NORMALIZED_COLUMNS if column not in frame.columns]
        for column in missing:
            frame[column] = None
        return frame

    def stats(self) -> dict[str, float | int]:
        files = self.parquet_files()
        return {
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        }
