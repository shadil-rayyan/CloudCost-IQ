from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import pytest

from app.core.config import UploadConfig
from app.core.exceptions import FileSafetyError
from app.ingestion.base_parser import read_tabular_file


def test_rejects_path_traversal_zip(tmp_path: Path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../evil.csv", "a,b\n1,2\n")
    with pytest.raises((FileSafetyError, Exception)):
        read_tabular_file(archive, upload_config=UploadConfig())


def test_rejects_decompression_bomb_gzip(tmp_path: Path):
    archive = tmp_path / "bomb.csv.gz"
    with gzip.open(archive, "wb") as handle:
        handle.write(b"x" * (1024 * 32))
    with pytest.raises((FileSafetyError, Exception)):
        read_tabular_file(archive, upload_config=UploadConfig(max_uncompressed_size_mb=0))
