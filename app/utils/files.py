"""Safe filesystem helpers for uploads, archives and parquet storage."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from app.core.exceptions import FileSafetyError

SUPPORTED_SUFFIXES = {".csv", ".parquet", ".gz", ".zip"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file through SHA-256 to build a stable fingerprint."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_filename(name: str) -> str:
    """Return a filesystem-safe filename derived from an upload name."""
    base = Path(str(name)).name
    base = _SAFE_NAME.sub("_", base).strip("._")
    return base or "upload"


def ensure_within_directory(path: Path, root: Path) -> Path:
    """Resolve ``path`` and fail if it escapes ``root`` (path traversal guard)."""
    resolved_root = Path(root).resolve()
    resolved = Path(path).resolve()
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise FileSafetyError("Resolved path escapes the permitted storage directory")
    return resolved


def file_extension(path: Path | str) -> str:
    """Effective extension, treating ``.csv.gz`` as ``csv``."""
    suffixes = [s.lower() for s in Path(path).suffixes]
    if not suffixes:
        return ""
    if suffixes[-1] == ".gz":
        if len(suffixes) >= 2:
            return suffixes[-2].lstrip(".")
        return "gz"
    return suffixes[-1].lstrip(".")


def is_supported_upload(path: Path | str) -> bool:
    extension = file_extension(path)
    return extension in {"csv", "parquet", "zip"}


def human_size(num_bytes: float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            return f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{size:,.1f} PB"


def write_parquet_atomic(frame, destination: Path | str) -> Path:
    """Write a parquet file atomically so a crash cannot leave partial output."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=destination.stem + ".", suffix=".tmp.parquet"
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        frame.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return destination


def safe_extract_zip(
    archive: Path | str,
    destination: Path | str,
    *,
    max_members: int = 16,
    max_uncompressed_bytes: int = 2 * 1024**3,
) -> list[Path]:
    """Extract only supported data files from a ZIP archive.

    Guards against path traversal, absolute paths, symlinks, archive bombs
    (uncompressed size limit) and excessive member counts.
    """
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    try:
        with zipfile.ZipFile(archive) as zf:
            members = [info for info in zf.infolist() if not info.is_dir()]
            if len(members) > max_members:
                raise FileSafetyError(
                    f"Archive contains more than {max_members} files; refusing to inspect it"
                )
            total_uncompressed = 0
            declared_total = 0
            for info in members:
                name = info.filename
                if name.startswith("/") or ".." in Path(name).parts:
                    raise FileSafetyError("Archive contains an unsafe path")
                if info.flag_bits & 0x1:
                    raise FileSafetyError("Encrypted archives are not supported")
                declared_total += info.file_size
                if info.file_size > max_uncompressed_bytes:
                    raise FileSafetyError("Archive member exceeds the uncompressed size limit")
                if not is_supported_upload(name):
                    continue
                target = ensure_within_directory(destination / Path(name).name, destination)
                with zf.open(info) as source, target.open("wb") as sink:
                    copied = 0
                    for chunk in iter(lambda: source.read(1024 * 256), b""):
                        copied += len(chunk)
                        total_uncompressed += len(chunk)
                        if total_uncompressed > max_uncompressed_bytes:
                            raise FileSafetyError(
                                "Archive exceeds the uncompressed size limit"
                            )
                        sink.write(chunk)
                if target.stat().st_size == 0:
                    target.unlink(missing_ok=True)
                    continue
                extracted.append(target)
            if declared_total > max_uncompressed_bytes:
                raise FileSafetyError("Archive declares an unsafe uncompressed size")
    except zipfile.BadZipFile as exc:
        raise FileSafetyError("File is not a valid ZIP archive") from exc

    if not extracted:
        raise FileSafetyError("Archive does not contain a supported CSV or Parquet file")
    return extracted


def cleanup_directory(path: Path | str) -> None:
    shutil.rmtree(path, ignore_errors=True)
