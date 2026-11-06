"""``scan_dataset``: open a stored dataset as polars LazyFrames."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .manifest import DatasetWarning, read_manifest
from .schema import POSITIONS_FILE, RUNS_FILE, TRACE_FILE


@dataclass(frozen=True)
class StoredDataset:
    """A dataset in format v1: its manifest and its tables as LazyFrames.

    ``positions`` is None when the dataset has no positions. Filters on
    ``function_id`` and ``dimension`` of ``trace`` and ``positions`` skip row groups.
    """

    path: Path
    manifest: dict[str, Any]
    runs: pl.LazyFrame
    trace: pl.LazyFrame
    positions: pl.LazyFrame | None

    @property
    def dataset_id(self) -> str:
        return self.manifest["dataset_id"]

    @property
    def format_version(self) -> str:
        return self.manifest["format_version"]


def scan_dataset(path: str | os.PathLike[str], *, warn: bool = True) -> StoredDataset:
    """Open the dataset directory ``path`` without reading its tables.

    Raises ``UnsupportedFormatVersion`` for an unknown major version and
    ``ManifestError`` for an invalid manifest. With ``warn``, each warning recorded
    in the manifest (for example about the precision of the source) is issued as a
    ``DatasetWarning``. It does not check the tables; ``validate_dataset`` does.
    """
    path = Path(path)
    manifest = read_manifest(path, warn=warn)
    for name in (RUNS_FILE, TRACE_FILE):
        if not (path / name).is_file():
            raise FileNotFoundError(f"{path / name} is missing")
    has_positions = "positions_rows" in manifest["counts"]
    if has_positions and not (path / POSITIONS_FILE).is_file():
        raise FileNotFoundError(
            f"{path / POSITIONS_FILE} is missing, but the manifest counts positions"
        )
    if warn:
        for w in manifest.get("warnings", []):
            where = f" ({w['source']})" if "source" in w else ""
            warnings.warn(
                f"{manifest['dataset_id']}: {w['code']}{where}: {w['message']}",
                DatasetWarning,
                stacklevel=2,
            )
    return StoredDataset(
        path=path,
        manifest=manifest,
        runs=pl.scan_parquet(path / RUNS_FILE),
        trace=pl.scan_parquet(path / TRACE_FILE),
        positions=pl.scan_parquet(path / POSITIONS_FILE) if has_positions else None,
    )
