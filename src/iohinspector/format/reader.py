"""``scan_dataset``: open a stored dataset as polars LazyFrames."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import polars as pl

from .descriptors import descriptor_columns
from .manifest import DatasetWarning, read_manifest
from .schema import (
    ALGORITHM_DESCRIPTORS,
    DESCRIPTORS_FILE,
    POSITIONS_FILE,
    RUNS_FILE,
    RUNS_SCHEMA,
    TRACE_FILE,
)


@dataclass(frozen=True)
class StoredDataset:
    """A dataset in format v1: its manifest and its tables as LazyFrames.

    ``positions`` is None when the dataset has no positions. Filters on
    ``function_id`` and ``dimension`` of ``trace`` and ``positions`` skip row groups.
    ``run_columns(by)`` gives the columns to group runs by (D17). ``files`` keeps a
    temporary folder alive (``iohinspector.convert.read_dataset``).
    """

    path: Path
    manifest: dict[str, Any]
    runs: pl.LazyFrame
    descriptors: pl.LazyFrame
    trace: pl.LazyFrame
    positions: pl.LazyFrame | None
    files: Any = field(default=None, repr=False, compare=False)

    @property
    def dataset_id(self) -> str:
        return self.manifest["dataset_id"]

    @property
    def format_version(self) -> str:
        return self.manifest["format_version"]

    @property
    def descriptor_keys(self) -> list[str]:
        """The descriptor keys, from the manifest summary."""
        return sorted(self.manifest["descriptors"])

    def run_columns(
        self,
        by: Sequence[str] = ("algorithm_name",),
        *,
        mixed: Literal["string", "error"] = "string",
    ) -> pl.LazyFrame:
        """``run_id`` and one column per name in ``by``, one row per run.

        Each name is a ``runs.parquet`` column (such as ``function_id``) or a
        descriptor key, turned into a column as ``descriptor_columns`` does; runs
        without the key get null. ``algorithm_name`` and ``algorithm_info`` are
        descriptors, so the default reads descriptors.parquet. Join the result to the
        trace on ``run_id`` and group by ``by``. Raises ``ValueError`` for an unknown
        name.
        """
        by = [b for b in dict.fromkeys(by) if b != "run_id"]
        columns, present, absent = _names(by, self.manifest["descriptors"])
        unknown = [b for b in absent if b not in ALGORITHM_DESCRIPTORS]
        if unknown:
            raise ValueError(f"no runs column or descriptor is named {unknown}")
        lf = self.runs.select("run_id", *columns)
        if present:
            values = descriptor_columns(self.descriptors, present, mixed=mixed)
            lf = lf.join(values, on="run_id", how="left")
        # a required descriptor of a dataset without runs: null
        lf = lf.with_columns(pl.lit(None, pl.String).alias(b) for b in absent)
        return lf.select("run_id", *by)


def _names(by: list[str], keys) -> tuple[list[str], list[str], list[str]]:
    """The names of ``by`` that are runs columns, descriptor ``keys``, and neither."""
    named = [b for b in by if b not in RUNS_SCHEMA]
    return (
        [b for b in by if b in RUNS_SCHEMA],
        [b for b in named if b in keys],
        [b for b in named if b not in keys],
    )


def scan_dataset(path: str | os.PathLike[str], *, warn: bool = True) -> StoredDataset:
    """Open the dataset directory ``path`` without reading its tables.

    Raises ``UnsupportedFormatVersion`` for an unknown major version and
    ``ManifestError`` for an invalid manifest. With ``warn``, each warning recorded
    in the manifest (for example about the precision of the source) is issued as a
    ``DatasetWarning``. It does not check the tables; ``validate_dataset`` does.
    """
    path = Path(path)
    manifest = read_manifest(path, warn=warn)
    for name in (RUNS_FILE, DESCRIPTORS_FILE, TRACE_FILE):
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
        descriptors=pl.scan_parquet(path / DESCRIPTORS_FILE),
        trace=pl.scan_parquet(path / TRACE_FILE),
        positions=pl.scan_parquet(path / POSITIONS_FILE) if has_positions else None,
    )
