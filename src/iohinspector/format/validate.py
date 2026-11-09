"""``validate_dataset``: check a stored dataset against format v1 in bounded memory."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from . import _frames as fr
from .descriptors import descriptor_problems, summary_problems
from .manifest import UnsupportedFormatVersion, check_format_version, validate_manifest
from .schema import (
    COPIED_RUN_COLUMNS,
    DATASET_FILES,
    DESCRIPTOR_COLUMNS,
    DESCRIPTORS_FILE,
    DESCRIPTORS_SCHEMA,
    EXTRA_COLUMN_DTYPES,
    FORMAT_MINOR,
    KV_FORMAT_VERSION,
    KV_TABLE,
    MANIFEST_FILE,
    POSITIONS_FILE,
    POSITIONS_KEY_COLUMNS,
    RUNS_COLUMNS,
    RUNS_FILE,
    TRACE_COLUMNS,
    TRACE_FILE,
    TRACE_STATISTICS_COLUMNS,
    Column,
    is_reserved_trace_name,
    position_columns,
)

DEFAULT_BATCH_ROWS = 2**18
_BATCH_BYTES = 64 * 2**20


@dataclass
class ValidationReport:
    """The outcome of ``validate_dataset``: errors break the format, warnings do not."""

    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        """Raise ``DatasetValidationError`` if there are errors."""
        if self.errors:
            raise DatasetValidationError(self)

    def __str__(self) -> str:
        lines = [f"{self.path}: {'valid' if self.ok else 'INVALID'}"]
        lines += [f"  error: {e}" for e in self.errors]
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


class DatasetValidationError(fr.FormatError):
    """A dataset failed ``validate_dataset``; ``report`` holds the details."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(str(report))


class _TooManyErrors(Exception):
    pass


class _Checker:
    def __init__(self, path: Path, max_errors: int, batch_rows: int):
        self.path = path
        self.report = ValidationReport(path)
        self.max_errors = max_errors
        self.batch_rows = batch_rows
        self.minor = FORMAT_MINOR
        self.manifest: dict[str, Any] = {}
        self.run_map: pl.DataFrame | None = None
        self.runs: pl.DataFrame | None = None
        self.trace_rows_per_run: np.ndarray | None = None

    def error(self, message: str) -> None:
        self.report.errors.append(message)
        if len(self.report.errors) >= self.max_errors:
            raise _TooManyErrors

    def warn(self, message: str) -> None:
        if message not in self.report.warnings:
            self.report.warnings.append(message)

    def newer_minor(self) -> bool:
        return self.minor > FORMAT_MINOR

    def problem(self, message: str) -> None:
        """An error, or only a warning when a newer minor version may allow it."""
        if self.newer_minor():
            self.warn(message)
        else:
            self.error(message)

    # -- manifest and files ----------------------------------------------------

    def check_manifest(self) -> bool:
        file = self.path / MANIFEST_FILE
        if not file.is_file():
            self.error(f"{MANIFEST_FILE} is missing")
            return False
        try:
            manifest = json.loads(
                file.read_text(encoding="utf-8"), parse_constant=fr._reject_constant
            )
        except (ValueError, UnicodeDecodeError) as exc:
            self.error(f"{MANIFEST_FILE} is not valid JSON: {exc}")
            return False
        if not isinstance(manifest, dict):
            self.error(f"{MANIFEST_FILE} must hold a JSON object")
            return False
        try:
            _, self.minor = check_format_version(manifest, warn=False)
        except UnsupportedFormatVersion as exc:
            self.error(str(exc))
            return False
        if self.newer_minor():
            self.warn(
                f"format version {manifest['format_version']} is newer than this "
                "iohinspector; unknown fields and columns are not checked"
            )
        self.manifest = manifest
        for problem in validate_manifest(manifest):
            self.error(problem)
        return True

    def check_files(self) -> None:
        for entry in sorted(self.path.iterdir()):
            if entry.name not in DATASET_FILES:
                self.warn(f"unexpected file {entry.name!r} in the dataset directory")
        counts = self.manifest.get("counts")
        has_count = isinstance(counts, dict) and "positions_rows" in counts
        has_file = (self.path / POSITIONS_FILE).exists()
        if has_count != has_file:
            self.error(
                "manifest counts.positions_rows must be present exactly when "
                f"{POSITIONS_FILE} exists"
            )

    def count(self, key: str, actual: int) -> None:
        counts = self.manifest.get("counts")
        if not isinstance(counts, dict) or key not in counts:
            return
        if counts[key] != actual:
            self.error(
                f"manifest counts.{key} is {counts[key]}, but the file has {actual} rows"
            )

    # -- Parquet files ---------------------------------------------------------

    def open_table(self, name: str, table: str) -> pq.ParquetFile | None:
        file = self.path / name
        if not file.is_file():
            self.error(f"{name} is missing")
            return None
        try:
            pf = pq.ParquetFile(file)
        except (OSError, ValueError) as exc:  # pyarrow's ArrowInvalid is a ValueError
            self.error(f"{name} is not a readable Parquet file: {exc}")
            return None
        kv = pf.metadata.metadata or {}
        if kv.get(KV_TABLE.encode()) not in (None, table.encode()):
            self.error(
                f"{name}: key-value metadata names table {kv[KV_TABLE.encode()]!r}"
            )
        if KV_FORMAT_VERSION.encode() not in kv or KV_TABLE.encode() not in kv:
            self.warn(
                f"{name}: key-value metadata {KV_FORMAT_VERSION} or {KV_TABLE} is missing"
            )
        return pf

    def schema_problems(
        self, name: str, schema: dict[str, pl.DataType], fixed: list[Column]
    ) -> bool:
        """Check the fixed columns; return True if the table can be read further."""
        readable = True
        for col in fixed:
            if col.name not in schema:
                self.error(f"{name}: column {col.name!r} is missing")
                readable = False
            elif schema[col.name] != col.dtype:
                self.error(
                    f"{name}: column {col.name!r} has type {schema[col.name]}, expected {col.dtype}"
                )
                readable = False
        names = list(schema)
        order = [c.name for c in fixed if c.name in schema]
        if [n for n in names if n in order] != order:
            self.warn(f"{name}: the fixed columns are not in the documented order")
        return readable

    def storage(
        self, name: str, pf: pq.ParquetFile, stats_columns: tuple[str, ...]
    ) -> None:
        meta = pf.metadata
        names = [meta.schema.column(i).name for i in range(meta.num_columns)]
        for i in range(meta.num_row_groups):
            rg = meta.row_group(i)
            for j, col in enumerate(names):
                chunk = rg.column(j)
                if chunk.compression != "ZSTD":
                    self.warn(
                        f"{name}: column {col!r} uses {chunk.compression}, not zstd"
                    )
                if col in stats_columns:
                    stats = chunk.statistics
                    if stats is None or not stats.has_min_max:
                        self.warn(f"{name}: column {col!r} lacks row-group statistics")

    def batches(self, pf: pq.ParquetFile, columns: list[str]):
        rows = max(
            1024, min(self.batch_rows, _BATCH_BYTES // (8 * max(1, len(columns))))
        )
        for batch in pf.iter_batches(batch_size=rows, columns=columns):
            df = pl.from_arrow(batch)
            assert isinstance(df, pl.DataFrame)
            yield df

    # -- runs ------------------------------------------------------------------

    def check_runs(self) -> None:
        pf = self.open_table(RUNS_FILE, "runs")
        if pf is None:
            return
        schema = dict(pl.scan_parquet(self.path / RUNS_FILE).collect_schema())
        if not self.schema_problems("runs", schema, list(RUNS_COLUMNS)):
            return
        known = {c.name for c in RUNS_COLUMNS}
        unknown = [c for c in schema if c not in known]
        if unknown:
            self.problem(f"runs: unknown column(s) {unknown}")
        df = pl.read_parquet(
            self.path / RUNS_FILE, columns=[c.name for c in RUNS_COLUMNS]
        )
        self.count("runs", df.height)
        sources = self.manifest.get("sources")
        paths = (
            {str(s.get("path")) for s in sources if isinstance(s, dict)}
            if isinstance(sources, list)
            else None
        )
        nulls = fr.null_problems(
            df, [c.name for c in RUNS_COLUMNS if not c.nullable], "runs"
        )
        for problem in nulls:
            self.error(problem)
        if nulls:
            return
        for problem in fr.runs_problems(
            df, str(self.manifest.get("dataset_id")), paths
        ):
            self.error(problem)
        ordered = fr.sort_runs(df)
        if not ordered["run_id"].equals(df["run_id"]):
            first = int((ordered["run_id"] != df["run_id"]).arg_true()[0])
            self.error(
                f"runs: rows are not in run order (by {', '.join(fr.RUNS_SORT_KEYS)}); "
                f"the first misplaced row is row {first}"
            )
        if df["run_id"].n_unique() == df.height:
            self.run_map = fr.run_map(ordered)
            self.runs = df

    # -- descriptors -----------------------------------------------------------

    def check_descriptors(self) -> None:
        pf = self.open_table(DESCRIPTORS_FILE, "descriptors")
        if pf is None:
            return
        schema = dict(pl.scan_parquet(self.path / DESCRIPTORS_FILE).collect_schema())
        if not self.schema_problems("descriptors", schema, list(DESCRIPTOR_COLUMNS)):
            return
        unknown = [c for c in schema if c not in DESCRIPTORS_SCHEMA]
        if unknown:
            self.problem(f"descriptors: unknown column(s) {unknown}")
        self.storage("descriptors", pf, ("run_id", "key"))
        df = pl.read_parquet(
            self.path / DESCRIPTORS_FILE, columns=list(DESCRIPTORS_SCHEMA)
        )
        self.count("descriptors_rows", df.height)
        problems = descriptor_problems(df, self.runs)
        for problem in problems:
            self.error(problem)
        if not any("null value" in p for p in problems):
            for problem in summary_problems(self.manifest.get("descriptors"), df):
                self.error(problem)

    # -- trace and positions ---------------------------------------------------

    def check_trace(self) -> None:
        pf = self.open_table(TRACE_FILE, "trace")
        if pf is None:
            return
        schema = dict(pl.scan_parquet(self.path / TRACE_FILE).collect_schema())
        readable = self.schema_problems("trace", schema, list(TRACE_COLUMNS))
        fixed = {c.name for c in TRACE_COLUMNS}
        for name, dtype in schema.items():
            if name in fixed:
                continue
            if is_reserved_trace_name(name):
                self.problem(f"trace: extra column {name!r} has a reserved name")
            if not any(dtype == allowed for allowed in EXTRA_COLUMN_DTYPES):
                self.problem(
                    f"trace: extra column {name!r} has type {dtype}, which is not allowed"
                )
        self.storage("trace", pf, TRACE_STATISTICS_COLUMNS)
        self.count("trace_rows", pf.metadata.num_rows)
        if not readable:
            return
        self.trace_rows_per_run = self.stream(
            "trace", pf, [c.name for c in TRACE_COLUMNS], contiguous=True
        )

    def check_positions(self) -> None:
        if not (self.path / POSITIONS_FILE).exists():
            return
        pf = self.open_table(POSITIONS_FILE, "positions")
        if pf is None:
            return
        schema = dict(pl.scan_parquet(self.path / POSITIONS_FILE).collect_schema())
        max_dim = 0
        if self.run_map is not None and self.run_map.height:
            max_dim = int(self.run_map["dimension"].max())  # type: ignore[arg-type]
        coords = [Column(x, pl.Float64(), True) for x in position_columns(max_dim)]
        fixed = list(POSITIONS_KEY_COLUMNS) + coords
        readable = self.schema_problems("positions", schema, fixed)
        extra = [c for c in schema if c not in {c.name for c in fixed}]
        if extra:
            self.problem(
                f"positions: unexpected column(s) {extra}; coordinates are x0 to "
                f"x{max_dim - 1}, for the largest dimension {max_dim}"
            )
        self.storage("positions", pf, ("run_id", "function_id", "dimension", "row"))
        self.count("positions_rows", pf.metadata.num_rows)
        if readable:
            self.stream("positions", pf, [c.name for c in fixed], contiguous=False)

    def stream(
        self, table: str, pf: pq.ParquetFile, columns: list[str], contiguous: bool
    ) -> np.ndarray | None:
        """Row-by-row checks in batches; returns the number of rows per run rank."""
        if self.run_map is None:
            return None
        run_map = self.run_map
        key_columns = [
            c.name for c in (TRACE_COLUMNS if contiguous else POSITIONS_KEY_COLUMNS)
        ]
        not_null = [
            c.name for c in TRACE_COLUMNS if not c.nullable and c.name in key_columns
        ]
        checker = fr.OrderChecker(table, contiguous, fr.describe_runs(run_map))
        order_ok = True
        per_run = np.zeros(run_map.height, dtype=np.int64)
        coords = [c for c in columns if c.startswith("x")]
        for batch in self.batches(pf, columns):
            for problem in fr.null_problems(batch, not_null, table):
                self.error(problem)
            # Keep the file's row order through the join: the order checks need it.
            df = (
                batch.with_row_index("__ioh_pos")
                .join(run_map, on="run_id", how="left", suffix="_run")
                .sort("__ioh_pos")
            )
            unknown = df.filter(pl.col(fr.RANK).is_null())
            if unknown.height:
                self.error(
                    f"{table}: {unknown.height} row(s) refer to run ids that are not in runs, "
                    f"for example {unknown['run_id'][0]}"
                )
                df = df.filter(pl.col(fr.RANK).is_not_null())
            for col in COPIED_RUN_COLUMNS:
                bad = df.filter(pl.col(col) != pl.col(f"{col}_run"))
                if bad.height:
                    self.error(
                        f"{table}: {bad.height} row(s) have a {col} that differs from their "
                        f"run's, for example run {bad['run_id'][0]}"
                    )
            if order_ok:
                keys = df.select(fr.sort_key(pl.col(fr.RANK), pl.col("row")))
                problem = checker.check(keys.to_series())
                if problem is not None:
                    self.error(problem)
                    order_ok = False
            counts = df.group_by(fr.RANK).len()
            np.add.at(per_run, counts[fr.RANK].to_numpy(), counts["len"].to_numpy())
            if not contiguous:
                self.positions_rows(df, coords)
        return per_run

    def positions_rows(self, df: pl.DataFrame, coords: list[str]) -> None:
        if self.trace_rows_per_run is not None:
            limit = pl.Series(self.trace_rows_per_run).gather(df[fr.RANK])
            missing = df.filter(df["row"] >= limit)
            if missing.height:
                self.error(
                    f"positions: {missing.height} row(s) have no trace row, for example "
                    f"run {missing['run_id'][0]} row {missing['row'][0]}"
                )
        if coords:
            pattern = pl.any_horizontal(
                [
                    pl.col(x).is_null() != (pl.col(fr.RUN_DIM) <= j)
                    for j, x in enumerate(coords)
                ]
            )
            bad = df.filter(pattern)
            if bad.height:
                self.error(
                    f"positions: {bad.height} row(s) have coordinates whose nulls do not match "
                    f"the run's dimension, for example run {bad['run_id'][0]} row {bad['row'][0]}"
                )


def validate_dataset(
    path: str | os.PathLike[str],
    *,
    max_errors: int = 100,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> ValidationReport:
    """Check the dataset directory ``path`` against format v1 (``docs/format.md``).

    Checks the manifest (JSON Schema, version, sources), the table schemas, the runs
    (run ids, uniqueness, sort order), the trace and positions (referential integrity
    with the runs, copied columns, sort order, row numbering, coordinates), and the
    manifest counts. Tables are read in batches of ``batch_rows`` rows, so memory
    stays bounded. Stops after ``max_errors`` errors.
    """
    checker = _Checker(Path(path), max(1, max_errors), max(1, batch_rows))
    report = checker.report
    try:
        if not checker.path.is_dir():
            checker.error(f"{checker.path} is not a directory")
        elif checker.check_manifest():
            checker.check_files()
            checker.check_runs()
            checker.check_descriptors()
            checker.check_trace()
            checker.check_positions()
    except _TooManyErrors:
        report.errors.append(f"stopped after {max_errors} errors")
    return report
