"""``write_dataset``: write a dataset in format v1 in bounded memory.

Trace and positions rows may arrive in any order. They are sorted with an external
merge sort: rows are buffered up to ``memory_limit`` bytes, each full buffer is sorted
and spilled to a temporary Arrow IPC file, and the spill files are then merged while
the Parquet file is written row group by row group.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from . import _frames as fr
from ._io import PARQUET_OPTIONS, write_table
from .descriptors import prepare_descriptors, summarize
from .ids import check_dataset_id, run_ids
from .manifest import (
    ManifestError,
    copy_manifest,
    utc_now,
    validate_manifest,
    write_manifest,
    writer_entry,
)
from .schema import (
    ALGORITHM_DESCRIPTORS,
    COPIED_RUN_COLUMNS,
    DESCRIPTOR_COLUMNS,
    DESCRIPTORS_FILE,
    FORMAT_NAME,
    FORMAT_VERSION,
    MANIFEST_FILE,
    POSITIONS_FILE,
    POSITIONS_KEY_COLUMNS,
    RUNS_COLUMNS,
    RUNS_DEFAULTS,
    RUNS_FILE,
    RUNS_REQUIRED_INPUT,
    RUNS_SCHEMA,
    TRACE_COLUMNS,
    TRACE_FILE,
    Column,
    arrow_schema,
    is_reserved_trace_name,
    position_columns,
)
from .validate import validate_dataset

FrameSource = pl.DataFrame | pl.LazyFrame | Iterable[pl.DataFrame | pl.LazyFrame]

DEFAULT_MEMORY_LIMIT = 256 * 2**20
MIN_ROW_GROUP_ROWS = 2**16
MAX_ROW_GROUP_ROWS = 2**20
ROW_GROUP_BYTES = 128 * 2**20
SPILL_BATCH_BYTES = 2**20
_INPUT_SLICE_ROWS = 2**18
# Lower bounds of the derived sizes; tests lower them to force many spills and row groups.
_MIN_MEMORY_LIMIT = 2**20
_MIN_SPILL_ROWS = 2**12
_MIN_BATCH_ROWS = 2**10
_PARQUET_OPTIONS = PARQUET_OPTIONS


def write_dataset(
    path: str | os.PathLike[str],
    manifest: Mapping[str, Any],
    runs: pl.DataFrame | pl.LazyFrame,
    trace: FrameSource,
    *,
    positions: FrameSource | None = None,
    descriptors: FrameSource | None = None,
    overwrite: bool = False,
    extra_columns: Mapping[str, pl.DataType] | None = None,
    memory_limit: int = DEFAULT_MEMORY_LIMIT,
    validate: bool = True,
) -> dict[str, Any]:
    """Write a dataset in format v1 to the directory ``path`` and return its manifest.

    ``manifest`` is usually built with ``new_manifest``; the writer sets ``format``,
    ``format_version``, ``writer`` and ``counts``, and never changes the caller's dict.

    ``runs`` has one row per run with the columns of ``RUNS_SCHEMA``. ``run_id`` is
    computed when absent and checked when present; optional columns that are absent
    get their defaults (``""`` for ``suite``, 1 for ``n_objectives``, null
    otherwise).

    ``descriptors`` holds the runs' descriptors (D17), with the columns of
    ``DESCRIPTORS_SCHEMA``; build it with ``descriptor_frame``. Every run needs an
    ``algorithm_name`` descriptor of type string, usually with origin
    ``algorithm``. A run without an ``algorithm_info`` descriptor gets one with
    the value ``""`` and origin ``algorithm``. A dataset read with
    ``scan_dataset`` can therefore be written back as it is.

    ``trace`` holds ``run_id``, ``row``, ``evaluations``, ``raw_y`` and any extra
    logged columns; ``function_id`` and ``dimension`` are copied from the runs.
    ``positions`` holds ``run_id``, ``row`` and ``x0`` ... Both may be a DataFrame, a
    LazyFrame or an iterable of DataFrame chunks, in any order. Extra column types are
    taken from the first chunk unless ``extra_columns`` gives them; a later chunk
    without some extra column gets nulls there.

    Memory use stays near ``2 * memory_limit`` plus one input chunk. The dataset is
    written to a temporary directory next to ``path``, checked with
    ``validate_dataset`` (unless ``validate`` is false), and then moved into place.
    ``overwrite`` replaces an existing dataset directory.

    Raises ``FormatError`` when the data breaks the format, and ``ManifestError``
    when the manifest is invalid.
    """
    path = Path(path)
    manifest = _writable_manifest(manifest)
    _check_target(path, overwrite)
    runs_df = prepare_runs(
        runs, manifest["dataset_id"], {s["path"] for s in manifest["sources"]}
    )
    descriptors_df = prepare_descriptors(descriptors, runs_df)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex[:12]}"
    tmp.mkdir()
    try:
        counts = _write_small_tables(tmp, runs_df, descriptors_df)
        manifest["descriptors"] = summarize(descriptors_df)
        run_map = fr.run_map(runs_df)
        sorted_tables = [(_TableSpec.trace(extra_columns), trace, TRACE_FILE)]
        if positions is not None:
            max_dim = int(runs_df["dimension"].max() or 0)  # type: ignore[arg-type]
            sorted_tables.append(
                (_TableSpec.positions(max_dim), positions, POSITIONS_FILE)
            )
        for spec, data, name in sorted_tables:
            writer = _SortedTableWriter(spec, run_map, tmp, memory_limit)
            counts[f"{spec.name}_rows"] = writer.write(data, tmp / name)
        manifest["counts"] = counts
        write_manifest(tmp / MANIFEST_FILE, manifest)
        if validate:
            validate_dataset(tmp).raise_for_errors()
        _publish(tmp, path, overwrite)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return manifest


def _writable_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of the caller's manifest, with the fields the writer owns reset.

    Raises ``ManifestError`` if the rest is invalid.
    """
    out = copy_manifest(manifest)
    out["format"] = FORMAT_NAME
    out["format_version"] = FORMAT_VERSION
    out["writer"] = writer_entry()
    out.setdefault("created", utc_now())
    out["counts"] = {"runs": 0, "descriptors_rows": 0, "trace_rows": 0}
    out["descriptors"] = {}
    problems = validate_manifest(out)
    if problems:
        raise ManifestError("invalid manifest:\n  " + "\n  ".join(problems))
    check_dataset_id(out["dataset_id"])
    return out


def _write_small_tables(
    tmp: Path, runs: pl.DataFrame, descriptors: pl.DataFrame
) -> dict[str, int]:
    """Write runs.parquet and descriptors.parquet; return their row counts."""
    write_table(runs, list(RUNS_COLUMNS), "runs", tmp / RUNS_FILE)
    write_table(
        descriptors, list(DESCRIPTOR_COLUMNS), "descriptors", tmp / DESCRIPTORS_FILE
    )
    return {"runs": runs.height, "descriptors_rows": descriptors.height}


def prepare_runs(
    runs: pl.DataFrame | pl.LazyFrame, dataset_id: str, source_paths: set[str] | None
) -> pl.DataFrame:
    """The runs table with format columns, types and defaults, in run order.

    Raises ``FormatError`` if a column is unknown, missing or of the wrong type, or if
    ``runs_problems`` finds a problem.
    """
    if isinstance(runs, pl.LazyFrame):
        runs = runs.collect()
    if not isinstance(runs, pl.DataFrame):
        raise TypeError("runs must be a polars DataFrame or LazyFrame")
    unknown = [c for c in runs.columns if c not in RUNS_SCHEMA]
    if unknown:
        hint = "other run information goes into descriptors"
        if any(c in ALGORITHM_DESCRIPTORS for c in unknown):
            hint = "algorithm_name and algorithm_info are descriptors, not runs columns"
        raise fr.FormatError(f"runs: unknown column(s) {unknown}; {hint}")
    missing = [c for c in RUNS_REQUIRED_INPUT if c not in runs.columns]
    if missing:
        raise fr.FormatError(f"runs: missing required column(s) {missing}")

    df = pl.DataFrame([_runs_column(runs, col, dataset_id) for col in RUNS_COLUMNS])
    problems = fr.runs_problems(df, dataset_id, source_paths)
    if problems:
        raise fr.FormatError(fr.first_problem(problems))
    return fr.sort_runs(df)


def _runs_column(runs: pl.DataFrame, col: Column, dataset_id: str) -> pl.Series:
    """One runs column: conformed, computed (``run_id``) or its default."""
    if col.name in runs.columns:
        return fr.conform(runs[col.name], col.dtype, f"runs column {col.name!r}").alias(
            col.name
        )
    if col.name == "run_id":
        return run_ids(dataset_id, runs["source_file"], runs["source_run_index"])
    return pl.repeat(
        RUNS_DEFAULTS[col.name], runs.height, dtype=col.dtype, eager=True
    ).alias(col.name)


def _check_target(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    if not path.is_dir() or (
        any(path.iterdir()) and not (path / MANIFEST_FILE).is_file()
    ):
        raise FileExistsError(
            f"{path} exists and is not a dataset directory; refusing to replace it"
        )


def _publish(tmp: Path, path: Path, overwrite: bool) -> None:
    if path.exists():
        _check_target(path, overwrite)
        old = path.parent / f".{path.name}.old-{uuid.uuid4().hex[:12]}"
        os.rename(path, old)
        os.rename(tmp, path)
        shutil.rmtree(old, ignore_errors=True)
    else:
        os.rename(tmp, path)


@dataclass
class _TableSpec:
    """What one sorted table (trace or positions) looks like."""

    name: str
    #: Columns every chunk must have (besides the copied run columns).
    required: tuple[Column, ...]
    #: Fixed columns a chunk may leave out; they are null then.
    fixed_optional: tuple[Column, ...]
    #: Extra column types; None until the first chunk (or ``extra_columns``) sets them.
    extras: dict[str, pl.DataType] | None
    extras_allowed: bool
    contiguous: bool

    @classmethod
    def trace(cls, extra_columns: Mapping[str, pl.DataType] | None) -> _TableSpec:
        """The trace: required keys and values, and extra logged columns."""
        extras = None
        if extra_columns is not None:
            extras = {}
            for name, dtype in extra_columns.items():
                _check_extra_name(name)
                extras[name] = fr.extra_column_dtype(dtype, f"extra column {name!r}")
        required = tuple(c for c in TRACE_COLUMNS if c.name not in COPIED_RUN_COLUMNS)
        return cls("trace", required, (), extras, True, contiguous=True)

    @classmethod
    def positions(cls, max_dimension: int) -> _TableSpec:
        """Positions: required keys and the coordinates ``x0`` to ``x(D-1)``."""
        required = tuple(
            c for c in POSITIONS_KEY_COLUMNS if c.name not in COPIED_RUN_COLUMNS
        )
        coords = tuple(
            Column(x, pl.Float64(), True) for x in position_columns(max_dimension)
        )
        return cls("positions", required, coords, {}, False, contiguous=False)

    def value_columns(self) -> list[Column]:
        """The fixed optional columns and the extra columns, in output order."""
        extras = [Column(n, t, True) for n, t in (self.extras or {}).items()]
        return [*self.fixed_optional, *extras]

    def output_columns(self) -> list[Column]:
        """All columns of the written table, in order."""
        key_cols = TRACE_COLUMNS if self.name == "trace" else POSITIONS_KEY_COLUMNS
        return [*key_cols, *self.value_columns()]


def _check_extra_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise fr.FormatError("trace: extra column names must be non-empty strings")
    if is_reserved_trace_name(name):
        raise fr.FormatError(
            f"trace: extra column {name!r} has a reserved name; rename it and record "
            "that in manifest['renamed_columns']"
        )


@dataclass(frozen=True)
class _Schemas:
    output: pa.Schema  # the Parquet file
    spill: pa.Schema  # the spill files: output plus the sort key and group


class _SortedTableWriter:
    """Sorts one table's rows by (run order, row) and writes it as Parquet."""

    def __init__(
        self, spec: _TableSpec, run_map: pl.DataFrame, tmp_dir: Path, memory_limit: int
    ):
        self.spec = spec
        self.run_map = run_map
        self.spill_dir = tmp_dir / f"_spill_{spec.name}"
        self.memory_limit = max(int(memory_limit), _MIN_MEMORY_LIMIT)
        self.describe = fr.describe_runs(run_map)
        self.buffer: list[pl.DataFrame] = []
        self.buffered_rows = 0
        self.sizes = _Sizes()
        self.spills: list[Path] = []
        self.spill_writer: pa.ipc.RecordBatchFileWriter | None = None
        self.spill_last_key = -1
        self._schemas: _Schemas | None = None

    # -- input -----------------------------------------------------------------

    def write(self, data: FrameSource, file: Path) -> int:
        """Sort and write every row of ``data``; return the number of rows."""
        try:
            for chunk in fr.iter_frames(data, _INPUT_SLICE_ROWS, self.spec.name):
                self._add(chunk)
            if self.spec.extras is None:  # no rows at all: no extra columns
                self.spec.extras = {}
            return self._write_sorted(file)
        finally:
            self._close_spill()
            shutil.rmtree(self.spill_dir, ignore_errors=True)

    def _add(self, chunk: pl.DataFrame) -> None:
        df = self._normalise(chunk)
        if df is None:
            return
        if not self.sizes.known:
            self.sizes = _Sizes.for_rows(
                df.estimated_size() / df.height, self.memory_limit
            )
        self.buffer.append(df)
        self.buffered_rows += df.height
        if self.buffered_rows >= self.sizes.spill_rows:
            self._spill()

    def _normalise(self, chunk: pl.DataFrame) -> pl.DataFrame | None:
        """The chunk with format types, run order key and group; None if empty."""
        self._check_chunk_columns(chunk)
        if chunk.height == 0:
            return None
        given = [c for c in COPIED_RUN_COLUMNS if c in chunk.columns]
        df = self._conformed(chunk, given)
        df = self._with_runs(df, given)
        out = [c.name for c in self.spec.output_columns()]
        return df.select(
            *out, fr.sort_key(pl.col(fr.RANK), pl.col("row")).alias(fr.KEY), fr.GROUP
        )

    def _check_chunk_columns(self, chunk: pl.DataFrame) -> None:
        """Required columns are present; extra columns are known or set by this chunk."""
        spec = self.spec
        missing = [c.name for c in spec.required if c.name not in chunk.columns]
        if missing:
            raise fr.FormatError(f"{spec.name}: missing required column(s) {missing}")
        known = (
            {c.name for c in spec.required}
            | set(COPIED_RUN_COLUMNS)
            | {c.name for c in spec.fixed_optional}
        )
        others = [c for c in chunk.columns if c not in known]
        if others and not spec.extras_allowed:
            raise fr.FormatError(
                f"{spec.name}: unknown column(s) {others}; positions hold run_id, row "
                f"and x0 to x{len(spec.fixed_optional) - 1} (the largest dimension)"
            )
        if spec.extras is None:
            spec.extras = _infer_extras(chunk, others)
            return
        unknown = [c for c in others if c not in spec.extras]
        if unknown:
            raise fr.FormatError(
                f"trace: column(s) {unknown} are not in the first chunk; pass every "
                "extra column and its type in extra_columns"
            )

    def _conformed(self, chunk: pl.DataFrame, given: list[str]) -> pl.DataFrame:
        """Chunk columns cast to the format types, missing optional ones as null.

        Copied run columns the caller gave are kept as ``__given_<name>``, to be
        compared with the run's value.
        """
        name = self.spec.name
        cols = [
            fr.conform(chunk[c.name], c.dtype, f"{name} column {c.name!r}").alias(
                c.name
            )
            for c in self.spec.required
        ]
        for col in self.spec.value_columns():
            if col.name in chunk.columns:
                series = fr.conform(
                    chunk[col.name], col.dtype, f"{name} column {col.name!r}"
                )
            else:
                series = pl.repeat(None, chunk.height, dtype=col.dtype, eager=True)
            cols.append(series.alias(col.name))
        for col in given:
            series = fr.conform(chunk[col], pl.Int32(), f"{name} column {col!r}")
            cols.append(series.alias(f"__given_{col}"))
        df = pl.DataFrame(cols)
        problems = fr.null_problems(df, [c.name for c in self.spec.required], name)
        if problems:
            raise fr.FormatError(fr.first_problem(problems))
        return df

    def _with_runs(self, df: pl.DataFrame, given: list[str]) -> pl.DataFrame:
        """Join the run's rank, group and copied columns, and check them.

        The join's row order does not matter: rows are sorted by key later.
        """
        name = self.spec.name
        df = df.join(self.run_map, on="run_id", how="left")
        unknown = df.filter(pl.col(fr.RANK).is_null())
        if unknown.height:
            raise fr.FormatError(
                f"{name}: {unknown.height} row(s) refer to run ids that are not in runs, "
                f"for example {unknown['run_id'][0]}"
            )
        for col in given:
            if df.filter(pl.col(f"__given_{col}") != pl.col(col)).height:
                raise fr.FormatError(
                    f"{name}: column {col!r} differs from the run's {col}"
                )
        if df.filter((pl.col("row") < 0) | (pl.col("row") >= fr.ROW_LIMIT)).height:
            raise fr.FormatError(
                f"{name}: row must be between 0 and {fr.ROW_LIMIT - 1}"
            )
        return df

    # -- spilling --------------------------------------------------------------

    def _get_schemas(self) -> _Schemas:
        if self._schemas is None:
            out = arrow_schema(self.spec.output_columns(), self.spec.name)
            internal = [
                pa.field(fr.KEY, pa.int64(), nullable=False),
                pa.field(fr.GROUP, pa.int64(), nullable=False),
            ]
            self._schemas = _Schemas(out, pa.schema([*out, *internal]))
        return self._schemas

    def _take_buffer(self, slice_rows: int) -> tuple[int, int, Iterator[pl.DataFrame]]:
        """Empty the buffer: its first and last key, and its rows in key order.

        The rows come in slices gathered through an arg-sort, so the buffer is never
        copied as a whole while it is sorted.
        """
        df = pl.concat(self.buffer, how="vertical", rechunk=False)
        self.buffer = []
        self.buffered_rows = 0
        keys = df[fr.KEY]
        if keys.is_sorted():
            return int(keys[0]), int(keys[-1]), df.iter_slices(slice_rows)
        order = keys.arg_sort()
        first, last = int(keys[int(order[0])]), int(keys[int(order[-1])])

        def slices() -> Iterator[pl.DataFrame]:
            for start in range(0, order.len(), slice_rows):
                yield df.select(pl.all().gather(order.slice(start, slice_rows)))

        return first, last, slices()

    def _spill(self) -> None:
        first, last, parts = self._take_buffer(self.sizes.spill_batch_rows)
        spill_schema = self._get_schemas().spill
        writer = self.spill_writer
        if writer is None or first <= self.spill_last_key:
            # The buffer does not continue the current sorted run: start a new file.
            writer = self._new_spill_file(spill_schema)
        for part in parts:
            for batch in part.to_arrow().cast(spill_schema).to_batches():
                writer.write_batch(batch)
        self.spill_last_key = last

    def _new_spill_file(self, schema: pa.Schema) -> pa.ipc.RecordBatchFileWriter:
        self._close_spill()
        self.spill_dir.mkdir(exist_ok=True)
        file = self.spill_dir / f"{len(self.spills):05d}.arrow"
        options = pa.ipc.IpcWriteOptions(compression="lz4")
        self.spill_writer = pa.ipc.new_file(str(file), schema, options=options)
        self.spills.append(file)
        return self.spill_writer

    def _close_spill(self) -> None:
        if self.spill_writer is not None:
            self.spill_writer.close()
            self.spill_writer = None

    # -- output ----------------------------------------------------------------

    def _sorted_blocks(self) -> Iterator[pl.DataFrame]:
        if not self.spills:
            if self.buffer:
                yield from self._take_buffer(max(self.sizes.spill_batch_rows, 2**16))[2]
            return
        if self.buffer:
            self._spill()
        self._close_spill()
        readers = [_SpillReader(p) for p in self.spills]
        try:
            yield from _merge(readers)
        finally:
            for r in readers:
                r.close()

    def _write_sorted(self, file: Path) -> int:
        checker = fr.OrderChecker(self.spec.name, self.spec.contiguous, self.describe)
        max_rows = self.sizes.max_group_rows
        out = _RowGroupWriter(
            file,
            self._get_schemas().output,
            min(MIN_ROW_GROUP_ROWS, max_rows),
            max_rows,
        )
        try:
            for block in self._sorted_blocks():
                problem = checker.check(block[fr.KEY])
                if problem is not None:
                    raise fr.FormatError(problem)
                out.add(block.drop(fr.KEY))
            return out.finish()
        finally:
            out.close()


def _infer_extras(chunk: pl.DataFrame, names: list[str]) -> dict[str, pl.DataType]:
    """Extra column types from the first chunk."""
    extras = {}
    for col in names:
        _check_extra_name(col)
        extras[col] = fr.extra_column_dtype(chunk[col].dtype, f"trace column {col!r}")
    return extras


@dataclass(frozen=True)
class _Sizes:
    """Row counts derived from the estimated bytes per row of the first chunk."""

    spill_rows: int = 0
    spill_batch_rows: int = 0
    max_group_rows: int = MAX_ROW_GROUP_ROWS

    @property
    def known(self) -> bool:
        return self.spill_rows > 0

    @classmethod
    def for_rows(cls, row_bytes: float, memory_limit: int) -> _Sizes:
        row_bytes = max(8.0, row_bytes)
        return cls(
            spill_rows=max(_MIN_SPILL_ROWS, int(memory_limit / row_bytes)),
            spill_batch_rows=max(_MIN_BATCH_ROWS, int(SPILL_BATCH_BYTES / row_bytes)),
            max_group_rows=max(
                _MIN_BATCH_ROWS,
                min(MAX_ROW_GROUP_ROWS, int(ROW_GROUP_BYTES / row_bytes)),
            ),
        )


class _SpillReader:
    """Reads the record batches of one sorted spill file, one at a time."""

    def __init__(self, file: Path):
        # Not a memory map: mapped pages would count towards the resident memory.
        self._source = pa.OSFile(str(file), "rb")
        self._reader = pa.ipc.open_file(self._source)
        self._count = self._reader.num_record_batches
        self._next = 0

    @property
    def exhausted(self) -> bool:
        """True when no batches are left after the ones already returned."""
        return self._next >= self._count

    def next(self) -> pl.DataFrame | None:
        """The next non-empty batch, or None at the end of the file."""
        while self._next < self._count:
            batch = self._reader.get_batch(self._next)
            self._next += 1
            if batch.num_rows:
                return fr.frame_from_arrow(batch)
        return None

    def close(self) -> None:
        self._source.close()


def _merge(readers: list[_SpillReader]) -> Iterator[pl.DataFrame]:
    """Merge sorted spill files into sorted blocks.

    Each round emits every buffered row whose key is at most the smallest "last
    buffered key" among the files that still have batches, so no later row can sort
    before an emitted one. The file that sets that bound is drained, so every round
    makes progress.
    """
    live = [(r, b) for r in readers if (b := r.next()) is not None]
    while live:
        bounds = [int(buf[fr.KEY][-1]) for r, buf in live if not r.exhausted]
        bound = min(bounds) if bounds else None
        parts, still = [], []
        for reader, buf in live:
            head, rest = _split_at(buf, bound)
            if head.height:
                parts.append(head)
            nxt = rest if rest.height else reader.next()
            if nxt is not None:
                still.append((reader, nxt))
        live = still
        if len(parts) == 1:
            yield parts[0]
        elif parts:
            yield pl.concat(parts, how="vertical").sort(fr.KEY)


def _split_at(
    buf: pl.DataFrame, bound: int | None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The rows of sorted ``buf`` with key at most ``bound`` (all if None), and the rest."""
    n = (
        buf.height
        if bound is None
        else int(buf[fr.KEY].search_sorted(bound, side="right"))
    )
    return buf.slice(0, n), buf.slice(n)


class _RowGroupWriter:
    """Writes sorted rows to Parquet, cutting row groups at row-group key changes.

    A row group ends at the first change of (suite, function_id, dimension) after it
    holds ``min_rows`` rows, and never holds more than ``max_rows`` rows.
    """

    def __init__(self, file: Path, schema: pa.Schema, min_rows: int, max_rows: int):
        self.schema = schema
        self.min_rows = min(min_rows, max_rows)
        self.max_rows = max_rows
        self.writer = pq.ParquetWriter(str(file), schema, **_PARQUET_OPTIONS)
        self.parts: list[pl.DataFrame] = []
        self.rows = 0
        self.bounds: list[int] = []  # buffered row positions where the group changes
        self.last_group: int | None = None
        self.written = 0

    def add(self, df: pl.DataFrame) -> None:
        groups = df[fr.GROUP]
        if self.last_group is not None and int(groups[0]) != self.last_group:
            self.bounds.append(self.rows)
        changes = groups.diff().ne(0).arg_true()
        self.bounds.extend((changes + self.rows).to_list())
        self.last_group = int(groups[-1])
        self.parts.append(df.drop(fr.GROUP))
        self.rows += df.height
        self._flush(final=False)

    def _flush(self, final: bool) -> None:
        while self.rows:
            # The first group change that leaves at least min_rows in the row group.
            cut = next((b for b in self.bounds if b >= self.min_rows), None)
            if cut is None or cut > self.max_rows:
                if self.rows >= self.max_rows:
                    cut = self.max_rows
                elif final:
                    cut = self.rows
                else:
                    return
            self._write(cut)

    def _write(self, cut: int) -> None:
        buf = (
            pl.concat(self.parts, how="vertical")
            if len(self.parts) > 1
            else self.parts[0]
        )
        head, tail = buf.slice(0, cut), buf.slice(cut)
        self.parts = [tail] if tail.height else []
        self.bounds = [b - cut for b in self.bounds if b > cut]
        self.rows -= cut
        table = head.to_arrow().cast(self.schema)
        self.writer.write_table(table, row_group_size=max(cut, 1))
        self.written += cut

    def finish(self) -> int:
        self._flush(final=True)
        return self.written

    def close(self) -> None:
        self.writer.close()
