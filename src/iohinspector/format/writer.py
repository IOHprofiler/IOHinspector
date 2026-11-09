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
    check_format_version,
    copy_manifest,
    utc_now,
    validate_manifest,
    write_manifest,
    writer_entry,
)
from .schema import (
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
    positions: FrameSource | None = None,
    descriptors: FrameSource | None = None,
    *,
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
    get their defaults (``""`` for ``suite`` and ``algorithm_info``, 1 for
    ``n_objectives``, null otherwise).

    ``descriptors`` holds the runs' other descriptors (D17), with the columns of
    ``DESCRIPTORS_SCHEMA``; build it with ``descriptor_frame``. The
    ``algorithm_name`` and ``algorithm_info`` descriptors are made from the runs
    columns, and must agree with them when given.

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
    manifest = copy_manifest(manifest)
    manifest["format"] = FORMAT_NAME
    manifest["format_version"] = FORMAT_VERSION
    manifest["writer"] = writer_entry()
    manifest.setdefault("created", utc_now())
    manifest["counts"] = {"runs": 0, "descriptors_rows": 0, "trace_rows": 0}
    manifest["descriptors"] = {}
    check_format_version(manifest, warn=False)
    problems = validate_manifest(manifest)
    if problems:
        raise ManifestError("invalid manifest:\n  " + "\n  ".join(problems))
    dataset_id = check_dataset_id(manifest["dataset_id"])
    _check_target(path, overwrite)

    runs_df = prepare_runs(runs, dataset_id, {s["path"] for s in manifest["sources"]})
    descriptors_df = prepare_descriptors(descriptors, runs_df)
    run_map = fr.run_map(runs_df)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex[:12]}"
    tmp.mkdir()
    try:
        write_table(runs_df, list(RUNS_COLUMNS), "runs", tmp / RUNS_FILE)
        write_table(
            descriptors_df,
            list(DESCRIPTOR_COLUMNS),
            "descriptors",
            tmp / DESCRIPTORS_FILE,
        )
        counts: dict[str, int] = {
            "runs": runs_df.height,
            "descriptors_rows": descriptors_df.height,
        }
        manifest["descriptors"] = summarize(descriptors_df)
        trace_table = _TableSpec.trace(extra_columns)
        counts["trace_rows"] = _SortedTableWriter(
            trace_table, run_map, tmp, memory_limit
        ).write(trace, tmp / TRACE_FILE)
        if positions is not None:
            max_dim = int(runs_df["dimension"].max() or 0)  # type: ignore[arg-type]
            positions_table = _TableSpec.positions(max_dim)
            counts["positions_rows"] = _SortedTableWriter(
                positions_table, run_map, tmp, memory_limit
            ).write(positions, tmp / POSITIONS_FILE)
        manifest["counts"] = counts
        write_manifest(tmp / MANIFEST_FILE, manifest)
        if validate:
            from .validate import validate_dataset

            validate_dataset(tmp).raise_for_errors()
        _publish(tmp, path, overwrite)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return manifest


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
        raise fr.FormatError(
            f"runs: unknown column(s) {unknown}; other run information goes into descriptors"
        )
    missing = [c for c in RUNS_REQUIRED_INPUT if c not in runs.columns]
    if missing:
        raise fr.FormatError(f"runs: missing required column(s) {missing}")

    columns = []
    for col in RUNS_COLUMNS:
        if col.name in runs.columns:
            columns.append(
                fr.conform(
                    runs[col.name], col.dtype, f"runs column {col.name!r}"
                ).alias(col.name)
            )
        elif col.name == "run_id":
            columns.append(
                run_ids(dataset_id, runs["source_file"], runs["source_run_index"])
            )
        else:
            columns.append(
                pl.repeat(
                    RUNS_DEFAULTS[col.name], runs.height, dtype=col.dtype, eager=True
                ).alias(col.name)
            )
    df = pl.DataFrame(columns)
    problems = fr.runs_problems(df, dataset_id, source_paths)
    if problems:
        raise fr.FormatError(fr.first_problem(problems))
    return fr.sort_runs(df)


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


class _TableSpec:
    """What one sorted table (trace or positions) looks like."""

    def __init__(
        self,
        name: str,
        required: tuple[Column, ...],
        fixed_optional: tuple[Column, ...],
        extras: dict[str, pl.DataType] | None,
        extras_allowed: bool,
        contiguous: bool,
    ):
        self.name = name
        self.required = required  # must be in every chunk (besides the copied columns)
        self.fixed_optional = (
            fixed_optional  # fixed columns a chunk may leave out (null)
        )
        self.extras = extras
        self.extras_allowed = extras_allowed
        self.contiguous = contiguous

    @classmethod
    def trace(cls, extra_columns: Mapping[str, pl.DataType] | None) -> _TableSpec:
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
        required = tuple(
            c for c in POSITIONS_KEY_COLUMNS if c.name not in COPIED_RUN_COLUMNS
        )
        coords = tuple(
            Column(x, pl.Float64(), True) for x in position_columns(max_dimension)
        )
        return cls("positions", required, coords, {}, False, contiguous=False)

    def output_columns(self) -> list[Column]:
        key_cols = TRACE_COLUMNS if self.name == "trace" else POSITIONS_KEY_COLUMNS
        cols = list(key_cols) + list(self.fixed_optional)
        cols += [Column(n, t, True) for n, t in (self.extras or {}).items()]
        return cols


def _check_extra_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise fr.FormatError("trace: extra column names must be non-empty strings")
    if is_reserved_trace_name(name):
        raise fr.FormatError(
            f"trace: extra column {name!r} has a reserved name; rename it and record that in "
            "manifest['renamed_columns']"
        )


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
        self.spill_rows = 0
        self.spill_batch_rows = 0
        self.max_group_rows = MAX_ROW_GROUP_ROWS
        self.spills: list[Path] = []
        self.spill_writer: pa.ipc.RecordBatchFileWriter | None = None
        self.spill_last_key = -1
        self._arrow_schema: pa.Schema | None = None
        self._spill_schema: pa.Schema | None = None

    # -- input -----------------------------------------------------------------

    def write(self, data: FrameSource, file: Path) -> int:
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
        if self.spill_rows == 0:
            row_bytes = max(8.0, df.estimated_size() / df.height)
            self.spill_rows = max(_MIN_SPILL_ROWS, int(self.memory_limit / row_bytes))
            self.spill_batch_rows = max(
                _MIN_BATCH_ROWS, int(SPILL_BATCH_BYTES / row_bytes)
            )
            self.max_group_rows = max(
                _MIN_BATCH_ROWS,
                min(MAX_ROW_GROUP_ROWS, int(ROW_GROUP_BYTES / row_bytes)),
            )
        self.buffer.append(df)
        self.buffered_rows += df.height
        if self.buffered_rows >= self.spill_rows:
            self._spill()

    def _normalise(self, chunk: pl.DataFrame) -> pl.DataFrame | None:
        spec = self.spec
        name = spec.name
        missing = [c.name for c in spec.required if c.name not in chunk.columns]
        if missing:
            raise fr.FormatError(f"{name}: missing required column(s) {missing}")
        known = (
            {c.name for c in spec.required}
            | set(COPIED_RUN_COLUMNS)
            | {c.name for c in spec.fixed_optional}
        )
        others = [c for c in chunk.columns if c not in known]
        if others and not spec.extras_allowed:
            raise fr.FormatError(
                f"{name}: unknown column(s) {others}; positions hold run_id, row and x0 to "
                f"x{len(spec.fixed_optional) - 1} (the largest dimension)"
            )
        if spec.extras is None:
            extras = {}
            for col in others:
                _check_extra_name(col)
                extras[col] = fr.extra_column_dtype(
                    chunk[col].dtype, f"trace column {col!r}"
                )
            spec.extras = extras
        else:
            unknown = [c for c in others if c not in spec.extras]
            if unknown:
                raise fr.FormatError(
                    f"trace: column(s) {unknown} are not in the first chunk; pass every extra column "
                    "and its type in extra_columns"
                )
        if chunk.height == 0:
            return None

        cols = [
            fr.conform(chunk[c.name], c.dtype, f"{name} column {c.name!r}").alias(
                c.name
            )
            for c in spec.required
        ]
        for col in [
            *spec.fixed_optional,
            *(Column(n, t, True) for n, t in spec.extras.items()),
        ]:
            if col.name in chunk.columns:
                cols.append(
                    fr.conform(
                        chunk[col.name], col.dtype, f"{name} column {col.name!r}"
                    ).alias(col.name)
                )
            else:
                cols.append(
                    pl.repeat(None, chunk.height, dtype=col.dtype, eager=True).alias(
                        col.name
                    )
                )
        given = [c for c in COPIED_RUN_COLUMNS if c in chunk.columns]
        for col in given:  # compared with the run's value after the join
            cols.append(
                fr.conform(chunk[col], pl.Int32(), f"{name} column {col!r}").alias(
                    f"__given_{col}"
                )
            )
        df = pl.DataFrame(cols)
        problems = fr.null_problems(df, [c.name for c in spec.required], name)
        if problems:
            raise fr.FormatError(fr.first_problem(problems))

        # The join's row order does not matter: rows are sorted by key later.
        df = df.join(self.run_map, on="run_id", how="left")
        unknown_runs = df.filter(pl.col(fr.RANK).is_null())
        if unknown_runs.height:
            raise fr.FormatError(
                f"{name}: {unknown_runs.height} row(s) refer to run ids that are not in runs, "
                f"for example {unknown_runs['run_id'][0]}"
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
        out = [c.name for c in spec.output_columns()]
        return df.select(
            *out, fr.sort_key(pl.col(fr.RANK), pl.col("row")).alias(fr.KEY), fr.GROUP
        )

    # -- spilling --------------------------------------------------------------

    def _schemas(self) -> tuple[pa.Schema, pa.Schema]:
        if self._arrow_schema is None:
            out = arrow_schema(self.spec.output_columns(), self.spec.name)
            internal = [
                pa.field(fr.KEY, pa.int64(), nullable=False),
                pa.field(fr.GROUP, pa.int64(), nullable=False),
            ]
            self._arrow_schema = out
            self._spill_schema = pa.schema([*out, *internal])
        assert self._spill_schema is not None
        return self._arrow_schema, self._spill_schema

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
        first, last, parts = self._take_buffer(self.spill_batch_rows)
        _, spill_schema = self._schemas()
        if self.spill_writer is None or first <= self.spill_last_key:
            # The buffer does not continue the current sorted run: start a new spill file.
            self._close_spill()
            self.spill_dir.mkdir(exist_ok=True)
            file = self.spill_dir / f"{len(self.spills):05d}.arrow"
            options = pa.ipc.IpcWriteOptions(compression="lz4")
            self.spill_writer = pa.ipc.new_file(
                str(file), spill_schema, options=options
            )
            self.spills.append(file)
        assert self.spill_writer is not None
        for part in parts:
            table = part.to_arrow().cast(spill_schema)
            for batch in table.to_batches():
                self.spill_writer.write_batch(batch)
        self.spill_last_key = last

    def _close_spill(self) -> None:
        if self.spill_writer is not None:
            self.spill_writer.close()
            self.spill_writer = None

    # -- output ----------------------------------------------------------------

    def _sorted_blocks(self) -> Iterator[pl.DataFrame]:
        if not self.spills:
            if self.buffer:
                yield from self._take_buffer(max(self.spill_batch_rows, 2**16))[2]
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
        arrow_out, _ = self._schemas()
        checker = fr.OrderChecker(self.spec.name, self.spec.contiguous, self.describe)
        out = _RowGroupWriter(
            file,
            arrow_out,
            min(MIN_ROW_GROUP_ROWS, self.max_group_rows),
            self.max_group_rows,
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


class _SpillReader:
    """Reads the record batches of one sorted spill file, one at a time."""

    def __init__(self, file: Path):
        self._source = pa.OSFile(
            str(file), "rb"
        )  # not mmap: mapped pages would count as RSS
        self._reader = pa.ipc.open_file(self._source)
        self._count = self._reader.num_record_batches
        self._next = 0

    @property
    def exhausted(self) -> bool:
        """True when no batches are left after the ones already returned."""
        return self._next >= self._count

    def next(self) -> pl.DataFrame | None:
        while self._next < self._count:
            batch = self._reader.get_batch(self._next)
            self._next += 1
            if batch.num_rows:
                df = pl.from_arrow(batch)
                assert isinstance(df, pl.DataFrame)
                return df
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
    buffers = [r.next() for r in readers]
    while True:
        live = [i for i, b in enumerate(buffers) if b is not None]
        if not live:
            return
        bounds = [int(buffers[i][fr.KEY][-1]) for i in live if not readers[i].exhausted]  # type: ignore[index]
        bound = min(bounds) if bounds else None
        parts = []
        for i in live:
            buf = buffers[i]
            assert buf is not None
            n = (
                buf.height
                if bound is None
                else int(buf[fr.KEY].search_sorted(bound, side="right"))
            )
            if n == 0:
                continue
            parts.append(buf.slice(0, n))
            rest = buf.slice(n)
            buffers[i] = rest if rest.height else readers[i].next()
        if len(parts) == 1:
            yield parts[0]
        elif parts:
            yield pl.concat(parts, how="vertical").sort(fr.KEY)


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
