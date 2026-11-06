"""Helpers shared by the writer and the validator. Not public API."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

import polars as pl

from .ids import run_ids, source_path_problem
from .schema import (
    EXTRA_COLUMN_DTYPES,
    ROW_GROUP_KEYS,
    RUNS_COLUMNS,
    RUNS_SORT_KEYS,
)

# Internal columns. Their names cannot clash with format columns.
KEY = "__ioh_key"
RANK = "__ioh_rank"
GROUP = "__ioh_group"
RUN_DIM = "__ioh_run_dim"

ROW_BITS = 32
ROW_LIMIT = 1 << ROW_BITS
_SHA256 = re.compile(r"[0-9a-f]{64}")


class FormatError(ValueError):
    """Data given to the writer, or found in a dataset, does not follow format v1."""


def conform(series: pl.Series, target: pl.DataType, where: str) -> pl.Series:
    """``series`` as ``target`` if that changes no value, else raise ``FormatError``.

    Allowed: integer to a wider or narrower integer when every value fits; Float32
    to Float64; an integer to Float64 when every value is exactly representable;
    Categorical or Enum to String; an all-null column to anything.
    """
    dtype = series.dtype
    if dtype == target:
        return series
    if dtype == pl.Null:
        return series.cast(target)
    if target.is_integer() and dtype.is_integer():
        try:
            return series.cast(target, strict=True)
        except pl.exceptions.PolarsError as exc:
            raise FormatError(
                f"{where}: values of type {dtype} do not fit in {target}"
            ) from exc
    if target == pl.Float64 and dtype == pl.Float32:
        return series.cast(pl.Float64)
    if target == pl.Float64 and dtype.is_integer():
        out = series.cast(pl.Float64)
        back = out.cast(dtype, strict=False)
        if not (back == series).fill_null(True).all():
            raise FormatError(
                f"{where}: integer values cannot be stored exactly as Float64"
            )
        return out
    if target == pl.String and isinstance(dtype, (pl.Categorical, pl.Enum)):
        return series.cast(pl.String)
    raise FormatError(f"{where}: type {dtype} cannot be stored as {target}")


def extra_column_dtype(dtype: pl.DataType, where: str) -> pl.DataType:
    """The stored type of an extra trace column that arrives as ``dtype``."""
    for allowed in EXTRA_COLUMN_DTYPES:
        if dtype == allowed:
            return allowed
    if dtype.is_integer():
        return pl.Int64()
    if dtype == pl.Float32:
        return pl.Float64()
    if isinstance(dtype, (pl.Categorical, pl.Enum)):
        return pl.String()
    if dtype == pl.Null:
        raise FormatError(
            f"{where}: the column has no type (all null); pass extra_columns to write_dataset"
        )
    raise FormatError(
        f"{where}: type {dtype} is not allowed; use Float64, Int64, Boolean or String"
    )


def null_problems(df: pl.DataFrame, columns: Iterable[str], table: str) -> list[str]:
    """Messages for not-null columns of ``df`` that contain nulls."""
    present = [c for c in columns if c in df.columns]
    if not present:
        return []
    counts = df.select(pl.col(present).null_count()).row(0)
    return [
        f"{table}: column {c!r} has {n} null value(s) but must not be null"
        for c, n in zip(present, counts)
        if n
    ]


def runs_problems(
    df: pl.DataFrame, dataset_id: str, source_paths: set[str] | None
) -> list[str]:
    """Problems with a runs table that already has the format's columns and types.

    ``source_paths`` are the manifest's source paths; None skips that check.
    """
    problems = null_problems(
        df, [c.name for c in RUNS_COLUMNS if not c.nullable], "runs"
    )
    if problems:
        return problems

    expected = run_ids(dataset_id, df["source_file"], df["source_run_index"])
    wrong = df.filter(df["run_id"] != expected)
    if wrong.height:
        r = wrong.row(0, named=True)
        problems.append(
            f"runs: {wrong.height} run id(s) differ from the stable hash, for example run {r['run_id']} "
            f"({r['source_file']!r}, index {r['source_run_index']}) should be "
            f"{expected.filter(df['run_id'] != expected)[0]}"
        )
    if df["run_id"].n_unique() != df.height:
        problems.append("runs: run_id is not unique")
    if (
        df.select(pl.struct("source_file", "source_run_index").n_unique()).item()
        != df.height
    ):
        problems.append("runs: (source_file, source_run_index) is not unique")

    for column, condition, rule in (
        ("dimension", pl.col("dimension") < 1, "at least 1"),
        ("source_run_index", pl.col("source_run_index") < 0, ">= 0"),
        ("n_objectives", pl.col("n_objectives") != 1, "1 in format version 1"),
    ):
        n = df.filter(condition).height
        if n:
            problems.append(f"runs: {n} run(s) have {column} that is not {rule}")

    for path in df["source_file"].unique().sort().to_list():
        problem = source_path_problem(path)
        if problem is not None:
            problems.append(f"runs: invalid source_file {path!r}: {problem}")
        elif source_paths is not None and path not in source_paths:
            problems.append(
                f"runs: source_file {path!r} is not listed in the manifest's sources"
            )

    bad_sha = df.filter(
        pl.col("source_rows_sha256").is_not_null()
        & ~pl.col("source_rows_sha256").str.contains(f"^{_SHA256.pattern}$")
    )
    if bad_sha.height:
        problems.append(
            f"runs: {bad_sha.height} source_rows_sha256 value(s) are not 64 lower-case hex digits"
        )

    for text in df["attributes"].unique().to_list():
        problem = attributes_problem(text)
        if problem is not None:
            problems.append(f"runs: attributes {text[:80]!r}: {problem}")
            break
    return problems


def attributes_problem(text: str) -> str | None:
    """Why ``text`` is not a valid ``attributes`` value, or None."""
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except ValueError as exc:
        return f"not valid JSON ({exc})"
    if not isinstance(value, dict):
        return "not a JSON object"
    return None


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not allowed in JSON")


def attributes_json(attributes: dict[str, Any] | None) -> str:
    """Canonical ``attributes`` text: a JSON object with sorted keys and compact separators.

    Raises ``ValueError`` for NaN or infinite numbers, which JSON cannot hold.
    """
    return json.dumps(
        attributes or {},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sort_runs(df: pl.DataFrame) -> pl.DataFrame:
    """Runs in run order (``RUNS_SORT_KEYS`` ascending; strings by UTF-8 bytes)."""
    return df.sort(list(RUNS_SORT_KEYS), maintain_order=True)


def run_map(sorted_runs: pl.DataFrame) -> pl.DataFrame:
    """Per run: id, copied columns, its run order (rank) and its row-group key group."""
    return sorted_runs.with_row_index(RANK).select(
        "run_id",
        "function_id",
        "dimension",
        pl.col(RANK).cast(pl.Int64),
        pl.struct(list(ROW_GROUP_KEYS)).rle_id().cast(pl.Int64).alias(GROUP),
        pl.col("dimension").alias(RUN_DIM),
        "source_file",
        "source_run_index",
    )


def describe_runs(run_map_df: pl.DataFrame) -> Callable[[int], str]:
    """A function that names the run of a rank, for error messages."""
    ids = run_map_df["run_id"].to_list()
    files = run_map_df["source_file"].to_list()
    index = run_map_df["source_run_index"].to_list()

    def describe(rank: int) -> str:
        if 0 <= rank < len(ids):
            return f"run {ids[rank]} ({files[rank]!r}, index {index[rank]})"
        return f"run at rank {rank}"

    return describe


class OrderChecker:
    """Checks that sort keys (rank * 2**32 + row) arrive in format order.

    With ``contiguous`` (the trace), each run's rows must be 0, 1, 2, ... in order.
    Without it (positions), keys must only be strictly increasing.
    """

    def __init__(self, table: str, contiguous: bool, describe: Callable[[int], str]):
        self.table = table
        self.contiguous = contiguous
        self.describe = describe
        self.last = -1

    def check(self, keys: pl.Series) -> str | None:
        """The first problem in ``keys``, which follow all keys checked before, or None."""
        if keys.len() == 0:
            return None
        frame = pl.DataFrame({"key": keys}).with_columns(
            prev=pl.col("key").shift(1, fill_value=self.last)
        )
        frame = frame.with_columns(
            rank=pl.col("key") // ROW_LIMIT,
            row=pl.col("key") % ROW_LIMIT,
            prank=pl.col("prev") // ROW_LIMIT,
            prow=pl.col("prev") % ROW_LIMIT,
        )
        ok = pl.col("key") > pl.col("prev")
        if self.contiguous:
            same = pl.col("rank") == pl.col("prank")
            ok = ok & (
                (same & (pl.col("key") == pl.col("prev") + 1))
                | (~same & (pl.col("row") == 0))
            )
        bad = frame.filter(~ok)
        self.last = int(keys[-1])
        if bad.height == 0:
            return None
        b = bad.row(0, named=True)
        run = self.describe(b["rank"])
        if b["key"] == b["prev"]:
            return f"{self.table}: {run} has row {b['row']} twice"
        if b["rank"] < b["prank"]:
            return (
                f"{self.table}: rows are not in run order: {run} comes after "
                f"{self.describe(b['prank'])}, or its rows are not contiguous"
            )
        if b["rank"] == b["prank"]:
            if b["row"] < b["prow"]:
                return (
                    f"{self.table}: {run}: row {b['row']} comes after row {b['prow']}"
                )
            return f"{self.table}: {run}: row {b['row']} follows row {b['prow']}; rows must have no gaps"
        return f"{self.table}: {run}: the first row is {b['row']}; rows must start at 0"


def sort_key(rank: pl.Expr, row: pl.Expr) -> pl.Expr:
    """The single Int64 sort key of a row."""
    return rank * ROW_LIMIT + row


def first_problem(problems: Sequence[str]) -> str:
    """Join problems into one exception message."""
    if len(problems) == 1:
        return problems[0]
    return f"{len(problems)} problems:\n  " + "\n  ".join(problems)


def iter_frames(data: Any, slice_rows: int, what: str) -> Iterator[pl.DataFrame]:
    """DataFrame chunks from a DataFrame, a LazyFrame or an iterable of either."""
    if isinstance(data, pl.DataFrame):
        yield from data.iter_slices(slice_rows)
    elif isinstance(data, pl.LazyFrame):
        yield from _lazy_batches(data, slice_rows)
    elif isinstance(data, (str, bytes)) or not isinstance(data, Iterable):
        raise TypeError(
            f"{what} must be a polars DataFrame, a LazyFrame or an iterable of DataFrames"
        )
    else:
        for item in data:
            if isinstance(item, pl.LazyFrame):
                yield from _lazy_batches(item, slice_rows)
            elif isinstance(item, pl.DataFrame):
                yield item
            else:
                raise TypeError(
                    f"{what} chunks must be polars DataFrames, got {type(item).__name__}"
                )


def _lazy_batches(lf: pl.LazyFrame, slice_rows: int) -> Iterator[pl.DataFrame]:
    collect_batches = getattr(lf, "collect_batches", None)
    if collect_batches is None:  # polars versions before collect_batches
        yield from lf.collect().iter_slices(slice_rows)
        return
    yield from collect_batches(chunk_size=slice_rows)
