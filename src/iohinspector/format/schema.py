"""Column schemas and constants of dataset format v1.

The normative description is ``docs/format.md``. This module only encodes it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl
import pyarrow as pa

FORMAT_NAME = "iohinspector-dataset"
FORMAT_MAJOR = 1
FORMAT_MINOR = 0
FORMAT_VERSION = f"{FORMAT_MAJOR}.{FORMAT_MINOR}"

MANIFEST_FILE = "manifest.json"
RUNS_FILE = "runs.parquet"
DESCRIPTORS_FILE = "descriptors.parquet"
TRACE_FILE = "trace.parquet"
POSITIONS_FILE = "positions.parquet"
DATASET_FILES = (MANIFEST_FILE, RUNS_FILE, DESCRIPTORS_FILE, TRACE_FILE, POSITIONS_FILE)

#: Parquet key-value metadata written into every table.
KV_FORMAT_VERSION = "iohinspector.format_version"
KV_TABLE = "iohinspector.table"


@dataclass(frozen=True)
class Column:
    """One fixed column: name, polars type, and whether it may hold nulls."""

    name: str
    dtype: pl.DataType
    nullable: bool


RUNS_COLUMNS: tuple[Column, ...] = (
    Column("run_id", pl.Int64(), False),
    Column("suite", pl.String(), False),
    Column("function_id", pl.Int32(), False),
    Column("function_name", pl.String(), True),
    Column("instance", pl.Int32(), True),
    Column("dimension", pl.Int32(), False),
    Column("maximization", pl.Boolean(), False),
    Column("optimum", pl.Float64(), True),
    Column("budget", pl.Int64(), True),
    Column("evals_used", pl.Int64(), True),
    Column("best_y", pl.Float64(), True),
    Column("n_objectives", pl.Int32(), False),
    Column("source_file", pl.String(), False),
    Column("source_run_index", pl.Int64(), False),
    Column("source_rows_sha256", pl.String(), True),
)

#: Descriptors that every run has, of type string and origin ``algorithm`` or
#: ``user``. They are not columns of runs.parquet.
ALGORITHM_DESCRIPTORS = ("algorithm_name", "algorithm_info")
ALGORITHM_DESCRIPTOR_ORIGINS = ("algorithm", "user")

#: Runs columns a writer must be given; the others have defaults (see
#: ``RUNS_DEFAULTS``).
RUNS_REQUIRED_INPUT = (
    "function_id",
    "dimension",
    "maximization",
    "source_file",
    "source_run_index",
)

#: Defaults for runs columns the caller leaves out. ``run_id`` is computed.
RUNS_DEFAULTS: dict[str, str | int | None] = {
    "suite": "",
    "function_name": None,
    "instance": None,
    "optimum": None,
    "budget": None,
    "evals_used": None,
    "best_y": None,
    "n_objectives": 1,
    "source_rows_sha256": None,
}

#: Sort order of runs.parquet; the trace follows it. No descriptor is a sort key
#: (D17), so editing descriptors never reorders the trace.
RUNS_SORT_KEYS = (
    "suite",
    "function_id",
    "dimension",
    "source_file",
    "source_run_index",
)

#: Row groups of the trace preferably start where these change.
ROW_GROUP_KEYS = ("suite", "function_id", "dimension")

TRACE_COLUMNS: tuple[Column, ...] = (
    Column("run_id", pl.Int64(), False),
    Column("function_id", pl.Int32(), False),
    Column("dimension", pl.Int32(), False),
    Column("row", pl.Int64(), False),
    Column("evaluations", pl.Int64(), False),
    Column("raw_y", pl.Float64(), False),
)

POSITIONS_KEY_COLUMNS: tuple[Column, ...] = TRACE_COLUMNS[:4]

#: Columns copied from the run into every trace and positions row.
COPIED_RUN_COLUMNS = ("function_id", "dimension")

#: Statistics that must exist for every row group of the trace.
TRACE_STATISTICS_COLUMNS = ("run_id", "function_id", "dimension", "row", "evaluations")

#: Types allowed for extra trace columns.
EXTRA_COLUMN_DTYPES: tuple[pl.DataType, ...] = (
    pl.Float64(),
    pl.Int64(),
    pl.Boolean(),
    pl.String(),
)

RUNS_SCHEMA: dict[str, pl.DataType] = {c.name: c.dtype for c in RUNS_COLUMNS}
TRACE_SCHEMA: dict[str, pl.DataType] = {c.name: c.dtype for c in TRACE_COLUMNS}
POSITIONS_KEY_SCHEMA: dict[str, pl.DataType] = {
    c.name: c.dtype for c in POSITIONS_KEY_COLUMNS
}

DESCRIPTOR_COLUMNS: tuple[Column, ...] = (
    Column("run_id", pl.Int64(), False),
    Column("key", pl.String(), False),
    Column("origin", pl.String(), False),
    Column("type", pl.String(), False),
    Column("value", pl.String(), False),
    Column("value_int", pl.Int64(), True),
    Column("value_float", pl.Float64(), True),
    Column("value_bool", pl.Boolean(), True),
)
DESCRIPTORS_SCHEMA: dict[str, pl.DataType] = {
    c.name: c.dtype for c in DESCRIPTOR_COLUMNS
}
DESCRIPTOR_ORIGINS = ("algorithm", "experiment", "path", "run", "user")
DESCRIPTOR_TYPES = ("bool", "float", "int", "string")
#: The typed column each type uses (``int`` only when the value fits in int64).
TYPED_VALUE_COLUMN = {"int": "value_int", "float": "value_float", "bool": "value_bool"}
DESCRIPTORS_SORT_KEYS = ("key", "run_id")
MAX_DESCRIPTOR_KEY_LENGTH = 256

_RESERVED_PATTERN = re.compile(r"[xy][0-9]+")
#: Names an extra trace column may not have. The algorithm descriptors are included
#: because analysis joins them to the trace.
RESERVED_TRACE_NAMES = (
    frozenset(TRACE_SCHEMA) | frozenset(RUNS_SCHEMA) | frozenset(ALGORITHM_DESCRIPTORS)
)

#: Keys a descriptor may not have: ``by`` resolves names in runs.parquet and the
#: descriptors, and the result is joined to the trace.
RESERVED_DESCRIPTOR_KEYS = frozenset(RUNS_SCHEMA) | frozenset(TRACE_SCHEMA)
_CONTROL = re.compile(r"[\x00-\x1f]")


def descriptor_key_problem(key: object) -> str | None:
    """Why ``key`` is not a valid descriptor key, or None."""
    if not isinstance(key, str) or not key or len(key) > MAX_DESCRIPTOR_KEY_LENGTH:
        return f"keys must have 1 to {MAX_DESCRIPTOR_KEY_LENGTH} characters"
    if _CONTROL.search(key):
        return "keys must not contain control characters"
    if key in RESERVED_DESCRIPTOR_KEYS:
        return "the name of a runs or trace column is reserved"
    return None


def is_reserved_trace_name(name: str) -> bool:
    """True if ``name`` may not be used for an extra trace column."""
    return name in RESERVED_TRACE_NAMES or _RESERVED_PATTERN.fullmatch(name) is not None


def position_columns(max_dimension: int) -> list[str]:
    """Names of the coordinate columns, ``x0`` to ``x(D-1)``."""
    return [f"x{i}" for i in range(max_dimension)]


def objective_columns(max_objectives: int) -> list[str]:
    """Names of the trace columns of objectives 1 to M-1, ``y1`` to ``y(M-1)``.

    Objective 0 is ``raw_y``, so there is no ``y0``; a single-objective dataset
    (M = 1) has no objective columns.
    """
    return [f"y{i}" for i in range(1, max_objectives)]


def indexed_columns(names: list[str]) -> list[tuple[str, int]]:
    """``(name, index)`` for columns named a letter and an index, such as ``x3``."""
    return [(name, int(name[1:])) for name in names]


_ARROW_TYPES = {
    pl.Int32(): pa.int32(),
    pl.Int64(): pa.int64(),
    pl.Float64(): pa.float64(),
    pl.Boolean(): pa.bool_(),
    pl.String(): pa.large_string(),
}


def arrow_type(dtype: pl.DataType) -> pa.DataType:
    """The Arrow type a format column of polars type ``dtype`` is written as."""
    for pl_type, pa_type in _ARROW_TYPES.items():
        if dtype == pl_type:
            return pa_type
    raise TypeError(f"no format type for polars dtype {dtype}")


def arrow_schema(columns: list[Column], table: str) -> pa.Schema:
    """Arrow schema with nullability and the format's key-value metadata."""
    fields = [
        pa.field(c.name, arrow_type(c.dtype), nullable=c.nullable) for c in columns
    ]
    return pa.schema(
        fields, metadata={KV_FORMAT_VERSION: FORMAT_VERSION, KV_TABLE: table}
    )
