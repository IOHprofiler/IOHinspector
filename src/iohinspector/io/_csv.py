"""The CSV reader shared by the Nevergrad and custom CSV readers.

A ``CsvMapping`` says how the columns of a CSV file become runs, a trace and
descriptors. The file is read once: each row is checked and assigned to its run,
and the trace goes to temporary files in chunks (``Tables``), so memory use grows
with the number of runs and not with the number of rows. Column types are
inferred from all tokens of a column. Not public API: use
``iohinspector.io.nevergrad`` or ``iohinspector.io.custom_csv``.
"""

from __future__ import annotations

import csv
import difflib
import gzip
import hashlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import polars as pl

from ..format import file_entry, run_id
from ..format.schema import (
    ALGORITHM_DESCRIPTORS,
    RESERVED_DESCRIPTOR_KEYS,
    is_reserved_trace_name,
)
from .base import (
    INTEGER,
    NUMBER,
    ReaderError,
    ReadResult,
    Tables,
    check_floats,
    check_integers,
)

CHUNK_ROWS = 100_000
BOOLS = {"true": True, "false": False, "1": True, "0": False}


class MappingError(ValueError):
    """A column mapping that is not valid, or does not fit the file's columns."""


@dataclass(frozen=True)
class Source:
    """Where one run-level value comes from: a column, a constant or a function.

    ``fn`` receives a function that gives the token of a named column in the row.
    """

    column: str | None = None
    constant: Any = None
    fn: Callable[[Callable[[str], str]], Any] | None = None


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class CsvMapping:
    """How the columns of a CSV file become runs, a trace and descriptors."""

    evaluations: str
    value: str
    fields: Mapping[str, Source]
    delimiter: str = ","
    run_by: Sequence[str] = ()
    descriptors: Sequence[str] | str = ()
    trace_columns: Sequence[str] = ()
    positions: Sequence[str] = ()


# Run-level fields: name -> (kind, required).
FIELDS: dict[str, tuple[str, bool]] = {
    "algorithm_name": ("string", True),
    "algorithm_info": ("string", False),
    "suite": ("string", False),
    "function_id": ("int", False),
    "function_name": ("string", False),
    "instance": ("int", False),
    "dimension": ("int", True),
    "maximization": ("bool", True),
    "optimum": ("float", False),
    "budget": ("int", False),
    "evals_used": ("int", False),
}


def parse_token(token: str, kind: str) -> Any:
    """A token as ``int``, ``float``, ``bool`` or ``string`` (``str``); raises
    ``ValueError`` saying what it is not. Numbers follow the format's value rules."""
    if kind == "int" and INTEGER.fullmatch(token):
        return int(token)
    if kind == "float" and NUMBER.fullmatch(token):
        return float(token)
    if kind == "bool" and token.lower() in BOOLS:
        return BOOLS[token.lower()]
    if kind in ("string", "str"):
        return token
    what = {"int": "an integer", "float": "a number", "bool": "a boolean"}[kind]
    hint = " (use true or false)" if kind == "bool" else ""
    raise ValueError(f"{token!r} is not {what}{hint}")


class _Types:
    """The type of a CSV column: bool, int, float or string, from all its tokens.

    Nothing is converted inexactly: an integer too large for float64 keeps the
    column from being float, and for a trace column (int64) it makes it a string.
    """

    __slots__ = ("big_int", "bool_ok", "float_ok", "int64_ok", "int_ok", "seen")

    def __init__(self) -> None:
        self.bool_ok = self.int_ok = self.int64_ok = self.float_ok = True
        self.big_int = False
        self.seen = False

    def add(self, token: str) -> None:
        if not token:
            return
        self.seen = True
        if self.bool_ok and token.lower() not in ("true", "false"):
            self.bool_ok = False
        if self.float_ok and not NUMBER.fullmatch(token):
            self.float_ok = False
        if INTEGER.fullmatch(token):
            value = abs(int(token))
            self.big_int = self.big_int or value > 2**53
            self.int64_ok = self.int64_ok and value < 2**63
        else:
            self.int_ok = False

    def kind(self, int64: bool = False) -> str:
        """The type name; ``int64`` for a column that must fit int64."""
        if not self.seen:
            return "string"
        if self.bool_ok:
            return "bool"
        if self.int_ok:
            return "int" if self.int64_ok or not int64 else "string"
        return "float" if self.float_ok and not self.big_int else "string"


def _cast(name: str, kind: str) -> pl.Expr:
    """A trace column of tokens as its type; empty tokens are null."""
    col = pl.when(pl.col(name) != "").then(pl.col(name))
    if kind == "bool":
        return (col.str.to_lowercase() == "true").alias(name)
    if kind == "int":
        return col.cast(pl.Int64).alias(name)
    if kind == "float":
        return col.str.to_lowercase().cast(pl.Float64).alias(name)
    return col.alias(name)


def open_text(path: Path) -> IO[str]:
    """The file as UTF-8 text; ``.gz`` files are decompressed."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return open(path, encoding="utf-8-sig", newline="")


def read_header(path: Path, delimiter: str = ",") -> list[str]:
    """The column names of a CSV file."""
    try:
        with open_text(path) as fh:
            header = next(csv.reader(fh, delimiter=delimiter), None)
    except UnicodeDecodeError as exc:
        raise ReaderError(f"not UTF-8 text ({exc})", path.name) from exc
    if not header:
        raise ReaderError("the file is empty", path.name)
    return header


def suggest(name: str, known: Sequence[str]) -> str:
    """`` Did you mean 'x'?`` for the closest of ``known``, or ""."""
    close = difflib.get_close_matches(name, list(known), n=1)
    return f" Did you mean {close[0]!r}?" if close else ""


def check_mapping(mapping: CsvMapping, header: Sequence[str]) -> None:
    """Raise ``MappingError`` when the mapping names a column the file lacks."""
    # #lizard forgives  (one check per kind of mapping error)
    if len(set(header)) != len(header):
        dup = sorted({h for h in header if list(header).count(h) > 1})
        raise MappingError(f"the file has duplicate column names: {dup}")
    used = [("evaluations", mapping.evaluations), ("value", mapping.value)]
    used += [(f"fields.{k}", s.column) for k, s in mapping.fields.items() if s.column]
    used += [("run_by", c) for c in mapping.run_by]
    if mapping.descriptors != "rest":
        used += [("descriptors", c) for c in mapping.descriptors]
    used += [("trace_columns", c) for c in mapping.trace_columns]
    used += [("positions", c) for c in mapping.positions]
    for role, column in used:
        if column not in header:
            shown = ", ".join(repr(k) for k in header[:30])
            hint = suggest(column, header)
            raise MappingError(
                f"{role}: column {column!r} is not in the file{'.' if hint else ''}"
                f"{hint} (columns: {shown})"
            )
    if mapping.evaluations == mapping.value:
        raise MappingError("evaluations and value name the same column")
    for role, names in (
        ("trace_columns", mapping.trace_columns),
        ("positions", mapping.positions),
    ):
        clash = {mapping.evaluations, mapping.value} & set(names)
        if clash:
            raise MappingError(f"{role} repeats the column {min(clash)!r}")
    both = set(mapping.trace_columns) & set(mapping.positions)
    if both:
        raise MappingError(f"column {min(both)!r} is in trace_columns and positions")


def _csv_file(source: Path) -> tuple[Path, str]:
    """The CSV file to read (``source`` or the only one in it) and its source path."""
    if not source.is_dir():
        return source, source.name
    files = sorted(
        p for p in source.rglob("*") if p.is_file() and ".csv" in p.name.lower()
    )
    if len(files) != 1:
        raise ReaderError(f"holds {len(files)} CSV files; give one", source)
    return files[0], files[0].relative_to(source).as_posix()


@dataclass
class _Run:
    """One run: its run-level values, the tokens of its run-level columns in its
    first row, the hash of its rows and where it starts."""

    rid: int
    values: dict[str, Any]
    attrs: tuple[str, ...]
    first_line: int
    digest: Any = field(default_factory=hashlib.sha256)
    rows: int = 0


def read_csv(  # pylint: disable=too-many-arguments
    source: str | Path,
    dataset_id: str,
    mapping: CsvMapping | Callable[[list[str]], CsvMapping],
    *,
    reader_name: str,
    origin: str,
    reader_options: Mapping[str, Any] | None = None,
) -> ReadResult:
    """Read the CSV file ``source`` (or the only one in the folder ``source``).

    ``mapping`` is a ``CsvMapping``, or a function that makes one from the file's
    column names. Descriptor columns get the origin ``origin``; ``reader_name`` is
    also the source kind.
    """
    path, name = _csv_file(Path(source))
    delimiter = mapping.delimiter if isinstance(mapping, CsvMapping) else ","
    header = read_header(path, delimiter)
    if callable(mapping):
        mapping = mapping(header)
    reader = _CsvReader(Tables(dataset_id), name, mapping, header)
    for line, row in _rows(path, name, mapping.delimiter, len(header)):
        reader.add_row(line, row)
    reader.flush()
    if not reader.runs:
        raise ReaderError("the file has no data rows", name)
    function_ids = reader.finish(origin)
    reader.tables.sources.append(
        file_entry(path, name, kind=reader_name, data_rows=reader.n)
    )
    result = reader.tables.result(
        reader_name,
        reader_options=dict(reader_options or {}),
        extra={"function_ids": function_ids} if function_ids else None,
    )
    result.lazy_trace = result.lazy_trace.with_columns(
        _cast(reader.extra_names[c], t.kind(int64=True))
        for c, t in reader.types.items()
    )
    return result


def _rows(
    path: Path, name: str, delimiter: str, n: int
) -> Iterator[tuple[int, list[str]]]:
    """``(line number, tokens)`` of each data row; blank lines are skipped."""
    try:
        with open_text(path) as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                if len(row) != n:
                    raise ReaderError(
                        f"expected {n} fields like the header, found {len(row)}",
                        name,
                        reader.line_num,
                    )
                yield reader.line_num, row
    except UnicodeDecodeError as exc:
        raise ReaderError(f"not UTF-8 text ({exc})", name) from exc
    except csv.Error as exc:
        raise ReaderError(str(exc), name) from exc


def _descriptor_columns(
    m: CsvMapping, header: list[str], field_cols: list[str]
) -> list[str]:
    """The columns kept as descriptors: the mapping's list, or for ``"rest"``
    every column not used otherwise, plus the ``run_by`` columns that are."""
    if m.descriptors != "rest":
        return list(m.descriptors)
    used = {m.evaluations, m.value, *field_cols}
    rest = used | set(m.trace_columns) | set(m.positions)
    return [h for h in header if h not in rest or (h in m.run_by and h in used)]


class _CsvReader:  # pylint: disable=too-many-instance-attributes
    """Reads the rows of one CSV file with a mapping into ``tables``."""

    def __init__(
        self, tables: Tables, name: str, mapping: CsvMapping, header: list[str]
    ) -> None:
        check_mapping(mapping, header)
        self.tables, self.name, self.mapping = tables, name, mapping
        self.col = {h: i for i, h in enumerate(header)}
        m = mapping
        field_cols = [s.column for s in m.fields.values() if s.column]
        self.descriptor_cols = _descriptor_columns(m, header, field_cols)
        self.n_field_cols = len(field_cols)
        self.attr_idx = [self.col[c] for c in field_cols + self.descriptor_cols]
        self.key_idx = [self.col[c] for c in m.run_by]
        self.descriptor_keys = self._renamed(
            self.descriptor_cols,
            "descriptors",
            lambda k: k in RESERVED_DESCRIPTOR_KEYS or k in ALGORITHM_DESCRIPTORS,
            set(ALGORITHM_DESCRIPTORS),
        )
        self.extra_names = self._renamed(
            list(m.trace_columns), "trace", is_reserved_trace_name, set()
        )
        self.types = {c: _Types() for c in m.trace_columns}
        self.descriptor_types = {c: _Types() for c in self.descriptor_cols}
        self.runs: list[_Run] = []
        self.index: dict[tuple[str, ...], int] = {}
        #: Buffered rows: line number, run index, row within the run, tokens.
        self.chunk: list[tuple[int, int, int, list[str]]] = []
        self.n = 0

    def _renamed(
        self,
        names: list[str],
        table: str,
        reserved: Callable[[str], bool],
        taken: Iterable[str],
    ) -> dict[str, str]:
        """Source column -> stored name, with ``_src`` added where reserved."""
        out: dict[str, str] = {}
        for name in names:
            others = {*taken, *out.values(), *names} - {name}
            where = ("src", self.name, table)
            out[name] = self.tables.renamer.rename(name, others, where, reserved)
        return out

    def add_row(self, line: int, row: list[str]) -> None:
        attrs = tuple(row[i] for i in self.attr_idx)
        key = tuple(row[i] for i in self.key_idx) if self.key_idx else None
        r = self.index.get(key) if key is not None else None
        if r is None:
            r = self._new_run(row, attrs, line)
            if key is not None:
                self.index[key] = r
        elif attrs != self.runs[r].attrs:
            self._constancy_error(self.runs[r], attrs, line)
        run = self.runs[r]
        run.digest.update((" ".join(row) + "\n").encode("utf-8"))
        for c, types in self.types.items():
            types.add(row[self.col[c]])
        self.chunk.append((line, r, run.rows, row))
        run.rows += 1
        self.n += 1
        if len(self.chunk) >= CHUNK_ROWS:
            self.flush()

    def _new_run(self, row: list[str], attrs: tuple[str, ...], line: int) -> int:
        index = len(self.runs)
        rid = run_id(self.tables.dataset_id, self.name, index)
        self.runs.append(_Run(rid, self._field_values(row, line), attrs, line))
        n = self.n_field_cols
        for c, token in zip(self.descriptor_cols, attrs[n:]):
            self.descriptor_types[c].add(token)
        return index

    def _field_values(self, row: list[str], line: int) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, source in self.mapping.fields.items():
            if source.fn is not None:
                raw: Any = source.fn(lambda c: row[self.col[c]])
            elif source.column is not None:
                raw = row[self.col[source.column]]
            else:
                values[name] = source.constant
                continue
            kind, required = FIELDS[name]
            if raw == "" or raw is None:
                if required:
                    raise ReaderError(
                        f"{name} is empty (column {source.column!r})", self.name, line
                    )
                raw = None
            elif isinstance(raw, str):
                try:
                    raw = parse_token(raw, kind)
                except ValueError as exc:
                    raise ReaderError(f"{name}: {exc}", self.name, line) from None
            values[name] = raw
        return values

    def _constancy_error(self, run: _Run, attrs: tuple[str, ...], line: int) -> None:
        cols = [s.column for s in self.mapping.fields.values() if s.column]
        for name, a, b in zip(cols + self.descriptor_cols, run.attrs, attrs):
            if a != b:
                by = ", ".join(self.mapping.run_by)
                raise ReaderError(
                    f"column {name!r} is {b!r}, but {a!r} in line {run.first_line}, "
                    f"which has the same value of {by}. A column that describes a "
                    "run must not change within it; add it to run_by or list it "
                    "under trace_columns",
                    self.name,
                    line,
                )

    def flush(self) -> None:
        """Check the numbers of the buffered rows and keep their trace."""
        if not self.chunk:
            return
        m, name = self.mapping, self.name
        lines = [c[0] for c in self.chunk]

        def tokens(column: str) -> pl.Series:
            i = self.col[column]
            return pl.Series([c[3][i] for c in self.chunk], dtype=pl.String)

        keys = pl.DataFrame(
            {
                "run_id": [self.runs[c[1]].rid for c in self.chunk],
                "row": [c[2] for c in self.chunk],
            },
            schema={"run_id": pl.Int64, "row": pl.Int64},
        )
        trace = keys.with_columns(
            check_integers(tokens(m.evaluations), lines, name, m.evaluations).alias(
                "evaluations"
            ),
            check_floats(tokens(m.value), lines, name, m.value).alias("raw_y"),
            *(tokens(c).alias(self.extra_names[c]) for c in m.trace_columns),
        )
        self.tables.add("trace", trace)
        if m.positions:
            xs = [
                check_floats(tokens(c), lines, name, c).alias(f"x{j}")
                for j, c in enumerate(m.positions)
            ]
            self.tables.add("positions", keys.with_columns(xs))
        self.chunk = []

    def finish(self, origin: str) -> dict[str, int] | None:
        """Check positions, add runs and descriptors; returns the function ids
        assigned to function names (None when the mapping gives the ids)."""
        d = len(self.mapping.positions)
        for run in self.runs:
            if d and run.values["dimension"] != d:
                raise ReaderError(
                    f"the run has dimension {run.values['dimension']}, but "
                    f"positions lists {d} columns",
                    self.name,
                    run.first_line,
                )
        function_ids = self._function_ids()
        kinds = [self.descriptor_types[c].kind() for c in self.descriptor_cols]
        for index, run in enumerate(self.runs):
            values = run.values
            self.tables.runs.append(
                {k: values.get(k) for k in FIELDS if k not in ALGORITHM_DESCRIPTORS}
                | {
                    "run_id": run.rid,
                    "suite": values.get("suite", ""),
                    "n_objectives": 1,
                    "source_file": self.name,
                    "run_index": index,
                    "source_rows_sha256": run.digest.hexdigest(),
                }
            )
            self._descriptors(run, kinds, origin)
        return function_ids

    def _function_ids(self) -> dict[str, int] | None:
        """Without ``function_id`` in the mapping: ids 1, 2, ... for the sorted
        function names (all 1 without names)."""
        fields = self.mapping.fields
        if "function_id" in fields:
            return None
        names = [r.values.get("function_name") or "" for r in self.runs]
        ids = {n: i for i, n in enumerate(sorted(set(names)), start=1)}
        for run, name in zip(self.runs, names):
            run.values["function_id"] = ids[name]
        return ids if "function_name" in fields else None

    def _descriptors(self, run: _Run, kinds: list[str], origin: str) -> None:
        out = self.tables.descriptors
        values = run.values
        out.append((run.rid, "algorithm_name", values["algorithm_name"], "algorithm"))
        if values.get("algorithm_info"):
            out.append(
                (run.rid, "algorithm_info", values["algorithm_info"], "algorithm")
            )
        n = self.n_field_cols
        tokens = run.attrs[n:]
        for c, kind, token in zip(self.descriptor_cols, kinds, tokens):
            if token:
                value = parse_token(token, kind)
                out.append((run.rid, self.descriptor_keys[c], value, origin))
