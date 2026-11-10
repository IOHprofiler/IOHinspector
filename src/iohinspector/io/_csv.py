"""A streaming CSV converter shared by the Nevergrad and custom CSV readers.

A CSV file is read twice. The first pass checks every token, builds the runs and
infers the column types; the second pass yields the trace, descriptors and
positions in chunks, so memory use grows with the number of runs and not with the
number of rows. Not public API: use ``iohinspector.io.nevergrad`` or
``iohinspector.io.custom_csv``.
"""

from __future__ import annotations

import csv
import difflib
import gzip
import hashlib
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import polars as pl

from iohinspector.format import (
    FormatError,
    check_dataset_id,
    check_source_path,
    descriptor_frame,
    file_entry,
    is_reserved_trace_name,
    new_manifest,
    run_ids,
    write_dataset,
)
from iohinspector.format.schema import ALGORITHM_DESCRIPTORS, RESERVED_DESCRIPTOR_KEYS

CHUNK_ROWS = 100_000

_INT = re.compile(r"[+-]?[0-9]+")
_FLOAT = re.compile(
    r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[+-]?(?:inf|nan)",
    re.IGNORECASE,
)
_BOOLS = {"true": True, "false": False, "1": True, "0": False}


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
_RUN_FIELDS = tuple(f for f in FIELDS if not f.startswith("algorithm_"))


def parse_float(token: str, where: str) -> float:
    """A float64 from a token, accepting only plain decimals, ``inf`` and ``nan``."""
    if not _FLOAT.fullmatch(token):
        raise FormatError(f"{where}: {token!r} is not a number")
    return float(token)


def parse_int(token: str, where: str) -> int:
    if not _INT.fullmatch(token):
        raise FormatError(f"{where}: {token!r} is not an integer")
    return int(token)


def parse_bool(token: str, where: str) -> bool:
    try:
        return _BOOLS[token.lower()]
    except KeyError:
        raise FormatError(
            f"{where}: {token!r} is not a boolean (use true or false)"
        ) from None


_PARSERS: dict[str, Callable[[str, str], Any]] = {
    "int": parse_int,
    "float": parse_float,
    "bool": parse_bool,
    "string": lambda token, where: token,
}


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
        if token == "":
            return
        self.seen = True
        if self.bool_ok and token.lower() not in ("true", "false"):
            self.bool_ok = False
        if self.float_ok and not _FLOAT.fullmatch(token):
            self.float_ok = False
        if _INT.fullmatch(token):
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


_DTYPES = {
    "bool": pl.Boolean(),
    "int": pl.Int64(),
    "float": pl.Float64(),
    "string": pl.String(),
}


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
        raise FormatError(f"{path.name}: not UTF-8 text ({exc})") from exc
    if not header:
        raise FormatError(f"{path.name}: the file is empty")
    return header


def _unknown(name: str, known: Sequence[str]) -> str:
    close = difflib.get_close_matches(name, list(known), n=1)
    hint = f". Did you mean {close[0]!r}?" if close else ""
    shown = ", ".join(repr(k) for k in known[:30])
    return f"column {name!r} is not in the file{hint} (columns: {shown})"


def check_mapping(mapping: CsvMapping, header: Sequence[str]) -> None:
    """Raise ``MappingError`` when the mapping names a column the file lacks."""
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
            raise MappingError(f"{role}: {_unknown(column, header)}")
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


@dataclass
class _Run:
    """One run found in the first pass."""

    values: dict[str, Any]
    attrs: tuple[str, ...]
    digest: Any
    first_line: int
    rows: int = 0


@dataclass
class _Scan:
    runs: list[_Run] = field(default_factory=list)
    extra_types: dict[str, _Types] = field(default_factory=dict)
    descriptor_types: dict[str, _Types] = field(default_factory=dict)
    rows: int = 0


def _rename(
    names: Sequence[str],
    table: str,
    source: str,
    clashes: Callable[[str], bool],
    taken: set[str],
    renamed: list[dict[str, str]],
) -> dict[str, str]:
    """A map from column names to the names they are stored under."""
    out: dict[str, str] = {}
    for name in names:
        new = name
        if clashes(name):
            new = f"{name}_src"
            n = 2
            while new in taken or new in names:
                new = f"{name}_src{n}"
                n += 1
            renamed.append({"from": name, "to": new, "source": source, "table": table})
        taken.add(new)
        out[name] = new
    return out


class CsvConverter:
    """Converts one CSV file with a mapping. See ``convert`` for the entry point."""

    def __init__(
        self,
        path: Path,
        source_name: str,
        mapping: CsvMapping,
        header: Sequence[str],
        origin: str,
    ) -> None:
        check_mapping(mapping, header)
        self.path = path
        self.name = source_name
        self.mapping = mapping
        self.header = list(header)
        self.col = {h: i for i, h in enumerate(header)}
        self.origin = origin
        self.renamed: list[dict[str, str]] = []
        self.function_ids: dict[str, int] | None = None
        self._plan_columns()

    def _plan_columns(self) -> None:
        m = self.mapping
        consumed = {m.evaluations, m.value}
        consumed |= {s.column for s in m.fields.values() if s.column}
        if m.descriptors == "rest":
            keep = [
                h
                for h in self.header
                if h not in consumed
                and h not in m.trace_columns
                and h not in m.positions
            ]
            keep += [h for h in m.run_by if h in consumed and h not in keep]
            keep = [h for h in self.header if h in keep]
        else:
            keep = list(m.descriptors)
        self.descriptor_cols = keep
        self.descriptor_keys = _rename(
            keep,
            "descriptors",
            self.name,
            lambda n: n in RESERVED_DESCRIPTOR_KEYS or n in ALGORITHM_DESCRIPTORS,
            set(ALGORITHM_DESCRIPTORS),
            self.renamed,
        )
        self.extra_cols = list(m.trace_columns)
        self.extra_names = _rename(
            self.extra_cols,
            "trace",
            self.name,
            is_reserved_trace_name,
            set(),
            self.renamed,
        )
        field_cols = [s.column for s in m.fields.values() if s.column]
        self.n_field_cols = len(field_cols)
        self.attr_cols = field_cols + self.descriptor_cols
        self.attr_idx = [self.col[c] for c in self.attr_cols]

    # -- reading -----------------------------------------------------------

    def _rows(self) -> Iterator[tuple[int, list[str]]]:
        """``(line number, tokens)`` of each data row; blank lines are skipped."""
        n = len(self.header)
        try:
            with open_text(self.path) as fh:
                reader = csv.reader(fh, delimiter=self.mapping.delimiter)
                next(reader)
                for row in reader:
                    if not row:
                        continue
                    if len(row) != n:
                        raise FormatError(
                            f"{self.name}:{reader.line_num}: expected {n} fields "
                            f"like the header, found {len(row)}"
                        )
                    yield reader.line_num, row
        except UnicodeDecodeError as exc:
            raise FormatError(f"{self.name}: not UTF-8 text ({exc})") from exc
        except csv.Error as exc:
            raise FormatError(f"{self.name}: {exc}") from exc

    def _run_key(self, row: list[str]) -> tuple[str, ...]:
        return tuple(row[self.col[c]] for c in self.mapping.run_by)

    def _field_values(self, row: list[str], where: str) -> dict[str, Any]:
        get = self._getter(row)
        values: dict[str, Any] = {}
        for name, (kind, required) in FIELDS.items():
            src = self.mapping.fields.get(name)
            if src is None:
                continue
            if src.fn is not None:
                raw: Any = src.fn(get)
            elif src.column is not None:
                raw = get(src.column)
            else:
                values[name] = src.constant
                continue
            if raw == "" or raw is None:
                if required:
                    raise FormatError(
                        f"{where}: {name} is empty (column {src.column!r})"
                    )
                values[name] = None
                continue
            values[name] = (
                raw
                if not isinstance(raw, str)
                else _PARSERS[kind](raw, f"{where}: {name}")
            )
        return values

    def _getter(self, row: list[str]) -> Callable[[str], str]:
        col = self.col
        return lambda name: row[col[name]]

    # -- first pass --------------------------------------------------------

    def scan(self) -> _Scan:
        m = self.mapping
        scan = _Scan()
        ev_i, val_i = self.col[m.evaluations], self.col[m.value]
        pos_i = [self.col[c] for c in m.positions]
        extra_i = [(c, self.col[c]) for c in self.extra_cols]
        scan.extra_types = {c: _Types() for c in self.extra_cols}
        scan.descriptor_types = {c: _Types() for c in self.descriptor_cols}
        index: dict[tuple[str, ...], int] = {}
        for line, row in self._rows():
            where = f"{self.name}:{line}"
            parse_int(row[ev_i], f"{where}: column {m.evaluations!r}")
            parse_float(row[val_i], f"{where}: column {m.value!r}")
            for c, i in zip(m.positions, pos_i, strict=True):
                parse_float(row[i], f"{where}: column {c!r}")
            for c, i in extra_i:
                scan.extra_types[c].add(row[i])
            attrs = tuple(row[i] for i in self.attr_idx)
            run = None
            if m.run_by:
                key = self._run_key(row)
                found = index.get(key)
                run = scan.runs[found] if found is not None else None
                if run is None:
                    index[key] = len(scan.runs)
            if run is None:
                run = _Run(
                    self._field_values(row, where), attrs, hashlib.sha256(), line
                )
                scan.runs.append(run)
                for c, tok in zip(
                    self.descriptor_cols, attrs[self.n_field_cols :], strict=True
                ):
                    scan.descriptor_types[c].add(tok)
            elif attrs != run.attrs:
                self._constancy_error(run, attrs, where)
            run.digest.update((" ".join(row) + "\n").encode("utf-8"))
            run.rows += 1
            scan.rows += 1
        if not scan.runs:
            raise FormatError(f"{self.name}: the file has no data rows")
        self._check_positions(scan)
        self._assign_function_ids(scan)
        return scan

    def _constancy_error(self, run: _Run, attrs: tuple[str, ...], where: str) -> None:
        for name, a, b in zip(self.attr_cols, run.attrs, attrs, strict=True):
            if a != b:
                by = ", ".join(self.mapping.run_by)
                raise FormatError(
                    f"{where}: column {name!r} is {b!r}, but {a!r} in line "
                    f"{run.first_line}, which has the same value of {by}. A column "
                    "that describes a run must not change within it; add it to run_by "
                    "or list it under trace_columns"
                )

    def _check_positions(self, scan: _Scan) -> None:
        d = len(self.mapping.positions)
        if not d:
            return
        for run in scan.runs:
            if run.values["dimension"] != d:
                raise FormatError(
                    f"{self.name}:{run.first_line}: the run has dimension "
                    f"{run.values['dimension']}, but positions lists {d} columns"
                )

    def _assign_function_ids(self, scan: _Scan) -> None:
        fields = self.mapping.fields
        if "function_id" in fields:
            return
        if "function_name" not in fields:
            for run in scan.runs:
                run.values["function_id"] = 1
            return
        names = sorted({r.values.get("function_name") or "" for r in scan.runs})
        self.function_ids = {n: i for i, n in enumerate(names, start=1)}
        for run in scan.runs:
            run.values["function_id"] = self.function_ids[
                run.values.get("function_name") or ""
            ]

    # -- tables ------------------------------------------------------------

    def runs_frame(self, scan: _Scan) -> pl.DataFrame:
        cols: dict[str, list[Any]] = {
            name: [r.values.get(name) for r in scan.runs]
            for name in (*_RUN_FIELDS, "function_id")
        }
        cols["source_file"] = [self.name] * len(scan.runs)
        cols["source_run_index"] = list(range(len(scan.runs)))
        cols["source_rows_sha256"] = [r.digest.hexdigest() for r in scan.runs]
        if "suite" not in self.mapping.fields:
            del cols["suite"]
        return pl.DataFrame(cols, schema_overrides=_run_dtypes(cols))

    def run_id_list(self, dataset_id: str, n: int) -> list[int]:
        return run_ids(dataset_id, [self.name] * n, range(n)).to_list()

    def chunks(self, scan: _Scan, ids: list[int], part: str) -> Iterator[pl.DataFrame]:
        """Chunks of the ``trace``, ``positions`` or ``descriptors`` table.

        Each call reads the file again, in the order of the first pass. The
        descriptors of a run come with its first row.
        """
        m = self.mapping
        ev_i, val_i = self.col[m.evaluations], self.col[m.value]
        extra = [
            (c, self.col[c], scan.extra_types[c].kind(int64=True))
            for c in self.extra_cols
        ]
        pos_i = [self.col[c] for c in m.positions]
        d_kinds = [scan.descriptor_types[c].kind() for c in self.descriptor_cols]
        index: dict[tuple[str, ...], int] = {}
        counter = [0] * len(ids)
        run_col: list[int] = []
        row_col: list[int] = []
        t_ev: list[int] = []
        t_val: list[float] = []
        t_extra: list[list[Any]] = [[] for _ in extra]
        p_cols: list[list[float]] = [[] for _ in pos_i]
        records: list[tuple[int, str, Any, str]] = []
        next_run = 0
        for line, row in self._rows():
            where = f"{self.name}:{line}"
            r = next_run
            if m.run_by:
                key = self._run_key(row)
                r = index.setdefault(key, next_run)
            is_new = r == next_run
            next_run += is_new
            if part == "descriptors":
                if is_new:
                    records.extend(
                        self._records(ids[r], row, d_kinds, where, scan.runs[r])
                    )
                if len(records) >= CHUNK_ROWS:
                    yield descriptor_frame(records)
                    records = []
                continue
            run_col.append(ids[r])
            row_col.append(counter[r])
            counter[r] += 1
            if part == "trace":
                t_ev.append(int(row[ev_i]))
                t_val.append(float(row[val_i]))
                for out, (_c, i, kind) in zip(t_extra, extra, strict=True):
                    out.append(None if row[i] == "" else _PARSERS[kind](row[i], where))
            else:
                for out, i in zip(p_cols, pos_i, strict=True):
                    out.append(float(row[i]))
            if len(run_col) >= CHUNK_ROWS:
                yield self._frame(
                    part, run_col, row_col, t_ev, t_val, extra, t_extra, p_cols
                )
                run_col, row_col, t_ev, t_val = [], [], [], []
                t_extra = [[] for _ in extra]
                p_cols = [[] for _ in pos_i]
        if part == "descriptors":
            yield descriptor_frame(records)
        elif run_col:
            yield self._frame(
                part, run_col, row_col, t_ev, t_val, extra, t_extra, p_cols
            )

    def _records(
        self,
        run_id: int,
        row: list[str],
        kinds: list[str],
        where: str,
        run: _Run,
    ) -> list[tuple[int, str, Any, str]]:
        values = run.values
        out: list[tuple[int, str, Any, str]] = [
            (run_id, "algorithm_name", values["algorithm_name"], "algorithm")
        ]
        if values.get("algorithm_info"):
            out.append(
                (run_id, "algorithm_info", values["algorithm_info"], "algorithm")
            )
        for c, kind in zip(self.descriptor_cols, kinds, strict=True):
            tok = row[self.col[c]]
            if tok != "":
                value = _PARSERS[kind](tok, f"{where}: column {c!r}")
                out.append((run_id, self.descriptor_keys[c], value, self.origin))
        return out

    def _frame(
        self,
        part: str,
        run: list[int],
        row: list[int],
        evals: list[int],
        val: list[float],
        extra: list[tuple[str, int, str]],
        extra_values: list[list[Any]],
        positions: list[list[float]],
    ) -> pl.DataFrame:
        data: dict[str, Any] = {
            "run_id": pl.Series(run, dtype=pl.Int64),
            "row": pl.Series(row, dtype=pl.Int64),
        }
        if part == "positions":
            for j, values in enumerate(positions):
                data[f"x{j}"] = pl.Series(values, dtype=pl.Float64)
            return pl.DataFrame(data)
        data["evaluations"] = pl.Series(evals, dtype=pl.Int64)
        data["raw_y"] = pl.Series(val, dtype=pl.Float64)
        for (c, _i, kind), values in zip(extra, extra_values, strict=True):
            data[self.extra_names[c]] = pl.Series(values, dtype=_DTYPES[kind])
        return pl.DataFrame(data)


def _run_dtypes(cols: Mapping[str, Any]) -> dict[str, pl.DataType]:
    dtypes = {
        "function_id": pl.Int32(),
        "function_name": pl.String(),
        "instance": pl.Int32(),
        "dimension": pl.Int32(),
        "maximization": pl.Boolean(),
        "optimum": pl.Float64(),
        "budget": pl.Int64(),
        "evals_used": pl.Int64(),
        "suite": pl.String(),
        "source_file": pl.String(),
        "source_run_index": pl.Int64(),
        "source_rows_sha256": pl.String(),
    }
    return {k: v for k, v in dtypes.items() if k in cols}


def _dataset_id_from(name: str) -> str:
    stem = name.split(".")[0] or "dataset"
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    return check_dataset_id(cleaned if cleaned[0].isalnum() else "d" + cleaned)


def convert(
    source: str | Path,
    output: str | Path,
    mapping: CsvMapping | Callable[[list[str]], CsvMapping],
    *,
    reader_name: str,
    kind: str,
    origin: str,
    dataset_id: str | None = None,
    source_name: str | None = None,
    overwrite: bool = False,
    reader_options: Mapping[str, Any] | None = None,
    delimiter: str = ",",
    extra: Callable[[CsvConverter], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert the CSV file ``source`` to a dataset in ``output``; returns its manifest.

    ``mapping`` is a ``CsvMapping``, or a function that makes one from the file's
    column names. ``source_name`` is the source path stored in the dataset (default:
    the file name).
    """
    path = Path(source)
    name = check_source_path(source_name or path.name)
    dataset_id = check_dataset_id(dataset_id) if dataset_id else _dataset_id_from(name)
    header = read_header(path, delimiter)
    if callable(mapping):
        mapping = mapping(header)
    conv = CsvConverter(path, name, mapping, header, origin)
    scan = conv.scan()
    runs = conv.runs_frame(scan)
    ids = conv.run_id_list(dataset_id, len(scan.runs))

    source_entry = file_entry(path, name, kind=kind, data_rows=scan.rows)
    fields: dict[str, Any] = {"reader_options": dict(reader_options or {})}
    if conv.renamed:
        fields["renamed_columns"] = conv.renamed
    info = dict(extra(conv)) if extra else {}
    if conv.function_ids is not None:
        info["function_ids"] = conv.function_ids
    if info:
        fields["extra"] = info
    manifest = new_manifest(dataset_id, reader_name, sources=[source_entry], **fields)
    return write_dataset(
        output,
        manifest,
        runs,
        conv.chunks(scan, ids, "trace"),
        positions=conv.chunks(scan, ids, "positions") if mapping.positions else None,
        descriptors=conv.chunks(scan, ids, "descriptors"),
        overwrite=overwrite,
        extra_columns={
            conv.extra_names[c]: _DTYPES[scan.extra_types[c].kind(int64=True)]
            for c in conv.extra_cols
        },
    )
