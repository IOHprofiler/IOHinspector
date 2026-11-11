"""Parts shared by the readers in ``iohinspector.io``.

A reader turns source files into the tables of dataset format v1
(``docs/format.md``) and returns them as a ``ReadResult``, which ``write`` stores
with ``iohinspector.format.write_dataset``. A reader keeps the trace of each
source file in a temporary file (``Tables``), so memory holds one source file at
a time, and the writer streams them.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from ..format import check_source_path, descriptor_frame, new_manifest, write_dataset
from ..format.schema import RUNS_SCHEMA, descriptor_key_problem, is_reserved_trace_name

#: A number token: decimal, optionally with an exponent, or inf / nan in any
#: letter case, each with an optional sign (``docs/format.md``, "Value rules").
NUMBER_PATTERN = (
    r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    r"|[+-]?(?:[iI][nN][fF]|[nN][aA][nN])"
)
INTEGER_PATTERN = r"[+-]?[0-9]+"
NUMBER = re.compile(f"(?:{NUMBER_PATTERN})")
INTEGER = re.compile(INTEGER_PATTERN)


class ReaderError(ValueError):
    """Malformed input. The message starts with the file name and, when known, the line."""

    def __init__(
        self,
        message: str,
        path: str | os.PathLike[str] | None = None,
        line: int | None = None,
    ):
        self.path = None if path is None else str(path)
        self.line = line
        where = ""
        if self.path is not None:
            where = f"{self.path}:{line}: " if line is not None else f"{self.path}: "
        super().__init__(where + message)


@dataclass
class ReadResult:
    """The tables a reader produced, ready for ``write_dataset``.

    ``lazy_trace`` and ``lazy_positions`` hold ``run_id`` and ``row``; the writer
    adds ``function_id`` and ``dimension`` and sorts every table. They scan the
    temporary files in ``files``, which live as long as this object.
    """

    manifest: dict[str, Any]
    runs: pl.DataFrame
    descriptors: pl.DataFrame
    lazy_trace: pl.LazyFrame
    lazy_positions: pl.LazyFrame | None = None
    files: Any = field(default=None, repr=False)

    @property
    def trace(self) -> pl.DataFrame:
        return self.lazy_trace.collect()

    @property
    def positions(self) -> pl.DataFrame | None:
        return None if self.lazy_positions is None else self.lazy_positions.collect()

    def write(self, path: str | os.PathLike[str], **kwargs: Any) -> dict[str, Any]:
        """Write the dataset to the directory ``path``; returns the manifest written."""
        return write_dataset(
            path,
            self.manifest,
            self.runs,
            self.lazy_trace,
            positions=self.lazy_positions,
            descriptors=self.descriptors,
            **kwargs,
        )


@dataclass(frozen=True)
class Reader:
    """One entry of the reader table.

    ``detect(path)`` says whether the reader recognises an input (None: never
    detected, the format must be named), and ``read(path, dataset_id, **options)``
    returns a ``ReadResult``.
    """

    name: str
    description: str
    detect: Callable[[Path], bool] | None
    read: Callable[..., ReadResult]


class Renamer:
    """Renames source names that the format reserves or that are already taken.

    A clash gets ``<name>_<suffix>``, then ``<name>_<suffix>2`` and so on. Every
    rename is recorded once per (source, table) for ``manifest.renamed_columns``.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, str]] = []

    def descriptor_key(
        self, key: str, taken: Iterable[str], suffix: str, source: str
    ) -> str:
        return self.rename(
            key,
            set(taken),
            (suffix, source, "descriptors"),
            lambda k: descriptor_key_problem(k) is not None,
        )

    def trace_column(self, name: str, taken: Iterable[str], source: str) -> str:
        return self.rename(
            name, set(taken), ("logged", source, "trace"), is_reserved_trace_name
        )

    def rename(
        self,
        name: str,
        taken: set[str],
        where: tuple[str, str, str],
        reserved: Callable[[str], bool],
    ) -> str:
        """``name``, or its first free ``<name>_<suffix>[n]``; ``where`` is
        ``(suffix, source, table)``."""
        if not reserved(name) and name not in taken:
            return name
        suffix, source, table = where
        n = 1
        new = f"{name}_{suffix}"
        while reserved(new) or new in taken:
            n += 1
            new = f"{name}_{suffix}{n}"
        record = {"from": name, "to": new, "source": source, "table": table}
        if record not in self.records:
            self.records.append(record)
        return new


def _scratch() -> Any:  # pylint: disable=consider-using-with
    """A temporary folder, removed when the object that holds it is collected."""
    return tempfile.TemporaryDirectory(
        prefix="iohinspector-", ignore_cleanup_errors=True
    )


@dataclass
class Tables:  # pylint: disable=too-many-instance-attributes
    """What a reader collects while it reads its files one by one.

    Run rows and descriptors stay in memory; each trace or positions frame is
    written to a temporary Arrow file at once.
    """

    dataset_id: str
    runs: list[dict[str, Any]] = field(default_factory=list)
    descriptors: list[tuple[int, str, Any, str]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    renamer: Renamer = field(default_factory=Renamer)
    files: Any = field(default_factory=_scratch)
    parts: dict[str, list[Path]] = field(
        default_factory=lambda: {"trace": [], "positions": []}
    )

    def warn(self, code: str, message: str, source: str | None = None) -> None:
        entry = {"code": code, "message": message}
        if source is not None:
            entry["source"] = source
        self.warnings.append(entry)

    def add(self, table: str, frame: pl.DataFrame) -> None:
        """Keep a ``trace`` or ``positions`` frame (with ``run_id`` and ``row``)."""
        parts = self.parts[table]
        path = Path(self.files.name) / f"{table}{len(parts)}.arrow"
        frame.write_ipc(path)
        parts.append(path)

    def result(self, reader_name: str, **manifest_fields: Any) -> ReadResult:
        manifest = new_manifest(
            self.dataset_id,
            reader_name,
            sources=self.sources,
            warnings=self.warnings,
            renamed_columns=self.renamer.records,
            **manifest_fields,
        )
        trace = self._frame("trace")
        return ReadResult(
            manifest=manifest,
            runs=pl.DataFrame(self.runs, schema=RUNS_SCHEMA),
            descriptors=descriptor_frame(self.descriptors),
            lazy_trace=pl.LazyFrame(schema=_EMPTY_TRACE) if trace is None else trace,
            lazy_positions=self._frame("positions"),
            files=self.files,
        )

    def _frame(self, table: str) -> pl.LazyFrame | None:
        parts = [pl.scan_ipc(p) for p in self.parts[table]]
        return pl.concat(parts, how="diagonal") if parts else None


_EMPTY_TRACE = {
    "run_id": pl.Int64,
    "row": pl.Int64,
    "evaluations": pl.Int64,
    "raw_y": pl.Float64,
}


def input_root(
    source: str | os.PathLike[str], root: str | os.PathLike[str] | None
) -> tuple[Path, Path]:
    """The input and the folder its source paths are relative to: ``root``, or
    the input itself when it is a folder, else its folder."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")
    if root:
        return source, Path(root)
    return source, source if source.is_dir() else source.parent


def source_path(file: Path, root: Path) -> str:
    """The path of ``file`` relative to ``root``, in the syntax of ``docs/format.md``."""
    try:
        rel = file.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ReaderError(f"is outside the input folder {root}", file) from exc
    try:
        return check_source_path(rel.as_posix())
    except ValueError as exc:
        raise ReaderError(str(exc), file) from exc


def read_text(path: Path, display: str) -> str:
    """The file as UTF-8 text; ``ReaderError`` when it is not UTF-8."""
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReaderError(f"not UTF-8: {exc}", display) from exc


def first_line(path: Path) -> str:
    """The first line of a file that is not blank, or "" (also when unreadable)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return next((line for line in fh if line.strip()), "")
    except OSError:
        return ""


# --------------------------------------------------------------------------- data lines


@dataclass
class DataLines:
    """The data lines of a text file, split into runs at header lines.

    Per run: the line number and text of its header, its number of data lines,
    and the sha256 of those lines (tokens joined by one space, one ``\\n`` each;
    the ``source_rows_sha256`` rule). ``frame`` has one row per data line:
    ``run_index``, ``row`` (within the run), ``line`` (number), ``width`` (tokens)
    and ``text`` (the tokens joined by one space).
    """

    headers: list[tuple[int, str]]
    rows: list[int]
    sha256: list[str]
    frame: pl.DataFrame

    def tokens(self, width: int, frame: pl.DataFrame | None = None) -> list[pl.Series]:
        """The tokens of each line of ``frame`` (default: all), one string series
        per position."""
        text = (self.frame if frame is None else frame)["text"]
        names = [f"c{i}" for i in range(width)]
        if text.is_empty():
            return [pl.Series(n, [], dtype=pl.String) for n in names]
        split = text.str.split_exact(" ", width - 1).struct.rename_fields(names)
        return split.struct.unnest().get_columns()


def split_lines(
    text: str, display: str, kind: Callable[[str, list[str]], bool | None]
) -> DataLines:
    """Split a whitespace-separated data file into runs.

    ``kind(line, tokens)`` is True for a header line (it starts a run), None for a
    comment and False for a data line. Blank lines are skipped.
    """
    headers: list[tuple[int, str]] = []
    digests: list[Any] = []
    rows: list[int] = []
    cols: dict[str, list[int]] = {k: [] for k in ("run_index", "row", "line", "width")}
    lines: list[str] = []
    for number, raw in enumerate(text.split("\n"), 1):
        tokens = raw.split()
        if not tokens:
            continue
        header = kind(raw, tokens)
        if header:
            headers.append((number, raw))
            digests.append(hashlib.sha256())
            rows.append(0)
        elif header is not None:
            if not headers:
                raise ReaderError(
                    "data line before the first header line", display, number
                )
            line = " ".join(tokens)
            digests[-1].update(line.encode("utf-8") + b"\n")
            lines.append(line)
            for key, value in zip(cols, (len(rows) - 1, rows[-1], number, len(tokens))):
                cols[key].append(value)
            rows[-1] += 1
    return DataLines(
        headers,
        rows,
        [d.hexdigest() for d in digests],
        pl.DataFrame(
            {k: pl.Series(v, dtype=pl.Int64) for k, v in cols.items()}
            | {"text": pl.Series(lines, dtype=pl.String)}
        ),
    )


# --------------------------------------------------------------------------- tokens


def first_false(mask: pl.Series) -> int | None:
    """Index of the first false (or null) value of a boolean series, or None."""
    bad = (~mask.fill_null(False)).arg_true()
    return None if bad.is_empty() else int(bad[0])


def is_number(tokens: pl.Series) -> pl.Series:
    """Whether each token is a number of the format's value rules."""
    return tokens.str.contains(f"^(?:{NUMBER_PATTERN})$")


def check_floats(
    tokens: pl.Series, lines: Sequence[int], path: str, column: str
) -> pl.Series:
    """float64 values of number tokens; fails with file:line at the first other token."""
    i = first_false(is_number(tokens))
    if i is not None:
        raise ReaderError(
            f"column {column!r}: {tokens[i]!r} is not a number", path, lines[i]
        )
    return tokens.str.to_lowercase().cast(pl.Float64).alias(column)


def check_integers(
    tokens: pl.Series, lines: Sequence[int], path: str, column: str
) -> pl.Series:
    """int64 values of integer tokens; fails with file:line at the first other token."""
    i = first_false(tokens.str.contains(f"^(?:{INTEGER_PATTERN})$"))
    if i is not None:
        raise ReaderError(
            f"column {column!r}: {tokens[i]!r} is not an integer", path, lines[i]
        )
    values = tokens.cast(pl.Int64, strict=False)
    i = first_false(values.is_not_null())
    if i is not None:
        raise ReaderError(
            f"column {column!r}: {tokens[i]} does not fit in int64", path, lines[i]
        )
    return values.alias(column)
