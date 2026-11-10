"""Parser of legacy IOHprofiler data files: ``.dat``, ``.cdat``, ``.idat``, ``.tdat``.

Each run starts with a header line of column names, in double quotes when they
hold spaces::

    "function evaluation" "current f(x)" "best-so-far f(x)" "current af(x)+b" ...

(one line in the file), followed by one line per logged evaluation, values
separated by whitespace. A header is a line whose first token is not a number
(IOHanalyzer's ``read_dat`` uses the same test). Runs are split on header lines
only.

The first column is the evaluation count and the second the objective value as
logged (``raw_y``). Columns ``x0`` ... ``x(d-1)``, when all are present, are the
search-space position. Every other column is kept as an extra trace column with its
source name. The two-column variant (IOHanalyzer's ``TWO_COL``) needs nothing
special.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from ..format import is_reserved_trace_name
from .base import INTEGER_PATTERN, NUMBER_PATTERN, ReaderError

_NAME_TOKEN = r"\"([^\"]*)\"|(\S+)"
_IS_NUMBER = f"^(?:{NUMBER_PATTERN})$"
_IS_INTEGER = f"^(?:{INTEGER_PATTERN})$"


@dataclass
class DataFile:  # pylint: disable=too-many-instance-attributes
    """One parsed data file.

    ``trace`` has ``run_index``, ``file_row`` (the data row's position in the
    file), ``line``, ``evaluations``, ``raw_y``, the extra columns and the position
    columns ``x0`` ...; extra columns are float64.
    """

    kind: str
    path: str
    trace: pl.DataFrame
    run_hashes: list[str]
    precision: dict[str, Any]
    extra_columns: list[str] = field(default_factory=list)
    has_positions: bool = False
    #: (source name, stored name) of extra columns renamed because they are reserved.
    renamed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.run_hashes)


def header_names(text: str) -> list[str]:
    """Column names of a header line; quoted names may hold spaces."""
    return [q if q else bare for q, bare in re.findall(_NAME_TOKEN, text)]


def _read_lines(local: Path, path: str) -> pl.DataFrame:
    try:
        text = local.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReaderError(
            f"not UTF-8 text ({exc.reason} at byte {exc.start})", path
        ) from exc
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    df = pl.DataFrame({"text": lines}, schema={"text": pl.String})
    df = df.with_row_index("line", offset=1).with_columns(
        pl.col("line").cast(pl.Int64),
        norm=pl.col("text").str.strip_chars().str.replace_all(r"\s+", " "),
    )
    df = df.filter(pl.col("norm") != "")
    first = pl.col("norm").str.extract(r"^(\S+)")
    return df.with_columns(header=~first.str.contains(_IS_NUMBER)).with_columns(
        section=pl.col("header").cast(pl.Int64).cum_sum()
    )


def _headers(df: pl.DataFrame, path: str) -> list[list[str]]:
    """The column names of each run, checked; index i is run i."""
    headers: list[list[str]] = []
    for line, text in df.filter("header").select("line", "text").iter_rows():
        names = header_names(text)
        if len(names) < 2:
            raise ReaderError(
                "a header needs at least two columns (evaluations and f(x))", path, line
            )
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ReaderError(f"the header names {dupes} more than once", path, line)
        headers.append(names)
    return headers


def _check_shape(data: pl.DataFrame, headers: list[list[str]], path: str) -> None:
    before = data.filter(pl.col("section") == 0)
    if before.height:
        raise ReaderError(
            "data line before the first header line", path, int(before["line"][0])
        )
    widths = {i + 1: len(h) for i, h in enumerate(headers)}
    want = data["section"].replace_strict(widths, return_dtype=pl.Int64)
    got = data["tokens"].list.len().cast(pl.Int64)
    bad = (got != want).arg_true()
    if len(bad):
        i = int(bad[0])
        raise ReaderError(
            f"{got[i]} values, but the header has {want[i]} columns",
            path,
            int(data["line"][i]),
        )


def _parse_column(
    tokens: pl.Series, lines: pl.Series, path: str, name: str, integer: bool
) -> pl.Series:
    pattern = _IS_INTEGER if integer else _IS_NUMBER
    bad = (~tokens.str.contains(pattern)).arg_true()
    if len(bad):
        i = int(bad[0])
        what = "an integer" if integer else "a number"
        raise ReaderError(
            f"column {name!r}: {tokens[i]!r} is not {what}", path, int(lines[i])
        )
    if integer:
        values = tokens.cast(pl.Int64, strict=False)
        bad = values.is_null().arg_true()
        if len(bad):
            i = int(bad[0])
            raise ReaderError(
                f"column {name!r}: {tokens[i]} does not fit in int64",
                path,
                int(lines[i]),
            )
        return values
    return tokens.str.to_lowercase().cast(pl.Float64)


def _stored_names(names: list[str], dimension: int) -> tuple[list[str], bool]:
    """Stored names of a header's columns, and whether it holds the positions."""
    positions = [f"x{i}" for i in range(dimension)]
    has_positions = all(p in names[2:] for p in positions)
    out = ["evaluations", "raw_y"]
    taken = set(out)
    for name in names[2:]:
        if has_positions and name in positions:
            out.append(name)
            continue
        new, n = name, 1
        while is_reserved_trace_name(new) or new in taken:
            new = f"{name}_logged" if n == 1 else f"{name}_logged{n}"
            n += 1
        taken.add(new)
        out.append(new)
    return out, has_positions


def _section_frame(
    data: pl.DataFrame, names: list[str], stored: list[str], path: str
) -> pl.DataFrame:
    cols: dict[str, pl.Series] = {}
    for i, (name, new) in enumerate(zip(names, stored, strict=True)):
        tokens = data["tokens"].list.get(i)
        cols[new] = _parse_column(tokens, data["line"], path, name, integer=i == 0)
    return pl.DataFrame(cols).with_columns(
        run_index=(data["section"] - 1), file_row=data["file_row"], line=data["line"]
    )


def _run_hashes(data: pl.DataFrame, n_runs: int) -> list[str]:
    joined = data.group_by("section").agg(pl.col("norm").str.join("\n"))
    by_run = dict(joined.iter_rows())
    out = []
    for section in range(1, n_runs + 1):
        text = by_run.get(section)
        payload = b"" if text is None else (text + "\n").encode("utf-8")
        out.append(hashlib.sha256(payload).hexdigest())
    return out


def precision_of(tokens: pl.Series) -> dict[str, Any]:
    """The manifest ``precision`` of number tokens: fixed or scientific, k digits."""
    finite = tokens.filter(~tokens.str.contains(r"(?i)inf|nan"))
    if finite.is_empty():
        return {"notation": "unknown"}
    fixed = finite.str.extract(r"^[+-]?[0-9]+\.([0-9]+)$").str.len_chars()
    sci = finite.str.extract(r"^[+-]?[0-9]\.([0-9]+)[eE][+-]?[0-9]+$").str.len_chars()
    for notation, digits in (("fixed", fixed), ("scientific", sci)):
        if digits.null_count() == 0 and digits.n_unique() == 1:
            return {"notation": notation, "digits": int(digits[0])}
    return {"notation": "unknown"}


def parse_data_file(local: Path, path: str, kind: str, dimension: int) -> DataFile:
    """Parse one data file; ``path`` is its source path, used in errors."""
    df = _read_lines(local, path)
    headers = _headers(df, path)
    data = df.filter(~pl.col("header")).with_columns(
        tokens=pl.col("norm").str.split(" ")
    )
    data = data.with_row_index("file_row").with_columns(
        pl.col("file_row").cast(pl.Int64)
    )
    _check_shape(data, headers, path)
    result = DataFile(kind, path, pl.DataFrame(), _run_hashes(data, len(headers)), {})
    frames = []
    for names in {tuple(h): None for h in headers}:
        sections = [i + 1 for i, h in enumerate(headers) if tuple(h) == names]
        part = data.filter(pl.col("section").is_in(sections))
        stored, positions = _stored_names(list(names), dimension)
        result.has_positions |= positions
        for old, new in zip(names[2:], stored[2:], strict=True):
            if old != new and (old, new) not in result.renamed:
                result.renamed.append((old, new))
        if part.height:
            frames.append(_section_frame(part, list(names), stored, path))
    result.trace = _combine(frames)
    fixed = {"run_index", "file_row", "line", "evaluations", "raw_y"}
    result.extra_columns = [
        c
        for c in result.trace.columns
        if c not in fixed and not _is_position(c, dimension)
    ]
    result.precision = precision_of(
        data["tokens"].list.get(1) if data.height else pl.Series([], dtype=pl.String)
    )
    return result


def _is_position(name: str, dimension: int) -> bool:
    return name in {f"x{i}" for i in range(dimension)}


def _combine(frames: list[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame(
            schema={
                "evaluations": pl.Int64,
                "raw_y": pl.Float64,
                "run_index": pl.Int64,
                "file_row": pl.Int64,
                "line": pl.Int64,
            }
        )
    return pl.concat(frames, how="diagonal").sort("file_row")
