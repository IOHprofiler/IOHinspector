"""COCO data files: ``.dat``, ``.tdat`` and ``.rdat`` of the single-objective loggers.

A data file holds the runs of one function and dimension one after the other.
Each run starts with a header line; then come data lines of five columns, plus
the position ``x1 ... xD`` when the logger wrote it (not above 20-D with recent
loggers, and never for some files)::

    % f evaluations | g evaluations | best noise-free fitness - Fopt (7.948000000000e+01) + sum g_i+ | measured fitness | best measured fitness or single-digit g-values | x1 | x2...
    1 0 +5.826987706e-01 +8.006269877e+01 +8.006269877e+01 -3.2755e-01 -6.6092e-01

Loggers before COCO 2 (the ``bbob`` code of 2009 to 2018) write::

    % function evaluation | noise-free fitness - Fopt (7.948000000000e+01) | best noise-free fitness - Fopt | measured fitness | best measured fitness | x1 | x2...

The Fopt in the header is the run's optimal value. Other lines starting with
``%`` are comments, such as ``% restart: independent restart`` in ``.rdat``
files; they neither start a run nor hold data.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from .base import NUMBER, ReaderError, check_floats, check_integers

#: Column names of the two header styles; ``raw_y`` is the best Δf.
NEW_COLUMNS = ("evaluations", "g_evaluations", "raw_y", "measured_f", "best_measured_f")
OLD_COLUMNS = (
    "evaluations",
    "current_delta_f",
    "raw_y",
    "measured_f",
    "best_measured_f",
)
INTEGER_COLUMNS = ("evaluations", "g_evaluations")
_FOPT = re.compile(r"Fopt\s*\(\s*([^)\s]*)\s*\)")


def header_style(line: str) -> str | None:
    """``new`` or ``old`` for a column header line, None for other ``%`` lines."""
    text = line.lstrip()
    if not text.startswith("%"):
        return None
    text = text[1:].strip().lower()
    if text.startswith("f evaluations"):
        return "new"
    if text.startswith("function evaluation"):
        return "old"
    return None


@dataclass
class DatRun:
    header_line: int
    #: The optimal value from the header, None when the header has none.
    fopt: float | None
    rows: int = 0
    sha256: str = ""


@dataclass
class CocoDat:
    """A parsed data file. ``frame`` has ``run_index``, ``file_row``, the value
    columns of its header style and ``x0`` ... where positions were written."""

    path: str
    kind: str
    style: str | None
    runs: list[DatRun]
    frame: pl.DataFrame
    comments: int = 0


@dataclass
class _Lines:
    text: list[str] = field(default_factory=list)
    numbers: list[int] = field(default_factory=list)
    run_index: list[int] = field(default_factory=list)
    width: list[int] = field(default_factory=list)


def _decode(path: Path, display: str) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReaderError(f"not UTF-8: {exc}", display) from exc


def _fopt(line: str, display: str, number: int) -> float | None:
    m = _FOPT.search(line)
    if m is None:
        return None
    if NUMBER.fullmatch(m[1]) is None:
        raise ReaderError(f"Fopt {m[1]!r} is not a number", display, number)
    return float(m[1].lower())


def _split(  # pylint: disable=too-many-locals
    text: str, display: str, dimension: int
) -> tuple[str | None, list[DatRun], _Lines, int]:
    # #lizard forgives  (one pass over the lines; the cases are the line kinds)
    style: str | None = None
    runs: list[DatRun] = []
    hashes: list = []
    lines = _Lines()
    comments = 0
    widths = (5, 5 + dimension)
    for number, raw in enumerate(text.split("\n"), 1):
        tokens = raw.split()
        if not tokens:
            continue
        if raw.lstrip().startswith("%"):
            this = header_style(raw)
            if this is None:
                comments += 1
                continue
            if style is not None and this != style:
                raise ReaderError(
                    "the header style changes within the file", display, number
                )
            style = this
            runs.append(DatRun(number, _fopt(raw, display, number)))
            hashes.append(hashlib.sha256())
            continue
        if not runs:
            raise ReaderError("data line before the first header line", display, number)
        if len(tokens) not in widths:
            raise ReaderError(
                f"{len(tokens)} fields; a {dimension}-D data line has 5 or {5 + dimension}",
                display,
                number,
            )
        line = " ".join(tokens)
        hashes[-1].update(line.encode("utf-8") + b"\n")
        lines.text.append(line)
        lines.numbers.append(number)
        lines.run_index.append(len(runs) - 1)
        lines.width.append(len(tokens))
        runs[-1].rows += 1
    for run, digest in zip(runs, hashes):
        run.sha256 = digest.hexdigest()
    return style, runs, lines, comments


def _columns(
    lines: _Lines, names: tuple[str, ...], dimension: int, display: str
) -> pl.DataFrame:
    """The typed columns of all lines, in file order."""
    base = pl.DataFrame(
        {
            "run_index": pl.Series(lines.run_index, dtype=pl.Int64),
            "file_row": pl.int_range(len(lines.text), dtype=pl.Int64, eager=True),
            "line": pl.Series(lines.numbers, dtype=pl.Int64),
            "width": pl.Series(lines.width, dtype=pl.Int64),
            "text": pl.Series(lines.text, dtype=pl.String),
        }
    )
    parts = []
    for width in sorted(set(lines.width)):
        part = base.filter(pl.col("width") == width)
        fields = [*names, *(f"x{i}" for i in range(width - len(names)))]
        split = (
            part["text"]
            .str.split_exact(" ", width - 1)
            .struct.rename_fields(fields)
            .struct.unnest()
        )
        numbers = part["line"].to_list()
        typed = [
            (check_integers if name in INTEGER_COLUMNS else check_floats)(
                split[name], numbers, display, name
            )
            for name in fields
        ]
        parts.append(part.select("run_index", "file_row").with_columns(typed))
    if not parts:
        return _empty(names)
    frame = pl.concat(parts, how="diagonal").sort("file_row")
    xs = [f"x{i}" for i in range(dimension) if f"x{i}" in frame.columns]
    return frame.select("run_index", "file_row", *names, *xs)


def _empty(names: tuple[str, ...]) -> pl.DataFrame:
    schema = {"run_index": pl.Int64, "file_row": pl.Int64}
    for name in names:
        schema[name] = pl.Int64 if name in INTEGER_COLUMNS else pl.Float64
    return pl.DataFrame(schema=schema)


def parse_coco_dat(
    path: str | Path, *, display: str | None = None, dimension: int, kind: str = "dat"
) -> CocoDat:
    """Parse a COCO data file of a ``dimension``-D problem.

    Runs are split on header lines. Every data line becomes a row of ``frame``;
    nothing is merged or dropped. Malformed lines raise ``ReaderError`` with
    file:line.
    """
    path = Path(path)
    display = display or str(path)
    style, runs, lines, comments = _split(_decode(path, display), display, dimension)
    names = OLD_COLUMNS if style == "old" else NEW_COLUMNS
    frame = _columns(lines, names, dimension, display)
    return CocoDat(display, kind, style, runs, frame, comments)
