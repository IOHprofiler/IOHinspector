"""Line scanners for ``verify``: the data lines of each source file kind.

Independent of the readers by design: everything here follows the file formats
as ``docs/format.md`` and the reader pages describe them, not the readers' code.
A scanner splits a file into runs and returns, per run, the hash of its data
lines (the ``source_rows_sha256`` rule) and its evaluations and objective values.

``KINDS`` maps a manifest ``sources[].kind`` to a layout. A kind that is not in
the table, and is not a metadata kind, is reported as not verified.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import polars as pl

#: Number tokens of the format's value rules: decimals, exponents, inf and nan.
_NUMBER = re.compile(
    r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[+-]?(?:inf|nan)",
    re.IGNORECASE,
)
_INTEGER = re.compile(r"[+-]?[0-9]+")
#: COCO column headers; other ``%`` lines are comments.
_COCO_HEADERS = ("f evaluations", "function evaluation")


@dataclass(frozen=True)
class TextLayout:
    """A whitespace-separated data file whose runs start at header lines.

    ``evaluations`` and ``value`` are column positions, or names looked up in the
    header. With ``coco``, only ``%`` lines that name the columns start a run and
    other ``%`` lines are comments; otherwise every line whose first token is not
    a number starts a run. ``family`` files (COCO's) log the runs of one
    run set in several files with one stem, merged in the trace (``logged_in``).
    """

    evaluations: int | str
    value: int | str
    coco: bool = False
    family: bool = False


@dataclass(frozen=True)
class CsvLayout:
    """A CSV file: one row per evaluation, runs by the ``run_by`` columns
    (every row its own run when empty)."""

    evaluations: str
    value: str
    run_by: tuple[str, ...] = ()
    delimiter: str = ","
    family: bool = False


Layout = TextLayout | CsvLayout

_COCO = TextLayout(0, 2, coco=True, family=True)
KINDS: dict[str, Layout] = {
    "ioh-dat": TextLayout("evaluations", "raw_y"),
    "coco-dat": _COCO,
    "coco-tdat": _COCO,
    "coco-rdat": _COCO,
    "nevergrad-csv": CsvLayout("budget", "loss"),
}
#: Kinds whose files hold metadata only: their sha256 is checked, nothing else.
METADATA_KINDS = ("ioh-json", "coco-info")


def custom_csv_layout(manifest: Mapping[str, Any]) -> CsvLayout | None:
    """The layout of a custom CSV source, from the mapping in the manifest."""
    mapping = (manifest.get("reader", {}).get("options") or {}).get("mapping")
    if not isinstance(mapping, Mapping):
        return None
    try:
        return CsvLayout(
            str(mapping["evaluations"]),
            str(mapping["value"]),
            tuple(mapping.get("run_by") or ()),
            str(mapping.get("delimiter") or ","),
        )
    except KeyError:
        return None


class ScanError(ValueError):
    """A file the scanner cannot read; the message names the line."""


@dataclass
class FileScan:
    """The runs of one data file: per run its line hash and row count, and
    ``rows`` with ``run_index``, ``evaluations`` and ``y`` in file order."""

    path: str
    hashes: list[str] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    rows: pl.DataFrame | None = None


_ROWS = {"run_index": pl.Int64, "evaluations": pl.Int64, "y": pl.Float64}
#: Rows move to polars in chunks: small allocations of one size, which the
#: allocator reuses from file to file. Large ones of every file's own size stay
#: in the resident memory of some (macOS), so the peak would grow with the input.
_CHUNK_ROWS = 1 << 16


@dataclass
class _Collector:
    path: str
    digests: list[Any] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    run_index: list[int] = field(default_factory=list)
    evaluations: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    chunks: list[pl.DataFrame] = field(default_factory=list)

    def new_run(self) -> int:
        self.digests.append(hashlib.sha256())
        self.counts.append(0)
        return len(self.counts) - 1

    def add(self, run: int, tokens: list[str], ev: str, y: str, line: int) -> None:
        if _INTEGER.fullmatch(ev) is None:
            raise ScanError(f"{self.path}:{line}: evaluations {ev!r} is not an integer")
        if _NUMBER.fullmatch(y) is None:
            raise ScanError(f"{self.path}:{line}: value {y!r} is not a number")
        self.digests[run].update((" ".join(tokens) + "\n").encode("utf-8"))
        self.counts[run] += 1
        self.run_index.append(run)
        self.evaluations.append(int(ev))
        self.values.append(float(y))
        if len(self.values) == _CHUNK_ROWS:
            self.flush()

    def flush(self) -> None:
        columns = (self.run_index, self.evaluations, self.values)
        self.chunks.append(pl.DataFrame(dict(zip(_ROWS, columns)), schema=_ROWS))
        for column in columns:
            column.clear()

    def result(self) -> FileScan:
        self.flush()
        return FileScan(
            self.path,
            [d.hexdigest() for d in self.digests],
            self.counts,
            pl.concat(self.chunks),
        )


def _gunzip(data: bytes, path: str) -> bytes:
    if not path.lower().endswith(".gz"):
        return data
    try:
        return gzip.decompress(data)
    except (OSError, EOFError) as exc:
        raise ScanError(f"{path}: not gzip data ({exc})") from exc


def _decode(data: bytes, where: str, bom: bool = False) -> str:
    try:
        return data.decode("utf-8-sig" if bom else "utf-8")
    except UnicodeDecodeError as exc:
        raise ScanError(f"{where}: not UTF-8 ({exc})") from exc


def _column(spec: int | str, header: list[str], path: str, line: int) -> int:
    if isinstance(spec, int):
        return spec
    if spec not in header:
        raise ScanError(f"{path}:{line}: the header has no column {spec!r}")
    return header.index(spec)


def _starts_run(tokens: list[str], text: str, coco: bool) -> bool | None:
    """True for a header line, None for a comment, False for a data line."""
    if coco:
        stripped = text.lstrip()
        if not stripped.startswith("%"):
            return False
        heading = stripped[1:].strip().lower()
        return True if heading.startswith(_COCO_HEADERS) else None
    return _NUMBER.fullmatch(tokens[0]) is None


def scan_text(data: bytes, path: str, layout: TextLayout) -> FileScan:
    """Split a whitespace-separated data file into runs at its header lines."""
    out = _Collector(path)
    columns: tuple[int, int] | None = None
    # line by line: no copy of the whole file as text or as a list of lines
    for number, line in enumerate(io.BytesIO(_gunzip(data, path)), 1):
        text = _decode(line, f"{path}:{number}")
        tokens = text.split()
        if not tokens:
            continue
        kind = _starts_run(tokens, text, layout.coco)
        if kind is None:
            continue
        if kind:
            out.new_run()
            columns = (
                _column(layout.evaluations, tokens, path, number),
                _column(layout.value, tokens, path, number),
            )
            continue
        if columns is None:
            raise ScanError(f"{path}:{number}: data line before the first header")
        if max(columns) >= len(tokens):
            raise ScanError(f"{path}:{number}: only {len(tokens)} fields")
        out.add(
            len(out.counts) - 1, tokens, tokens[columns[0]], tokens[columns[1]], number
        )
    return out.result()


def scan_csv(data: bytes, path: str, layout: CsvLayout) -> FileScan:
    """Read a CSV file with the ``csv`` module; runs by ``layout.run_by``."""
    reader = csv.reader(
        io.StringIO(_decode(_gunzip(data, path), path, bom=True), newline=""),
        delimiter=layout.delimiter,
    )
    header = next(reader, None) or []
    try:
        ev, y = header.index(layout.evaluations), header.index(layout.value)
        keys = [header.index(c) for c in layout.run_by]
    except ValueError as exc:
        raise ScanError(f"{path}:1: {exc}") from exc
    out = _Collector(path)
    runs: dict[tuple[str, ...], int] = {}
    try:
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ScanError(f"{path}:{reader.line_num}: {len(row)} fields")
            key = tuple(row[i] for i in keys)
            run = runs.get(key) if keys else None
            if run is None:
                run = out.new_run()
                runs[key] = run
            out.add(run, row, row[ev], row[y], reader.line_num)
    except csv.Error as exc:
        raise ScanError(f"{path}: {exc}") from exc
    return out.result()


def scan(data: bytes, path: str, layout: Layout) -> FileScan:
    if isinstance(layout, CsvLayout):
        return scan_csv(data, path, layout)
    return scan_text(data, path, layout)
