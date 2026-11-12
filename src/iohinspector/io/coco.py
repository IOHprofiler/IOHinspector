"""Reader for COCO single-objective data: ``.info`` files with ``.dat``, ``.tdat``
and ``.rdat`` data files, as in the COCO data archive.

Covers the suites ``bbob``, ``bbob-noisy``, ``bbob-largescale`` and
``bbob-boxed`` and every logger version in the archive (2009 to today). The
input is a folder or an ``.info`` file; one input may hold many algorithms
(``iohinspector.io.read`` opens archives first). ``docs/readers-coco.md``
describes the mapping to format v1 in full.

In short:

- each ``.info`` block names a data file and lists its runs; runs are matched in
  order with the runs of the data file, which start at header lines. Instance
  ids come from the ``.info`` entries;
- ``raw_y`` is the best noise-free Δf = f - Fopt the logger wrote (column 3), the
  value COCO's targets apply to; ``optimum`` is the Fopt of each run's header,
  so raw f is ``raw_y + optimum``. Every other column is kept;
- the ``.dat``, ``.tdat`` and ``.rdat`` files of one stem log the same runs with
  different triggers. Their rows are merged under the identical-row rule
  (``iohinspector.io.merge``): ``logged_in`` says which files hold a row,
  ``merge_conflict`` flags rows the files disagree on. The ``.dat`` file is the
  primary source file of a run.

A data file holds the runs of one function and dimension one after the other.
Each run starts with a header line; then come data lines of five columns (six
with ``bbob-new``, below), plus the position ``x1 ... xD`` when the logger wrote
it::

    % f evaluations | g evaluations | best noise-free fitness - Fopt (7.948e+01)
      + sum g_i+ | measured fitness | best measured fitness or single-digit
      g-values | x1 | x2...

(one line; wrapped here). COCO 2.2.1 (``data_format = 'bbob-new'``) writes
``f evaluations | g evaluations | best noise-free fitness - Fopt | noise-free
fitness - Fopt (F) | measured fitness | best measured fitness | x1 | x2...``; the
header's fourth field decides. Loggers before COCO 2 write ``% function evaluation |
noise-free fitness - Fopt (7.948e+01) | best noise-free fitness - Fopt |
measured fitness | best measured fitness | x1 | x2...``. The Fopt in the header
is the run's optimal value. Other ``%`` lines are comments, such as ``% restart:
independent restart`` in ``.rdat`` files.
"""

from __future__ import annotations

import posixpath
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from ..format import check_source_path, file_entry, run_id
from ._coco_info import ALIASES, InfoBlock, InfoEntry, InfoFile, parse_info
from .base import (
    NUMBER,
    DataLines,
    ReaderError,
    ReadResult,
    Tables,
    check_floats,
    check_integers,
    first_line,
    input_root,
    read_text,
    source_path,
    split_lines,
)
from .merge import merge_files

READER_NAME = "coco"
#: Data file kinds, in merge order; the first that exists is a run's source file.
KINDS = ("dat", "tdat", "rdat")
#: Files COCO loggers write that this reader lists but does not read.
UNREAD_KINDS = ("mdat", "cdat")
TRIGGERS = {
    "dat": [{"kind": "fixed_target"}],
    "tdat": [{"kind": "fixed_budget"}],
    "rdat": [{"kind": "other", "params": {"event": "restart"}}],
}
#: Values are written with ``%+10.9e``, positions with ``%+5.4e``.
PRECISION = {"notation": "scientific", "digits": 9}
#: Header keys stored as runs columns rather than descriptors.
RUN_HEADER_KEYS = ("funcId", "DIM", "algId", "suite", "folder")
FINAL_KEY = "final_delta_f"
#: Column names of the header styles; ``raw_y`` is the best Δf. ``bbob-new``
#: (COCO 2.2.1 ``data_format``) adds the current noise-free Δf after it.
COLUMNS = {
    "bbob-new": (
        "evaluations",
        "g_evaluations",
        "raw_y",
        "current_delta_f",
        "measured_f",
        "best_measured_f",
    ),
    "new": ("evaluations", "g_evaluations", "raw_y", "measured_f", "best_measured_f"),
    "old": ("evaluations", "current_delta_f", "raw_y", "measured_f", "best_measured_f"),
}
INTEGER_COLUMNS = ("evaluations", "g_evaluations")
_FOPT = re.compile(r"Fopt\s*\(\s*([^)\s]*)\s*\)")
_SUMMARIES = {
    "suite_inferred": "run(s) without a suite in the .info header: bbob-noisy "
    "for function ids above 100, else bbob (as cocopp does)",
    "unfinalized_run": "run(s) the logger did not finalise (a bare instance in "
    "the .info file): no evaluation count",
    "missing_algorithm": "run(s) without algId; algorithm_name is empty",
}


# --------------------------------------------------------------------------- data files


def header_style(line: str) -> str | None:
    """``new``, ``bbob-new`` or ``old`` for a column header line, None for other
    ``%`` lines. ``bbob-new`` names a second noise-free column after the best."""
    text = line.lstrip()
    if not text.startswith("%"):
        return None
    text = text[1:].strip().lower()
    if text.startswith("f evaluations"):
        parts = text.split("|")
        return "bbob-new" if parts[3:4] and "noise-free" in parts[3] else "new"
    if text.startswith("function evaluation"):
        return "old"
    return None


@dataclass
class CocoDat:
    """A parsed data file. ``frame`` has ``run_index``, ``file_row``, the value
    columns of its header style and ``x0`` ... where positions were written;
    ``fopt`` is the optimal value in each run's header (None when it has none)."""

    path: str
    kind: str
    style: str | None
    lines: DataLines
    fopt: list[float | None]
    frame: pl.DataFrame


def _is_header(line: str, _tokens: list[str]) -> bool | None:
    if not line.lstrip().startswith("%"):
        return False
    return True if header_style(line) else None


def _fopt(line: str, display: str, number: int) -> float | None:
    m = _FOPT.search(line)
    if m is None:
        return None
    if NUMBER.fullmatch(m[1]) is None:
        raise ReaderError(f"Fopt {m[1]!r} is not a number", display, number)
    return float(m[1].lower())


def parse_coco_dat(
    path: str | Path, *, display: str | None = None, dimension: int, kind: str = "dat"
) -> CocoDat:
    """Parse a COCO data file of a ``dimension``-D problem.

    Runs are split on header lines. Every data line becomes a row of ``frame``;
    nothing is merged or dropped. Malformed lines raise ``ReaderError`` with
    file:line.
    """
    display = display or str(path)
    lines = split_lines(read_text(Path(path), display), display, _is_header)
    styles = [header_style(text) for _, text in lines.headers]
    for (number, _), style in zip(lines.headers, styles):
        if style != styles[0]:
            raise ReaderError(
                "the header style changes within the file", display, number
            )
    style = styles[0] if styles else None
    names = COLUMNS["old" if style == "old" else style or "new"]
    fixed = len(names)
    bad = lines.frame.filter(~pl.col("width").is_in([fixed, fixed + dimension]))
    if not bad.is_empty():
        raise ReaderError(
            f"{bad['width'][0]} fields; a {dimension}-D data line has {fixed} or "
            f"{fixed + dimension}",
            display,
            bad["line"][0],
        )
    fopt = [_fopt(text, display, number) for number, text in lines.headers]
    frame = _columns(lines, names, dimension, display)
    return CocoDat(display, kind, style, lines, fopt, frame)


def _columns(
    lines: DataLines, names: tuple[str, ...], dimension: int, display: str
) -> pl.DataFrame:
    """The typed columns of all lines, in file order."""
    base = lines.frame.with_columns(file_row=pl.int_range(pl.len(), dtype=pl.Int64))
    parts = [
        pl.DataFrame(schema={"run_index": pl.Int64, "file_row": pl.Int64}).with_columns(
            pl.Series(n, [], dtype=_dtype(n)) for n in names
        )
    ]
    fixed = len(names)
    for width in (fixed, fixed + dimension) if dimension else (fixed,):
        part = base.filter(pl.col("width") == width)
        if part.is_empty():
            continue
        fields = [*names, *(f"x{i}" for i in range(width - fixed))]
        numbers = part["line"].to_list()
        typed = [
            (check_integers if name in INTEGER_COLUMNS else check_floats)(
                tokens, numbers, display, name
            )
            for name, tokens in zip(fields, lines.tokens(width, part))
        ]
        parts.append(part.select("run_index", "file_row").with_columns(typed))
    return pl.concat(parts, how="diagonal").sort("file_row")


def _dtype(name: str) -> type[pl.DataType]:
    return pl.Int64 if name in INTEGER_COLUMNS else pl.Float64


# --------------------------------------------------------------------------- detection


def _coco_data(local: Path) -> bool | None:
    """Whether a data file looks like COCO's: its first line is a ``%`` line.
    None when it starts with data, so that the ``.info`` header decides. Legacy
    IOHprofiler data (a quoted ``"function evaluation"`` header) is refused."""
    line = first_line(local).strip()
    if line.startswith('"function evaluation"'):
        raise ReaderError("legacy IOHprofiler format is not supported", local)
    if line.startswith("%"):
        return True
    tokens = line.split()
    return None if not tokens or NUMBER.fullmatch(tokens[0]) else False


def _is_coco(info: InfoFile, folder: Path) -> bool:
    """The first data file found starts with a COCO header line; with no data
    file, the header has a ``Precision``."""
    if not info.blocks:
        return False
    data = folder.joinpath(*info.blocks[0].data_file.split("/"))
    for kind in KINDS:
        local = data.with_suffix("." + kind)
        if local.is_file():
            verdict = _coco_data(local)
            if verdict is not None:
                return verdict
            break
    return info.blocks[0].get("Precision") is not None


def _parse_quietly(text: str) -> InfoFile | None:
    try:
        return parse_info(text, "")
    except ReaderError:
        return None


def is_coco_info(path: Path) -> bool:
    """A COCO ``.info`` file: a header with ``funcId`` and ``DIM`` whose data file
    starts with a ``%`` header line (or, when it is missing, a ``Precision``)."""
    if path.suffix.lower() != ".info" or not path.is_file():
        return False
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            info = _parse_quietly(fh.read(1 << 16))
    except OSError:
        return False
    return info is not None and _is_coco(info, path.parent)


def detect_coco(path: str | Path) -> bool:
    """True for a COCO ``.info`` file, or a folder that holds one."""
    path = Path(path)
    if path.is_dir():
        return any(is_coco_info(p) for p in sorted(path.rglob("*.info")))
    return is_coco_info(path)


# --------------------------------------------------------------------------- reading


@dataclass
class _Ref:
    """A block of an ``.info`` file and the entries it gives one data file."""

    info: str
    block: InfoBlock


@dataclass
class _Reader:
    tables: Tables
    root: Path
    used: set[str] = field(default_factory=set)
    counts: Counter[str] = field(default_factory=Counter)

    def local(self, path: str) -> Path:
        return self.root.joinpath(*path.split("/"))


def read_coco(
    source: str | Path,
    dataset_id: str,
    *,
    root: str | Path | None = None,
) -> ReadResult:
    """Read COCO single-objective data into the tables of format v1.

    ``source`` is a folder (searched recursively) or an ``.info`` file. Source
    paths are relative to ``root``, by default the folder itself or the file's
    folder.
    """
    source, root_path = input_root(source, root)
    reader = _Reader(Tables(dataset_id), root_path)
    infos = _read_infos(source, reader)
    if not infos:
        raise ReaderError("no COCO .info file found", source)
    for stem, refs in _stems(infos).items():
        _add_stem(stem, refs, reader)
    if source.is_dir():
        _unreferenced(source, reader)
    for code, n in reader.counts.items():
        reader.tables.warn(code, f"{n} {_SUMMARIES[code]}")
    return reader.tables.result(READER_NAME)


def _files(source: Path, root: Path, kinds: tuple[str, ...]) -> list[tuple[str, Path]]:
    """The files under source with a suffix in kinds, by source path."""
    paths = source.rglob("*") if source.is_dir() else [source]
    found = [
        (source_path(p, root), p)
        for p in paths
        if p.suffix.lower()[1:] in kinds and p.is_file()
    ]
    return sorted(found)


def _read_infos(source: Path, reader: _Reader) -> list[InfoFile]:
    infos = []
    for path, local in _files(source, reader.root, ("info",)):
        info = _load_info(path, local, reader.tables)
        if info is None:
            if not source.is_dir():
                raise ReaderError("not a COCO .info file", path)
            reader.tables.warn("skipped_file", "not a COCO .info file", path)
            continue
        infos.append(info)
        reader.used.add(path)
        _info_source(info, local, reader.tables)
    return infos


def _load_info(path: str, local: Path, tables: Tables) -> InfoFile | None:
    """The parsed ``.info`` file, or None when it is not COCO's."""
    data = local.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        tables.warn("not_utf8", "read as Latin-1, as it is not valid UTF-8", path)
        text = data.decode("latin-1")
    try:
        info = parse_info(text, path)
    except ReaderError:
        if _parse_quietly(text.split("\n", 1)[0]) is None:
            return None
        raise
    return info if _is_coco(info, local.parent) else None


def _info_source(info: InfoFile, local: Path, tables: Tables) -> None:
    entry = file_entry(local, info.path, kind="coco-info")
    version = info.blocks[0].get("coco_version") if info.blocks else None
    if version is not None:
        entry["logger"] = {"name": "COCO", "version": str(version)}
    tables.sources.append(entry)
    for line, text in info.skipped:
        tables.warn(
            "info_line_skipped",
            f"line {line} is no header, comment or data line: {text[:80]!r}",
            info.path,
        )
    if info.incomplete:
        tables.warn(
            "incomplete_info_record",
            f"header line(s) {info.incomplete} have no data line",
            info.path,
        )


def _stems(infos: list[InfoFile]) -> dict[str, list[_Ref]]:
    """The blocks that name each data file stem, in file and block order."""
    stems: dict[str, list[_Ref]] = {}
    for info in infos:
        folder = posixpath.dirname(info.path)
        for block in info.blocks:
            name = posixpath.normpath(posixpath.join(folder, block.data_file))
            try:
                stem = check_source_path(posixpath.splitext(name)[0])
            except ValueError as exc:
                raise ReaderError(str(exc), info.path, block.line) from exc
            stems.setdefault(stem, []).append(_Ref(info.path, block))
    return stems


def _add_stem(stem: str, refs: list[_Ref], reader: _Reader) -> None:
    dims = {ref.block.dimension for ref in refs}
    if len(dims) > 1:
        raise ReaderError(
            f"{stem}.dat is named with dimensions {sorted(dims)}", refs[0].info
        )
    files = _parse_files(stem, dims.pop(), reader)
    entries = [(ref, e) for ref in refs for e in ref.block.entries]
    if not files:
        primary = f"{stem}.dat"
        reader.tables.sources.append(
            {"path": primary, "sha256": None, "bytes": None, "kind": "coco-dat"}
        )
        reader.tables.warn(
            "missing_data_file", f"named in {refs[0].info} but not found", primary
        )
        _add_runs(primary, None, entries, refs, reader)
        return
    counts = {f.kind: len(f.fopt) for f in files}
    if len(set(counts.values())) > 1:
        raise ReaderError(
            f"the data files of this stem hold different numbers of runs: {counts}",
            files[0].path,
        )
    if len(files[0].fopt) != len(entries):
        reader.tables.warn(
            "run_count_mismatch",
            f"{len(files[0].fopt)} run(s) in the data file, {len(entries)} in the "
            ".info file(s); runs are matched in order",
            files[0].path,
        )
    ids = _add_runs(files[0].path, files[0], entries, refs, reader)
    _add_trace(files, ids, reader.tables)


def _parse_files(stem: str, dimension: int, reader: _Reader) -> list[CocoDat]:
    files = []
    for kind in KINDS:
        path = f"{stem}.{kind}"
        local = reader.local(path)
        if not local.is_file():
            continue
        dat = parse_coco_dat(local, display=path, dimension=dimension, kind=kind)
        files.append(dat)
        reader.used.add(path)
        reader.tables.sources.append(
            file_entry(
                local,
                path,
                kind=f"coco-{kind}",
                data_rows=dat.frame.height,
                triggers=TRIGGERS[kind],
                precision=dict(PRECISION),
            )
        )
    if len({f.style for f in files if f.style is not None}) > 1:
        raise ReaderError(
            "the data files of this stem have different header styles", stem
        )
    return files


def _add_runs(
    primary: str,
    dat: CocoDat | None,
    entries: list[tuple[_Ref, InfoEntry]],
    refs: list[_Ref],
    reader: _Reader,
) -> dict[int, int]:
    """Add the runs of one stem; returns the run id of each run index."""
    fopt, hashes = ([], []) if dat is None else (dat.fopt, dat.lines.sha256)
    ids: dict[int, int] = {}
    for index in range(max(len(fopt), len(entries))):
        rid = ids[index] = run_id(reader.tables.dataset_id, primary, index)
        ref, entry = entries[index] if index < len(entries) else (refs[-1], None)
        block = ref.block
        if entry is not None and entry.evals is None:
            reader.counts["unfinalized_run"] += 1
        reader.tables.runs.append(
            {
                "run_id": rid,
                "suite": _suite(block, reader),
                "function_id": block.function_id,
                "instance": None if entry is None else entry.instance,
                "dimension": block.dimension,
                "maximization": False,
                "optimum": fopt[index] if index < len(fopt) else None,
                "evals_used": None if entry is None else entry.evals,
                "n_objectives": 1,
                "source_file": primary,
                "source_run_index": index,
                "source_rows_sha256": hashes[index] if index < len(hashes) else None,
            }
        )
        reader.tables.descriptors.extend(_descriptors(rid, ref, entry, reader))
    return ids


def _suite(block: InfoBlock, reader: _Reader) -> str:
    suite = block.get("suite")
    if suite is not None:
        return str(suite)
    reader.counts["suite_inferred"] += 1
    return "bbob-noisy" if block.function_id > 100 else "bbob"


def _descriptors(
    rid: int, ref: _Ref, entry: InfoEntry | None, reader: _Reader
) -> list[tuple[int, str, Any, str]]:
    block = ref.block
    algorithm = block.get("algId")
    if algorithm is None:
        reader.counts["missing_algorithm"] += 1
    out: list[tuple[int, str, Any, str]] = [
        (
            rid,
            "algorithm_name",
            "" if algorithm is None else str(algorithm),
            "algorithm",
        ),
        (rid, "algorithm_info", block.comment.lstrip("%").strip(), "algorithm"),
    ]
    taken = {"algorithm_name", "algorithm_info", FINAL_KEY}
    for key, value in block.header:
        if ALIASES.get(key, key) not in RUN_HEADER_KEYS:
            name = reader.tables.renamer.descriptor_key(
                key, taken, "experiment", ref.info
            )
            out.append((rid, name, value, "experiment"))
    if entry is not None and entry.value is not None:
        out.append((rid, FINAL_KEY, entry.value, "run"))
    return out


def _add_trace(files: list[CocoDat], ids: dict[int, int], tables: Tables) -> None:
    primary = files[0].path
    merged = merge_files([(f.kind, f.frame) for f in files])
    if merged.conflicts:
        tables.warn(
            "merge_conflict",
            f"{merged.conflicts} row(s) of evaluations the data files log with "
            "different values; all are kept",
            primary,
        )
    if merged.unsorted:
        tables.warn(
            "unsorted_evaluations",
            "evaluations decrease within a run; the files' rows are not merged",
            primary,
        )
    frame = merged.trace.select(
        pl.col("run_index").replace_strict(ids, return_dtype=pl.Int64).alias("run_id"),
        pl.exclude("run_index"),
    )
    xs = [c for c in frame.columns if c[:1] == "x" and c[1:].isdigit()]
    if xs:
        tables.add(
            "positions",
            frame.filter(pl.col(xs[0]).is_not_null()).select("run_id", "row", *xs),
        )
    tables.add("trace", frame.drop(xs))


def _unreferenced(source: Path, reader: _Reader) -> None:
    for path, _ in _files(source, reader.root, ("info", *KINDS, *UNREAD_KINDS)):
        if path in reader.used:
            continue
        kind = posixpath.splitext(path)[1][1:]
        message = (
            f"COCO .{kind} files are not read"
            if kind in UNREAD_KINDS
            else "no .info file names this file; not read"
        )
        reader.tables.warn("unreferenced_data_file", message, path)
