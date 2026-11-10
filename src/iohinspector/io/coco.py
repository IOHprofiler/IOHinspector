"""Reader for COCO single-objective data: ``.info`` files with ``.dat``, ``.tdat``
and ``.rdat`` data files, as in the COCO data archive.

Covers the suites ``bbob``, ``bbob-noisy``, ``bbob-largescale`` and
``bbob-boxed`` and every logger version in the archive (2009 to today). The
input is a folder, an ``.info`` file, or an archive (zip, tar, tgz); archives
inside folders and archives are opened too (``iohinspector.io.archive``), and
one input may hold many algorithms. ``docs/readers-coco.md`` describes the
mapping to format v1 in full.

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
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from ..format import (
    check_source_path,
    descriptor_frame,
    file_entry,
    new_manifest,
    run_id,
)
from ._coco_dat import CocoDat, parse_coco_dat
from ._coco_info import ALIASES, InfoBlock, InfoEntry, InfoFile, parse_info
from .archive import SourceTree, is_archive, open_tree
from .base import NUMBER, ReaderError, ReadResult, Renamer
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
ARCHIVE_KEY = "archive"
POSITION_PREFIX = "x"


def _wanted(path: str) -> bool:
    suffix = posixpath.splitext(path)[1].lower()[1:]
    return suffix == "info" or suffix in KINDS or suffix in UNREAD_KINDS


# --------------------------------------------------------------------------- detection


def _first_line(path: Path) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    return line
    except OSError:
        pass
    return ""


def _coco_data(local: Path) -> bool | None:
    """Whether a data file looks like COCO's: its first line is a ``%`` line.
    None when it starts with data, so that the ``.info`` header decides."""
    line = _first_line(local).strip()
    if line.startswith("%"):
        return True
    tokens = line.split()
    return None if not tokens or NUMBER.fullmatch(tokens[0]) else False


def is_coco_info(path: Path, read_data: bool = True) -> bool:
    """A COCO ``.info`` file: a header with ``funcId`` and ``DIM`` whose data file
    starts with a ``%`` header line (or, when it is missing, a ``Precision``)."""
    if path.suffix.lower() != ".info" or not path.is_file():
        return False
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            head = fh.read(1 << 16)
    except OSError:
        return False
    info = parse_info_quietly(head)
    if info is None or not info.blocks:
        return False
    if read_data:
        data = path.parent / info.blocks[0].data_file
        for kind in KINDS:
            local = data.with_suffix("." + kind)
            if local.is_file():
                verdict = _coco_data(local)
                if verdict is not None:
                    return verdict
                break
    return info.blocks[0].get("Precision") is not None


def parse_info_quietly(text: str) -> InfoFile | None:
    try:
        return parse_info(text, "")
    except ReaderError:
        return None


def _archive_is_coco(path: Path) -> bool:
    with open_tree(path, wanted=lambda p: p.endswith(".info")) as tree:
        return any(
            is_coco_info(local, read_data=False) for local in tree.files.values()
        )


def detect_coco(path: str | Path) -> bool:
    """True for a COCO ``.info`` file, an archive holding one, or a folder with
    either."""
    path = Path(path)
    if path.is_dir():
        if any(is_coco_info(p) for p in sorted(path.rglob("*.info"))):
            return True
        candidates = [p for p in sorted(path.rglob("*")) if is_archive(p.name)]
        return any(_archive_is_coco(p) for p in candidates)
    if is_archive(path.name):
        return _archive_is_coco(path)
    return is_coco_info(path)


# --------------------------------------------------------------------------- reading


@dataclass
class _Ref:
    """A block of an ``.info`` file and the entries it gives one data file."""

    info: str
    block: InfoBlock


@dataclass
class _Builder:  # pylint: disable=too-many-instance-attributes
    dataset_id: str
    tree: SourceTree
    runs: list[dict[str, Any]] = field(default_factory=list)
    descriptors: list[tuple[int, str, Any, str]] = field(default_factory=list)
    traces: list[pl.DataFrame] = field(default_factory=list)
    positions: list[pl.DataFrame] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    renamer: Renamer = field(default_factory=Renamer)
    counters: dict[str, int] = field(default_factory=dict)
    used: set[str] = field(default_factory=set)

    def warn(self, code: str, message: str, source: str | None = None) -> None:
        entry = {"code": code, "message": message}
        if source is not None:
            entry["source"] = source
        self.warnings.append(entry)

    def count(self, code: str, n: int = 1) -> None:
        self.counters[code] = self.counters.get(code, 0) + n

    def archive_fields(self, path: str) -> dict[str, str]:
        archive = self.tree.archive_of.get(path)
        return {} if archive is None else {"archive": archive}


def read_coco(
    source: str | Path,
    dataset_id: str,
    *,
    root: str | Path | None = None,
) -> ReadResult:
    """Read COCO single-objective data into the tables of format v1.

    ``source`` is a folder (searched recursively, archives included), an
    ``.info`` file or an archive. Source paths are relative to ``root``, by
    default the folder itself or the file's folder; files inside archives are
    named ``<archive path>/<member path>``.
    """
    source = Path(source)
    with open_tree(source, root, _wanted) as tree:
        builder = _Builder(dataset_id, tree)
        builder.warnings.extend(tree.warnings)
        infos = _read_infos(
            tree, builder, single=not source.is_dir() and not is_archive(source.name)
        )
        if not infos:
            raise ReaderError("no COCO .info file found", source)
        for stem, refs in _stems(infos).items():
            _add_stem(stem, refs, builder)
        _unreferenced(builder)
        _summary_warnings(builder)
        return _result(builder)


def _decode_info(local: Path, path: str, builder: _Builder) -> str:
    data = local.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        builder.warn("not_utf8", "read as Latin-1, as it is not valid UTF-8", path)
        return data.decode("latin-1")


def _data_stem(info_path: str, block: InfoBlock) -> str:
    folder = posixpath.dirname(info_path)
    name = posixpath.normpath(posixpath.join(folder, block.data_file))
    return posixpath.splitext(name)[0]


def _is_coco(info: InfoFile, tree: SourceTree) -> bool:
    """The first data file found starts with a COCO header line; with no data
    file, the header has a ``Precision``."""
    if not info.blocks:
        return False
    stem = _data_stem(info.path, info.blocks[0])
    for kind in KINDS:
        local = tree.resolve(f"{stem}.{kind}")
        if local is not None:
            verdict = _coco_data(local)
            if verdict is not None:
                return verdict
            break
    return info.blocks[0].get("Precision") is not None


def _load_info(path: str, local: Path, builder: _Builder) -> InfoFile | None:
    """The parsed ``.info`` file, or None when it is not COCO's."""
    text = _decode_info(local, path, builder)
    try:
        info = parse_info(text, path)
    except ReaderError:
        if parse_info_quietly(text.split("\n", 1)[0]) is None:
            return None
        raise
    return info if _is_coco(info, builder.tree) else None


def _read_infos(tree: SourceTree, builder: _Builder, single: bool) -> list[InfoFile]:
    infos = []
    for path, local in sorted(tree.files.items()):
        if not path.lower().endswith(".info"):
            continue
        info = _load_info(path, local, builder)
        if info is None:
            if single:
                raise ReaderError("not a COCO .info file", path)
            builder.warn("skipped_file", "not a COCO .info file", path)
            continue
        infos.append(info)
        builder.used.add(path)
        _info_source(info, local, builder)
    return infos


def _info_source(info: InfoFile, local: Path, builder: _Builder) -> None:
    entry = file_entry(
        local, info.path, kind="coco-info", **builder.archive_fields(info.path)
    )
    version = info.blocks[0].get("coco_version") if info.blocks else None
    if version is not None:
        entry["logger"] = {"name": "COCO", "version": str(version)}
    builder.sources.append(entry)
    for line, text in info.skipped:
        builder.warn(
            "info_line_skipped",
            f"line {line} is no header, comment or data line: {text[:80]!r}",
            info.path,
        )
    if info.incomplete:
        builder.warn(
            "incomplete_info_record",
            f"header line(s) {info.incomplete} have no data line",
            info.path,
        )


def _stems(infos: list[InfoFile]) -> dict[str, list[_Ref]]:
    """The blocks that name each data file stem, in file and block order."""
    stems: dict[str, list[_Ref]] = {}
    for info in infos:
        for block in info.blocks:
            try:
                stem = check_source_path(_data_stem(info.path, block))
            except ValueError as exc:
                raise ReaderError(str(exc), info.path, block.line) from exc
            stems.setdefault(stem, []).append(_Ref(info.path, block))
    return stems


def _add_stem(stem: str, refs: list[_Ref], builder: _Builder) -> None:
    dims = {ref.block.dimension for ref in refs}
    if len(dims) > 1:
        raise ReaderError(
            f"{stem}.dat is named with dimensions {sorted(dims)}", refs[0].info
        )
    dimension = dims.pop()
    files = _parse_files(stem, dimension, builder)
    entries = [(ref, e) for ref in refs for e in ref.block.entries]
    if not files:
        primary = f"{stem}.dat"
        builder.sources.append(
            {"path": primary, "sha256": None, "bytes": None, "kind": "coco-dat"}
        )
        builder.warn(
            "missing_data_file", f"named in {refs[0].info} but not found", primary
        )
        _add_runs(primary, None, entries, refs, builder)
        return
    counts = {f.kind: len(f.runs) for f in files}
    if len(set(counts.values())) > 1:
        raise ReaderError(
            f"the data files of this stem hold different numbers of runs: {counts}",
            files[0].path,
        )
    if len(files[0].runs) != len(entries):
        builder.warn(
            "run_count_mismatch",
            f"{len(files[0].runs)} run(s) in the data file, {len(entries)} in the "
            ".info file(s); runs are matched in order",
            files[0].path,
        )
    _add_runs(files[0].path, files, entries, refs, builder)


def _parse_files(stem: str, dimension: int, builder: _Builder) -> list[CocoDat]:
    files = []
    for kind in KINDS:
        path = f"{stem}.{kind}"
        local = builder.tree.resolve(path)
        if local is None:
            continue
        dat = parse_coco_dat(local, display=path, dimension=dimension, kind=kind)
        files.append(dat)
        builder.used.add(path)
        builder.sources.append(
            file_entry(
                local,
                path,
                kind=f"coco-{kind}",
                data_rows=dat.frame.height,
                triggers=TRIGGERS[kind],
                precision=dict(PRECISION),
                **builder.archive_fields(path),
            )
        )
    styles = {f.style for f in files if f.style is not None}
    if len(styles) > 1:
        raise ReaderError(
            "the data files of this stem have different header styles", stem
        )
    return files


def _add_runs(
    primary: str,
    files: list[CocoDat] | None,
    entries: list[tuple[_Ref, InfoEntry]],
    refs: list[_Ref],
    builder: _Builder,
) -> None:
    runs = files[0].runs if files else []
    ids: dict[int, int] = {}
    for index in range(max(len(runs), len(entries))):
        rid = run_id(builder.dataset_id, primary, index)
        ids[index] = rid
        ref, entry = entries[index] if index < len(entries) else (refs[-1], None)
        dat_run = runs[index] if index < len(runs) else None
        builder.runs.append(
            _run_row(rid, primary, index, ref.block, entry, dat_run, builder)
        )
        builder.descriptors.extend(_descriptors(rid, ref, entry, builder))
    if files:
        _add_trace(files, ids, primary, builder)


def _suite(block: InfoBlock, builder: _Builder) -> str:
    suite = block.get("suite")
    if suite is not None:
        return str(suite)
    builder.count("suite_inferred")
    return "bbob-noisy" if block.function_id > 100 else "bbob"


def _run_row(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    rid: int,
    primary: str,
    index: int,
    block: InfoBlock,
    entry: InfoEntry | None,
    dat_run: Any,
    builder: _Builder,
) -> dict[str, Any]:
    if entry is not None and entry.evals is None:
        builder.count("unfinalized_run")
    return {
        "run_id": rid,
        "suite": _suite(block, builder),
        "function_id": block.function_id,
        "function_name": None,
        "instance": None if entry is None else entry.instance,
        "dimension": block.dimension,
        "maximization": False,
        "optimum": None if dat_run is None else dat_run.fopt,
        "budget": None,
        "evals_used": None if entry is None else entry.evals,
        "best_y": None,
        "n_objectives": 1,
        "source_file": primary,
        "source_run_index": index,
        "source_rows_sha256": None if dat_run is None else dat_run.sha256,
    }


def _descriptors(
    rid: int,
    ref: _Ref,
    entry: InfoEntry | None,
    builder: _Builder,
) -> list[tuple[int, str, Any, str]]:
    block = ref.block
    algorithm = block.get("algId")
    if algorithm is None:
        builder.count("missing_algorithm")
    out: list[tuple[int, str, Any, str]] = [
        (
            rid,
            "algorithm_name",
            "" if algorithm is None else str(algorithm),
            "algorithm",
        ),
        (rid, "algorithm_info", block.comment.lstrip("%").strip(), "algorithm"),
    ]
    taken = {"algorithm_name", "algorithm_info", FINAL_KEY, ARCHIVE_KEY}
    for key, value in block.header:
        if ALIASES.get(key, key) in RUN_HEADER_KEYS:
            continue
        name = builder.renamer.descriptor_key(key, taken, "experiment", ref.info)
        out.append((rid, name, value, "experiment"))
    if entry is not None and entry.value is not None:
        out.append((rid, FINAL_KEY, entry.value, "run"))
    archive = builder.tree.outer_archive_of.get(ref.info)
    if archive is not None:
        out.append((rid, ARCHIVE_KEY, archive, "path"))
    return out


def _add_trace(
    files: list[CocoDat], ids: dict[int, int], primary: str, builder: _Builder
) -> None:
    merged = merge_files([(f.kind, f.frame) for f in files])
    if merged.conflicts:
        builder.warn(
            "merge_conflict",
            f"{merged.conflicts} row(s) of evaluations the data files log with "
            "different values; all are kept",
            primary,
        )
    if merged.unsorted:
        builder.warn(
            "unsorted_evaluations",
            "evaluations decrease within a run; the files' rows are not merged",
            primary,
        )
    frame = merged.trace.with_columns(
        run_id=pl.col("run_index").replace_strict(ids, return_dtype=pl.Int64)
    ).drop("run_index")
    xs = [c for c in frame.columns if c[:1] == POSITION_PREFIX and c[1:].isdigit()]
    if xs:
        builder.positions.append(
            frame.filter(pl.col(xs[0]).is_not_null()).select("run_id", "row", *xs)
        )
    builder.traces.append(frame.select("run_id", pl.exclude("run_id", *xs)))


def _unreferenced(builder: _Builder) -> None:
    if not builder.tree.walked and not builder.tree.archives:
        return
    for path in sorted(builder.tree.files):
        if path in builder.used:
            continue
        kind = posixpath.splitext(path)[1][1:]
        message = (
            f"COCO .{kind} files are not read"
            if kind in UNREAD_KINDS
            else "no .info file names this file; not read"
        )
        builder.warn("unreferenced_data_file", message, path)


def _summary_warnings(builder: _Builder) -> None:
    messages = {
        "suite_inferred": "run(s) without a suite in the .info header: bbob-noisy "
        "for function ids above 100, else bbob (as cocopp does)",
        "unfinalized_run": "run(s) the logger did not finalise (a bare instance in "
        "the .info file): no evaluation count",
        "missing_algorithm": "run(s) without algId; algorithm_name is empty",
    }
    for code, n in builder.counters.items():
        builder.warn(code, f"{n} {messages[code]}")


def _result(builder: _Builder) -> ReadResult:
    manifest = new_manifest(
        builder.dataset_id,
        READER_NAME,
        sources=builder.sources,
        archives=builder.tree.archives,
        warnings=builder.warnings,
        renamed_columns=builder.renamer.records,
    )
    runs = (
        pl.DataFrame(builder.runs, schema=_RUNS_INPUT, orient="row")
        if builder.runs
        else pl.DataFrame(schema=_RUNS_INPUT)
    )
    trace = (
        pl.concat(builder.traces, how="diagonal")
        if builder.traces
        else pl.DataFrame(
            schema={
                "run_id": pl.Int64,
                "row": pl.Int64,
                "evaluations": pl.Int64,
                "raw_y": pl.Float64,
            }
        )
    )
    positions = (
        pl.concat(builder.positions, how="diagonal") if builder.positions else None
    )
    return ReadResult(
        manifest=manifest,
        runs=runs,
        trace=trace,
        descriptors=descriptor_frame(builder.descriptors),
        positions=positions,
    )


_RUNS_INPUT = {
    "run_id": pl.Int64,
    "suite": pl.String,
    "function_id": pl.Int32,
    "function_name": pl.String,
    "instance": pl.Int32,
    "dimension": pl.Int32,
    "maximization": pl.Boolean,
    "optimum": pl.Float64,
    "budget": pl.Int64,
    "evals_used": pl.Int64,
    "best_y": pl.Float64,
    "n_objectives": pl.Int32,
    "source_file": pl.String,
    "source_run_index": pl.Int64,
    "source_rows_sha256": pl.String,
}
