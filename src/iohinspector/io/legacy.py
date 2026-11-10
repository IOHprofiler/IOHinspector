"""Reader for legacy IOHprofiler output: ``.info`` files with their data files.

This is the format of the IOHprofiler loggers before the JSON metadata of
IOHexperimenter 0.3.3: the original IOHprofiler, IOHexperimenter 0.2 (``Default``
logger) and the ``.info`` Analyzer of IOHexperimenter 0.3.0 to 0.3.2. IOHanalyzer
reads it with ``read_index_file__IOH`` and ``read_dat`` (``R/readFiles.R``). The
page ``docs/readers-legacy.md`` describes the mapping to format v1.

The reader is lossless: every data line of every data file of a run set becomes
a trace row, except that a row two files log identically for the same evaluation
is stored once (``iohinspector.io.merge``, shared with the COCO reader). Nothing is deduplicated within a file or made
monotone. Malformed input raises ``ReaderError`` with the file name and line.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from ..format import check_source_path, descriptor_frame, file_entry, new_manifest
from ..format import run_id as make_run_id
from ..format.schema import descriptor_key_problem
from ._legacy_dat import DataFile, parse_data_file
from ._legacy_info import InfoBlock, InfoRun, parse_info
from ._legacy_meta import BlockMeta, FunctionIds, block_meta, run_value
from .base import ReaderError, ReadResult
from .merge import Merged, merge_files

READER_NAME = "ioh-legacy"
INFO_KIND = "ioh-legacy-info"
#: Which file is the run set's ``source_file`` when the ``.info`` file names one
#: that does not exist: IOHanalyzer's preference (``R/DataSet.R``), then ``.idat``.
PRIMARY_ORDER = ("dat", "tdat", "cdat", "idat")
DATA_SUFFIXES = tuple(f".{k}" for k in PRIMARY_ORDER)
TRIGGERS: dict[str, list[dict[str, Any]]] = {
    "cdat": [{"kind": "all"}],
    "tdat": [{"kind": "fixed_budget", "params": {"file": "tdat (time points)"}}],
    "idat": [
        {"kind": "fixed_budget", "params": {"file": "idat (every n-th evaluation)"}}
    ],
}


# --------------------------------------------------------------------------- detection


#: File kinds in their merge order.
MERGE_ORDER = ("dat", "tdat", "cdat", "idat")


def _stem(path: Path) -> Path:
    return path.with_suffix("") if path.suffix in DATA_SUFFIXES else path


def _first_line(path: Path) -> str | None:
    try:
        with open(path, "rb") as fh:
            for raw in fh:
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    return line
    except OSError:
        return None
    return None


def info_flavour(path: Path) -> str:
    """``legacy``, ``coco`` (COCO data files start with ``%``) or ``unknown``.

    Looks at the first data file the ``.info`` file names that exists; ``unknown``
    when the file cannot be parsed or none of its data files exist.
    """
    try:
        blocks = parse_info(path, str(path))
    except ReaderError:
        return "unknown"
    for block in blocks:
        stem = _stem(path.parent / block.datafile)
        for suffix in DATA_SUFFIXES:
            line = _first_line(stem.with_suffix(suffix))
            if line is not None:
                return "coco" if line.startswith("%") else "legacy"
    return "unknown"


def detect_legacy(path: str | os.PathLike[str]) -> bool:
    """True for a legacy IOHprofiler ``.info`` file, or a folder that holds one."""
    path = Path(path)
    if path.is_dir():
        return any(info_flavour(p) == "legacy" for p in sorted(path.rglob("*.info")))
    return path.suffix == ".info" and info_flavour(path) == "legacy"


# --------------------------------------------------------------------------- reading


@dataclass
class _RunSet:
    """The data files of one stem, and the ``.info`` blocks that name it."""

    stem: Path
    named: Path
    blocks: list[InfoBlock] = field(default_factory=list)


@dataclass
class _Builder:  # pylint: disable=too-many-instance-attributes
    dataset_id: str
    root: Path
    runs: list[dict[str, Any]] = field(default_factory=list)
    descriptors: list[tuple[int, str, Any, str]] = field(default_factory=list)
    traces: list[pl.DataFrame] = field(default_factory=list)
    positions: list[pl.DataFrame] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    renamed: list[dict[str, str]] = field(default_factory=list)

    def warn(self, code: str, message: str, source: str | None = None) -> None:
        entry = {"code": code, "message": message}
        if source is not None:
            entry["source"] = source
        if entry not in self.warnings:
            self.warnings.append(entry)

    def display(self, path: Path) -> str:
        try:
            rel = path.resolve().relative_to(self.root.resolve())
        except ValueError as exc:
            raise ReaderError(f"is outside the input folder {self.root}", path) from exc
        try:
            return check_source_path(rel.as_posix())
        except ValueError as exc:
            raise ReaderError(str(exc), path) from exc

    def rename(self, old: str, new: str, source: str, table: str) -> None:
        entry = {"from": old, "to": new, "source": source, "table": table}
        if entry not in self.renamed:
            self.renamed.append(entry)


def read_legacy(
    source: str | os.PathLike[str],
    dataset_id: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> ReadResult:
    """Read legacy IOHprofiler ``.info`` + data files into the tables of format v1.

    ``source`` is an ``.info`` file or a folder, searched recursively for ``.info``
    files; COCO ``.info`` files in a folder are skipped with a ``skipped_file``
    warning. Source paths are relative to ``root``, by default the folder itself or
    the ``.info`` file's folder.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")
    root_path = (
        Path(root)
        if root is not None
        else (source if source.is_dir() else source.parent)
    )
    builder = _Builder(dataset_id, root_path)
    run_sets, skipped = _collect(source, builder)
    ids = FunctionIds.of(b for rs in run_sets for b in rs.blocks)
    if ids.assigned:
        builder.warn("function_ids_assigned", ids.describe())
    for run_set in run_sets:
        _add_run_set(run_set, ids, builder)
    if source.is_dir():
        _unreferenced(source, run_sets, skipped, builder)
    return _result(builder)


def _info_files(source: Path) -> list[Path]:
    if source.is_dir():
        return [p for p in sorted(source.rglob("*.info")) if p.is_file()]
    return [source]


def _collect(source: Path, builder: _Builder) -> tuple[list[_RunSet], set[Path]]:
    """Parse the ``.info`` files and group their blocks by data file."""
    run_sets: dict[Path, _RunSet] = {}
    skipped: set[Path] = set()
    for path in _info_files(source):
        display = builder.display(path)
        if source.is_dir() and not path.read_bytes().strip():
            builder.warn("skipped_file", "an empty .info file; not read", display)
            continue
        if source.is_dir() and info_flavour(path) == "coco":
            builder.warn("skipped_file", "a COCO .info file; not read", display)
            skipped.update(
                _stem(path.parent / b.datafile) for b in parse_info(path, display)
            )
            continue
        builder.sources.append(file_entry(path, display, kind=INFO_KIND))
        for block in parse_info(path, display):
            named = path.parent / block.datafile
            stem = _stem(named).resolve()
            run_sets.setdefault(stem, _RunSet(stem, named)).blocks.append(block)
    if not run_sets:
        raise ReaderError("no legacy IOHprofiler .info file found", source)
    return list(run_sets.values()), skipped


def _parse_files(run_set: _RunSet, dimension: int, builder: _Builder) -> list[DataFile]:
    files = []
    for kind in PRIMARY_ORDER:
        local = run_set.stem.with_suffix(f".{kind}")
        if local.is_file():
            files.append(
                parse_data_file(local, builder.display(local), kind, dimension)
            )
    return files


def _primary(run_set: _RunSet, files: list[DataFile], builder: _Builder) -> DataFile:
    named = builder.display(run_set.named) if run_set.named.is_file() else None
    for f in files:
        if f.path == named:
            return f
    return files[0]


def _add_run_set(run_set: _RunSet, ids: FunctionIds, builder: _Builder) -> None:
    metas = [block_meta(b, ids) for b in run_set.blocks]
    for meta in metas:
        for code, message in meta.warnings:
            builder.warn(code, message, meta.block.info_path)
    entries = [(meta, run) for meta in metas for run in meta.block.runs]
    files = _parse_files(run_set, metas[0].dimension, builder)
    if not files:
        _missing(run_set, entries, metas[-1], builder)
        return
    primary = _primary(run_set, files, builder)
    for f in files:
        if f.n_runs != primary.n_runs:
            raise ReaderError(
                f"{f.n_runs} run(s), but {primary.path} has {primary.n_runs}; the runs "
                "of the files cannot be matched",
                f.path,
            )
    ordered = sorted(files, key=lambda f: MERGE_ORDER.index(f.kind))
    merged = merge_files([(f.kind, f.trace.drop("line")) for f in ordered])
    _file_sources(files, metas[0].maximization, builder)
    _merge_warnings(merged, files, primary, builder)
    if primary.n_runs != len(entries):
        builder.warn(
            "run_count_mismatch",
            f"{primary.n_runs} run(s) in the data file, {len(entries)} in the .info "
            "file; runs are matched in order",
            primary.path,
        )
    run_ids = _add_runs(primary, entries, metas[-1], builder)
    _add_trace(merged, metas[0], run_ids, builder)


def _missing(
    run_set: _RunSet,
    entries: list[tuple[BlockMeta, InfoRun]],
    last: BlockMeta,
    builder: _Builder,
) -> None:
    display = builder.display(run_set.named)
    builder.sources.append(
        {"path": display, "sha256": None, "bytes": None, "kind": "ioh-legacy-dat"}
    )
    builder.warn("missing_data_file", "named in an .info file but not found", display)
    _add_runs(None, entries, last, builder, display)


def _file_sources(files: list[DataFile], maximization: bool, builder: _Builder) -> None:
    limited = []
    for f in files:
        local = builder.root / f.path
        entry: dict[str, Any] = {
            "kind": f"ioh-legacy-{f.kind}",
            "data_rows": f.trace.height,
            "precision": f.precision,
        }
        triggers = TRIGGERS.get(f.kind) or dat_triggers(f.trace, maximization)
        if triggers:
            entry["triggers"] = triggers
        builder.sources.append(file_entry(local, f.path, **entry))
        if f.precision.get("notation") == "fixed":
            limited.append(f)
        for old, new in f.renamed:
            builder.rename(old, new, f.path, "trace")
    for f in limited:
        digits = f.precision["digits"]
        builder.warn(
            "limited_precision",
            f"values are written with {digits} decimals, so values below "
            f"5e-{digits + 1} are stored as 0; nothing was corrected",
            f.path,
        )


def dat_triggers(
    trace: pl.DataFrame, maximization: bool
) -> list[dict[str, Any]] | None:
    """Triggers of a ``.dat`` file, inferred: ``all`` or ``improvement``, else None.

    ``all`` when every run logs evaluations 1, 2, 3, ...; ``improvement`` when every
    row but a run's last improves on the run's earlier rows.
    """
    if trace.is_empty():
        return None
    y = pl.col("raw_y")
    best = (y.cum_max() if maximization else y.cum_min()).shift(1).over("run_index")
    better = (y > best) if maximization else (y < best)
    flags = trace.select(
        all=(
            pl.col("evaluations") == pl.int_range(1, pl.len() + 1).over("run_index")
        ).all(),
        improvement=(
            better.fill_null(True)
            | (
                pl.int_range(pl.len()).over("run_index")
                == pl.len().over("run_index") - 1
            )
        ).all(),
    )
    params = {"inferred": True}
    if flags["all"][0]:
        return [{"kind": "all", "params": params}]
    if flags["improvement"][0]:
        return [{"kind": "improvement", "params": params}]
    return None


def _merge_warnings(
    merged: Merged, files: list[DataFile], primary: DataFile, builder: _Builder
) -> None:
    names = ", ".join(f.path for f in files)
    if merged.unsorted:
        builder.warn(
            "unsorted_evaluations",
            f"evaluation counts decrease within a run in {names}; rows are not merged",
            primary.path,
        )
    if merged.conflicts:
        builder.warn(
            "merge_conflict",
            f"{merged.conflicts} row(s) of {names} log the same evaluation with "
            "different values; all are kept with merge_conflict = true",
            primary.path,
        )


def _add_runs(
    primary: DataFile | None,
    entries: list[tuple[BlockMeta, InfoRun]],
    last: BlockMeta,
    builder: _Builder,
    display: str | None = None,
) -> dict[int, int]:
    source = primary.path if primary is not None else display or ""
    n_data = primary.n_runs if primary is not None else 0
    out: dict[int, int] = {}
    for index in range(max(n_data, len(entries))):
        meta, info = entries[index] if index < len(entries) else (last, None)
        rid = make_run_id(builder.dataset_id, source, index)
        out[index] = rid
        sha = primary.run_hashes[index] if primary and index < n_data else None
        builder.runs.append(meta.run_row(rid, info, source, index, sha))
        builder.descriptors.extend(_descriptors(rid, meta, info, builder))
    return out


def _descriptors(
    rid: int, meta: BlockMeta, info: InfoRun | None, builder: _Builder
) -> list[tuple[int, str, Any, str]]:
    out = [
        (rid, "algorithm_name", meta.algorithm_name, "algorithm"),
        (rid, "algorithm_info", meta.algorithm_info, "algorithm"),
    ]
    taken = {"algorithm_name", "algorithm_info"}
    pairs = list(meta.experiment)
    if info is not None:
        pairs += [
            (k, v, "run")
            for k, v in zip(meta.run_attributes, _run_values(info), strict=True)
        ]
    for key, value, origin in pairs:
        new = _descriptor_key(key, taken, origin)
        if new != key:
            builder.rename(key, new, meta.block.info_path, "descriptors")
        taken.add(new)
        out.append((rid, new, value, origin))
    return out


def _run_values(info: InfoRun) -> Iterable[Any]:
    return [run_value(v) for v in info.attributes]


def _descriptor_key(key: str, taken: set[str], origin: str) -> str:
    new, n = key, 1
    while descriptor_key_problem(new) is not None or new in taken:
        new = f"{key}_{origin}" if n == 1 else f"{key}_{origin}{n}"
        n += 1
    return new


def _add_trace(
    merged: Merged, meta: BlockMeta, run_ids: dict[int, int], builder: _Builder
) -> None:
    trace = merged.trace
    if trace.is_empty():
        return
    trace = trace.with_columns(
        run_id=pl.col("run_index").replace_strict(run_ids, return_dtype=pl.Int64)
    ).drop("run_index")
    xs = [f"x{i}" for i in range(meta.dimension)]
    if all(x in trace.columns for x in xs):
        pos = trace.select("run_id", "row", *xs).filter(pl.col("x0").is_not_null())
        builder.positions.append(pos)
        trace = trace.drop(xs)
    fixed = ["run_id", "row", "evaluations", "raw_y"]
    builder.traces.append(
        trace.select(fixed + [c for c in trace.columns if c not in fixed])
    )
    _check_best_so_far(trace, meta.maximization, builder)


#: The legacy column with the logger's own best-so-far value.
BEST_SO_FAR = "best-so-far f(x)"


def _check_best_so_far(
    trace: pl.DataFrame, maximization: bool, builder: _Builder
) -> None:
    """Warn when the logged best-so-far differs from the running best of ``raw_y``.

    Analysis computes best-so-far values from ``raw_y``; IOHanalyzer reads the
    logged column instead. They differ when the logger did not log every
    improvement (``.tdat`` or ``.idat`` without ``.dat``).
    """
    if BEST_SO_FAR not in trace.columns:
        return
    y = pl.col("raw_y")
    best = (y.cum_max() if maximization else y.cum_min()).over("run_id")
    differs = trace.select(
        n=(pl.col(BEST_SO_FAR).is_not_null() & (pl.col(BEST_SO_FAR) != best)).sum()
    )["n"][0]
    if differs:
        builder.warn(
            "best_so_far_differs",
            f"in {differs} row(s) the logged {BEST_SO_FAR!r} is not the running best "
            "of raw_y, so not every improvement was logged; analyses that use raw_y "
            "differ there from IOHanalyzer, which reads the logged column",
        )


def _unreferenced(
    source: Path, run_sets: list[_RunSet], skipped: set[Path], builder: _Builder
) -> None:
    known = {rs.stem for rs in run_sets} | {s.resolve() for s in skipped}
    for path in sorted(p for p in source.rglob("*") if p.suffix in DATA_SUFFIXES):
        if _stem(path).resolve() not in known:
            builder.warn(
                "unreferenced_data_file",
                "no .info file names this file; not read",
                builder.display(path),
            )


def _result(builder: _Builder) -> ReadResult:
    manifest = new_manifest(
        builder.dataset_id,
        READER_NAME,
        sources=builder.sources,
        warnings=builder.warnings,
        renamed_columns=builder.renamed,
    )
    runs = (
        pl.DataFrame(builder.runs, schema=RUNS_INPUT, orient="row")
        if builder.runs
        else pl.DataFrame(schema=RUNS_INPUT)
    )
    trace = (
        pl.concat(builder.traces, how="diagonal")
        if builder.traces
        else pl.DataFrame(schema=EMPTY_TRACE)
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


RUNS_INPUT = {
    "run_id": pl.Int64,
    "suite": pl.String,
    "function_id": pl.Int32,
    "function_name": pl.String,
    "instance": pl.Int32,
    "dimension": pl.Int32,
    "maximization": pl.Boolean,
    "evals_used": pl.Int64,
    "best_y": pl.Float64,
    "n_objectives": pl.Int32,
    "source_file": pl.String,
    "source_run_index": pl.Int64,
    "source_rows_sha256": pl.String,
}
EMPTY_TRACE = {
    "run_id": pl.Int64,
    "row": pl.Int64,
    "evaluations": pl.Int64,
    "raw_y": pl.Float64,
}

__all__ = ["MERGE_ORDER", "detect_legacy", "info_flavour", "read_legacy"]
