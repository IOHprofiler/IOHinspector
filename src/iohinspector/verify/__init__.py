"""Check a stored dataset against its source files: ``iohinspector verify``.

``verify(source, dataset)`` is an independent line scanner. It shares no code
with the readers: it walks the input itself (``_sources``), splits each data
file into runs with its own rules (``_scan``), and compares every run with the
dataset. Per run of every data file it checks:

- the number of data rows;
- the first and last evaluation and the best value (by the run's maximisation);
- the hash of the data lines against ``runs.source_rows_sha256`` (for the run's
  own ``source_file``), and a fingerprint of every (evaluations, value) pair
  against the trace, so a changed value on either side is found.

It also checks the sha256 and size of every listed source file and archive, the
manifest's ``data_rows``, that every trace row belongs to a verified source run,
and runs ``validate_dataset``. Any mismatch fails the check. Files that merge into
one run set (COCO ``.dat``/``.tdat``/...) are compared through the
trace's ``logged_in`` column: a file's rows are those whose ``logged_in`` names
its kind (its file name suffix), or all rows of its own runs when it is null.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from ..format import read_manifest, validate_dataset
from ._scan import (
    KINDS,
    METADATA_KINDS,
    CsvLayout,
    FileScan,
    Layout,
    ScanError,
    custom_csv_layout,
    scan,
)
from ._sources import SourceFile, walk

_SUFFIX = re.compile(r"\.([^./]*)$")
_KEYS = ["run_id", "label"]


@dataclass
class Mismatch:
    """One difference between the source and the dataset."""

    check: str
    message: str
    source: str | None = None
    run_index: int | None = None
    run_id: int | None = None
    expected: Any = None
    found: Any = None


@dataclass
class VerifyReport:  # pylint: disable=too-many-instance-attributes
    """The result of ``verify``. ``ok`` is true when nothing differs."""

    source: str
    dataset: str
    files: int = 0
    runs: int = 0
    rows: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    not_verified: list[str] = field(default_factory=list)
    unlisted_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def add(self, check: str, message: str, **fields: Any) -> None:
        self.mismatches.append(Mismatch(check, message, **fields))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["ok"] = self.ok
        out["mismatches"] = [
            {k: _json_value(v) for k, v in asdict(m).items() if v is not None}
            for m in self.mismatches
        ]
        return out

    def write(self, path: str | os.PathLike[str]) -> None:
        """Write the report as JSON."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def summary(self, limit: int = 10) -> str:
        """A few lines for the terminal."""
        state = "OK" if self.ok else f"FAILED: {len(self.mismatches)} mismatch(es)"
        counts = f"{self.files} data file(s), {self.runs} run(s), {self.rows} row(s)"
        lines = [f"{state}. {counts} checked"]
        for m in self.mismatches[:limit]:
            where = m.source or ""
            if m.run_index is not None:
                where += f" run {m.run_index}"
            lines.append(f"  {m.check}: {where}: {m.message}")
        if len(self.mismatches) > limit:
            lines.append(f"  ... and {len(self.mismatches) - limit} more")
        if self.not_verified:
            lines.append(f"  not verified (no scanner): {', '.join(self.not_verified)}")
        return "\n".join(lines)


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def suffix(path: str) -> str:
    """The kind label of a file in ``logged_in``: its file name suffix."""
    m = _SUFFIX.search(path)
    return m[1] if m else ""


def _stem(path: str) -> str:
    m = _SUFFIX.search(path)
    return path[: m.start()] if m else path


# ------------------------------------------------------------------- fingerprints


def _aggregate(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Per (run_id, label): row count, first and last evaluation, smallest and
    largest value (NaN ignored), and an order-sensitive fingerprint of the rows.

    ``frame`` has ``run_id``, ``label``, ``evaluations`` and ``y``, in row order.
    """
    y = pl.col("y")
    nan = y.is_nan()
    frame = frame.with_columns(
        pos=pl.int_range(pl.len(), dtype=pl.Int64).over(_KEYS),
        y_nan=nan,
        y_val=pl.when(nan).then(pl.lit(0.0)).otherwise(y),
    )
    h = pl.struct("pos", "evaluations", "y_nan", "y_val").hash(seed=20261110)
    return frame.group_by(_KEYS, maintain_order=True).agg(
        n=pl.len().cast(pl.Int64),
        first=pl.col("evaluations").first(),
        last=pl.col("evaluations").last(),
        ymin=y.filter(~nan).min(),
        ymax=y.filter(~nan).max(),
        h_lo=(h % (1 << 32)).sum(),
        h_hi=(h // (1 << 32)).sum(),
    )


def _trace_aggregates(dataset: Path, runs: pl.DataFrame) -> pl.DataFrame:
    trace = pl.scan_parquet(dataset / "trace.parquet")
    names = trace.collect_schema().names()
    own = pl.col("source_file").str.extract(_SUFFIX.pattern, 1).fill_null("")
    cols = ["run_id", "row", "evaluations", pl.col("raw_y").alias("y")]
    if "logged_in" in names:
        cols.append("logged_in")
        label = (
            pl.when(pl.col("logged_in").is_null())
            .then(pl.concat_list(own))
            .otherwise(pl.col("logged_in").str.split("+"))
        )
    else:
        label = pl.concat_list(own)
    frame = (
        trace.select(cols)
        .join(runs.lazy().select("run_id", "source_file"), on="run_id", how="left")
        .with_columns(label=label)
        .explode("label")
        .sort("run_id", "label", "row")
    )
    return _aggregate(frame).collect()


# --------------------------------------------------------------------------- the walk


@dataclass
class _State:  # pylint: disable=too-many-instance-attributes
    report: VerifyReport
    manifest: dict[str, Any]
    runs: pl.DataFrame
    listed: dict[str, dict[str, Any]]
    archives: dict[str, dict[str, Any]]
    primaries: dict[str, set[str]]
    custom: CsvLayout | None
    seen: set[str] = field(default_factory=set)
    scans: list[pl.DataFrame] = field(default_factory=list)


def _layout(kind: str | None, state: _State) -> Layout | None:
    if kind == "custom-csv":
        return state.custom
    return KINDS.get(kind or "")


def _check_bytes(
    entry: Mapping[str, Any], data: bytes, path: str, state: _State
) -> bool:
    if entry.get("sha256") is None:
        state.report.add(
            "source_present",
            "the dataset lists this file as missing, but the input has it",
            source=path,
        )
        return False
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["sha256"] or len(data) != entry.get("bytes"):
        state.report.add(
            "source_sha256",
            "the file differs from the one converted",
            source=path,
            expected=entry["sha256"],
            found=digest,
        )
    return True


def _visit(file: SourceFile, state: _State) -> None:
    path = file.path
    entry = state.listed.get(path) or state.archives.get(path)
    if entry is None:
        if not file.archive:
            state.report.unlisted_files.append(path)
        return
    state.seen.add(path)
    data = file.load()
    if not _check_bytes(entry, data, path, state) or path not in state.listed:
        return
    kind = entry.get("kind")
    layout = _layout(kind, state)
    if layout is None:
        if kind not in METADATA_KINDS:
            state.report.not_verified.append(f"{path} ({kind})")
        return
    try:
        result = scan(data, path, layout)
    except ScanError as exc:
        state.report.add("scan", str(exc), source=path)
        return
    _check_data_rows(entry, result, state)
    state.scans.append(_file_aggregates(result, layout, state))
    state.report.files += 1
    state.report.rows += sum(result.counts)


def _check_data_rows(entry: Mapping[str, Any], result: FileScan, state: _State) -> None:
    expected = entry.get("data_rows")
    if expected is not None and expected != sum(result.counts):
        state.report.add(
            "data_rows",
            "manifest data_rows differs from the data lines of the file",
            source=result.path,
            expected=sum(result.counts),
            found=expected,
        )


def _primary(path: str, layout: Layout, state: _State) -> str | None:
    """The ``source_file`` of the runs that ``path`` holds rows of."""
    if path in state.primaries.get(_stem(path), set()):
        return path
    if layout.family:
        candidates = state.primaries.get(_stem(path), set())
        if len(candidates) == 1:
            return next(iter(candidates))
    return None


def _file_aggregates(result: FileScan, layout: Layout, state: _State) -> pl.DataFrame:
    primary = _primary(result.path, layout, state)
    n = len(result.counts)
    ids = state.runs.filter(pl.col("source_file") == primary).select(
        pl.col("source_run_index").alias("run_index"), "run_id"
    )
    per_run = pl.DataFrame(
        {
            "run_index": pl.Series(range(n), dtype=pl.Int64),
            "line_sha256": pl.Series(result.hashes, dtype=pl.String),
        }
    ).join(ids, on="run_index", how="left")
    rows = result.rows if result.rows is not None else pl.DataFrame()
    agg = _aggregate(
        rows.lazy()
        .with_row_index("seq")
        .join(per_run.lazy().select("run_index", "run_id"), on="run_index")
        .with_columns(label=pl.lit(suffix(result.path)))
        .sort("seq")  # joins do not keep the order of the rows
        .rename({"run_index": "file_run"})
        .with_columns(pl.col("run_id").fill_null(-1 - pl.col("file_run")))
    ).collect()
    return per_run.with_columns(
        source=pl.lit(result.path),
        label=pl.lit(suffix(result.path)),
        is_primary=pl.lit(primary == result.path),
        key=pl.col("run_id").fill_null(-1 - pl.col("run_index")),
    ).join(agg.rename({"run_id": "key"}).drop("label"), on="key", how="left")


# --------------------------------------------------------------------------- compare


def _same(a: Any, b: Any) -> bool:
    if (
        isinstance(a, float)
        and isinstance(b, float)
        and math.isnan(a)
        and math.isnan(b)
    ):
        return True
    return a == b


def _compare_run(row: Mapping[str, Any], report: VerifyReport) -> None:
    where = {
        "source": row["source"],
        "run_index": row["run_index"],
        "run_id": row["run_id"],
    }
    if row["run_id"] is None:
        report.add("run_missing", "the dataset has no run for this source run", **where)
        return
    best = "ymax" if row["maximization"] else "ymin"
    checks = [
        ("rows", "n", "number of data rows"),
        ("first_evaluation", "first", "first evaluation"),
        ("last_evaluation", "last", "last evaluation"),
        ("best_value", best, "best value"),
    ]
    for check, col, what in checks:
        src, ds = row[col], row[f"{col}_ds"]
        if col == "n":
            src, ds = src or 0, ds or 0
        if not _same(src, ds):
            report.add(check, f"{what} differs", expected=src, found=ds, **where)
    if (row["h_lo"], row["h_hi"]) != (row["h_lo_ds"], row["h_hi_ds"]) and (
        row["n"] or row["n_ds"]
    ):
        report.add("values", "the (evaluations, value) pairs differ", **where)
    if row["is_primary"] and row["line_sha256"] != row["source_rows_sha256"]:
        report.add(
            "source_rows_sha256",
            "the hash of the data lines differs",
            expected=row["line_sha256"],
            found=row["source_rows_sha256"],
            **where,
        )


def _compare(state: _State, dataset: Path) -> None:
    report = state.report
    trace = _trace_aggregates(dataset, state.runs)
    if not state.scans:
        source = pl.DataFrame(schema={"run_id": pl.Int64, "label": pl.String})
    else:
        source = pl.concat(state.scans, how="vertical")
        joined = (
            source.join(trace, on=_KEYS, how="left", suffix="_ds")
            .join(
                state.runs.select("run_id", "maximization", "source_rows_sha256"),
                on="run_id",
                how="left",
            )
            .sort("source", "run_index")
        )
        for row in joined.iter_rows(named=True):
            _compare_run(row, report)
        report.runs = source.filter(pl.col("is_primary")).height
    extra = trace.join(source.select(_KEYS), on=_KEYS, how="anti")
    for rid, label, n in extra.select("run_id", "label", "n").iter_rows():
        report.add(
            "rows_not_in_source",
            f"{n} trace row(s) logged in {label!r} belong to no verified source run",
            run_id=rid,
        )
    _orphan_hashes(state, source)


def _orphan_hashes(state: _State, source: pl.DataFrame) -> None:
    """Runs of a verified file beyond its runs in the source must have no hash."""
    if source.is_empty():
        return
    files = source.filter(pl.col("is_primary")).select("source").unique()
    known = source.filter(pl.col("is_primary")).select("run_id")
    orphans = (
        state.runs.join(files, left_on="source_file", right_on="source")
        .join(known, on="run_id", how="anti")
        .filter(pl.col("source_rows_sha256").is_not_null())
    )
    for rid, path, index in orphans.select(
        "run_id", "source_file", "source_run_index"
    ).iter_rows():
        state.report.add(
            "run_missing_in_source",
            "the dataset has a run with data lines that the source file lacks",
            source=path,
            run_index=index,
            run_id=rid,
        )


def _missing(state: _State) -> None:
    for path, entry in {**state.archives, **state.listed}.items():
        if path in state.seen or entry.get("sha256") is None:
            continue
        state.report.add("source_missing", "the input has no such file", source=path)


def _state(dataset: Path, report: VerifyReport) -> _State:
    manifest = read_manifest(dataset, warn=False)
    runs = pl.read_parquet(
        dataset / "runs.parquet",
        columns=[
            "run_id",
            "source_file",
            "source_run_index",
            "maximization",
            "source_rows_sha256",
        ],
    )
    primaries: dict[str, set[str]] = {}
    for path in runs["source_file"].unique().to_list():
        primaries.setdefault(_stem(path), set()).add(path)
    return _State(
        report,
        manifest,
        runs,
        {s["path"]: s for s in manifest.get("sources", [])},
        {a["path"]: a for a in manifest.get("archives", [])},
        primaries,
        custom_csv_layout(manifest),
    )


def verify(
    source: str | os.PathLike[str],
    dataset: str | os.PathLike[str],
    *,
    validate: bool = True,
) -> VerifyReport:
    """Compare the dataset in ``dataset`` with the input ``source`` it was converted
    from (the same file, folder or archive). Returns a ``VerifyReport``."""
    source, dataset = Path(source), Path(dataset)
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")
    report = VerifyReport(str(source), str(dataset))
    if validate:
        for problem in validate_dataset(dataset).errors:
            report.add("dataset", str(problem))
    state = _state(dataset, report)
    for file in walk(source):
        _visit(file, state)
    _missing(state)
    _compare(state, dataset)
    return report


__all__ = ["Mismatch", "VerifyReport", "verify"]
