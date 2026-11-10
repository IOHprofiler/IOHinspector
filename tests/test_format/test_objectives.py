"""Multiple objectives (H2 question 10): ``raw_y`` is objective 0, ``y1`` ... the rest."""

import math
import shutil

import polars as pl
import pytest

from iohinspector.format import (
    FormatError,
    objective_columns,
    scan_dataset,
    validate_dataset,
    write_dataset,
)
from iohinspector.format import writer as writer_module

from .synthetic import make_dataset, shuffled_chunks

COUNTS = (1, 2, 3)


@pytest.fixture
def small_spills(monkeypatch):
    """Make the writer spill and merge after a few hundred rows."""
    monkeypatch.setattr(writer_module, "_MIN_MEMORY_LIMIT", 1)
    monkeypatch.setattr(writer_module, "_MIN_SPILL_ROWS", 1)
    monkeypatch.setattr(writer_module, "_MIN_BATCH_ROWS", 1)
    monkeypatch.setattr(writer_module, "SPILL_BATCH_BYTES", 7 * 64)


def multi_objective(seed=0, rows=lambda r: r.randint(0, 12)):
    """A dataset whose runs have 1, 2 and 3 objectives, mixed in every file.

    ``y1`` and ``y2`` hold values (NaN, inf and -0.0 included) up to each run's
    count and null beyond it.
    """
    syn = make_dataset(functions=(1, 2), dimensions=(2, 3), runs_per_file=3, rows=rows)
    syn.runs = syn.runs.with_columns(
        n_objectives=pl.int_range(pl.len()).mod(3).add(1).cast(pl.Int32)
    )
    count = syn.trace.join(syn.runs.select("run_id", "n_objectives"), on="run_id")
    special = pl.when(pl.col("row") % 5 == 1).then(float("nan"))
    special = special.when(pl.col("row") % 5 == 2).then(float("-inf"))
    special = special.when(pl.col("row") % 5 == 3).then(-0.0)
    syn.trace = count.with_columns(
        pl.when(pl.col("n_objectives") > j)
        .then(special.otherwise(pl.col("raw_y") * (j + 1) + pl.col("row")))
        .otherwise(None)
        .cast(pl.Float64)
        .alias(f"y{j}")
        for j in (1, 2)
    ).drop("n_objectives")
    return syn


def write(path, syn, trace=None, **kw):
    return write_dataset(
        path,
        syn.manifest,
        syn.runs,
        syn.trace if trace is None else trace,
        descriptors=syn.descriptors,
        **kw,
    )


def same(a, b):
    """Equal float values, with NaN equal to NaN and the sign of zero kept."""
    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b and math.copysign(1, a) == math.copysign(1, b)


def test_objective_columns():
    assert objective_columns(1) == [] and objective_columns(0) == []
    assert objective_columns(3) == ["y1", "y2"]


def test_mixed_objective_counts_round_trip(tmp_path, small_spills):
    syn = multi_objective(seed=1)
    assert sorted(syn.runs["n_objectives"].unique()) == list(COUNTS)
    write(tmp_path / "ds", syn, shuffled_chunks(syn.trace, 9), memory_limit=1)
    report = validate_dataset(tmp_path / "ds")
    assert report.ok, report.errors

    ds = scan_dataset(tmp_path / "ds")
    trace = ds.trace.collect()
    fixed = ["run_id", "function_id", "dimension", "row", "evaluations", "raw_y"]
    assert trace.columns[:8] == [*fixed, "y1", "y2"]
    assert trace.schema["y1"] == trace.schema["y2"] == pl.Float64
    assert "y0" not in trace.columns
    assert trace.height == syn.trace.height

    expected = {(r["run_id"], r["row"]): r for r in syn.trace.iter_rows(named=True)}
    counts = dict(ds.runs.select("run_id", "n_objectives").collect().iter_rows())
    for row in trace.iter_rows(named=True):
        source = expected[(row["run_id"], row["row"])]
        for col in ("raw_y", "y1", "y2", "sigma"):
            assert same(row[col], source[col]), col
        assert (row["restart"], row["tag"]) == (source["restart"], source["tag"])
        n = counts[row["run_id"]]
        assert (row["y1"] is None) == (n < 2) and (row["y2"] is None) == (n < 3)


def test_chunks_may_leave_out_objectives_their_runs_do_not_need(tmp_path):
    syn = multi_objective(seed=2)
    n = syn.trace.join(syn.runs.select("run_id", "n_objectives"), on="run_id")
    chunks = []
    for count in COUNTS:
        part = n.filter(pl.col("n_objectives") == count).drop("n_objectives")
        chunks.append(part.drop([f"y{j}" for j in (1, 2) if j >= count]))
    assert any("y1" not in c.columns for c in chunks)
    write(tmp_path / "a", syn)
    write(tmp_path / "b", syn, chunks)
    a = scan_dataset(tmp_path / "a").trace.collect()
    b = scan_dataset(tmp_path / "b").trace.collect()
    assert a.columns == b.columns
    assert a.select("run_id", "row").equals(b.select("run_id", "row"))
    for col in ("y1", "y2"):
        assert [same(x, y) for x, y in zip(a[col], b[col])] == [True] * a.height


def test_largest_count_sets_the_columns(tmp_path):
    syn = multi_objective(seed=3)
    syn.runs = syn.runs.with_columns(
        pl.col("n_objectives").clip(upper_bound=2).cast(pl.Int32)
    )
    write(tmp_path / "ds", syn, syn.trace.drop("y2"))
    trace = scan_dataset(tmp_path / "ds").trace.collect()
    assert "y1" in trace.columns and "y2" not in trace.columns
    assert validate_dataset(tmp_path / "ds").ok


def _set_at(run_index, column, value):
    """Change ``column`` in every trace row of the run at ``run_index``."""

    def change(syn):
        rid = syn.runs.filter(pl.col("n_objectives") == run_index)["run_id"][0]
        return syn.trace.with_columns(
            pl.when(pl.col("run_id") == rid)
            .then(pl.lit(value, pl.Float64))
            .otherwise(pl.col(column))
            .alias(column)
        )

    return change


@pytest.mark.parametrize(
    "mutate, message",
    [
        (_set_at(1, "y1", 1.0), "objective values whose nulls do not match"),
        (_set_at(2, "y2", 1.0), "objective values whose nulls do not match"),
        (_set_at(3, "y2", None), "objective values whose nulls do not match"),
        (lambda s: s.trace.drop("y2"), "objective values whose nulls do not match"),
        (lambda s: s.trace.with_columns(pl.lit(1.0).alias("y3")), "y1 to y2"),
        (lambda s: s.trace.with_columns(pl.lit(1.0).alias("y0")), "no y0"),
        (lambda s: s.trace.with_columns(pl.col("y1").cast(pl.String)), "y1"),
    ],
)
def test_bad_objectives_are_refused(tmp_path, mutate, message):
    syn = multi_objective(seed=4, rows=lambda r: r.randint(2, 6))
    with pytest.raises(FormatError, match=message):
        write(tmp_path / "ds", syn, mutate(syn))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("count", [0, -1])
def test_n_objectives_must_be_at_least_one(tmp_path, count):
    syn = multi_objective(seed=5)
    syn.runs = syn.runs.with_columns(pl.lit(count).alias("n_objectives"))
    with pytest.raises(FormatError, match="n_objectives that is not at least 1"):
        write(tmp_path / "ds", syn, syn.trace.drop("y1", "y2"))


# -- validation -----------------------------------------------------------------


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    path = tmp_path_factory.mktemp("objectives") / "ds"
    write(path, multi_objective(seed=6, rows=lambda r: r.randint(2, 6)))
    return path


@pytest.fixture
def ds(stored, tmp_path):
    shutil.copytree(stored, tmp_path / "ds")
    return tmp_path / "ds"


def _first_with(path, count):
    runs = pl.read_parquet(path / "runs.parquet")
    return runs.filter(pl.col("n_objectives") == count)["run_id"][0]


def _rewrite_trace(path, change):
    file = path / "trace.parquet"
    change(pl.read_parquet(file)).write_parquet(file, compression="zstd")


def _set_in_trace(path, count, column, value):
    rid = _first_with(path, count)
    _rewrite_trace(
        path,
        lambda df: df.with_columns(
            pl.when(pl.col("run_id") == rid)
            .then(pl.lit(value, pl.Float64))
            .otherwise(pl.col(column))
            .alias(column)
        ),
    )


def _errors(path):
    report = validate_dataset(path)
    assert not report.ok
    return "\n".join(report.errors)


def test_stored_dataset_is_valid(stored):
    report = validate_dataset(stored)
    assert report.ok, report.errors
    report = validate_dataset(stored, batch_rows=3)
    assert report.ok, report.errors


@pytest.mark.parametrize(
    "count, column, value", [(1, "y1", 2.0), (2, "y2", 2.0), (3, "y1", None)]
)
def test_validator_checks_objective_nulls(ds, count, column, value):
    _set_in_trace(ds, count, column, value)
    assert "objective values whose nulls do not match the run's n_objectives" in (
        _errors(ds)
    )


@pytest.mark.parametrize(
    "change, fragment",
    [
        (lambda df: df.with_columns(pl.col("y1").cast(pl.Float32)), "'y1' has type"),
        (
            lambda df: df.with_columns(pl.col("y2").cast(pl.String)),
            "'y2' has type",
        ),
        (lambda df: df.drop("y2"), "column 'y2' is missing"),
        (lambda df: df.with_columns(pl.lit(None, pl.Float64).alias("y3")), "y3"),
        (lambda df: df.with_columns(pl.col("raw_y").alias("y0")), "'y0'"),
    ],
)
def test_validator_checks_objective_columns(ds, change, fragment):
    _rewrite_trace(ds, change)
    assert fragment in _errors(ds)


def test_validator_needs_columns_for_the_largest_count(ds):
    file = ds / "runs.parquet"
    runs = pl.read_parquet(file)
    runs.with_columns(
        pl.when(pl.col("n_objectives") == 3)
        .then(4)
        .otherwise(pl.col("n_objectives"))
        .cast(pl.Int32)
        .alias("n_objectives")
    ).write_parquet(file, compression="zstd")
    assert "column 'y3' is missing" in _errors(ds)
