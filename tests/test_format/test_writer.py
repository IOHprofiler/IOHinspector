import math

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest
from polars.testing import assert_frame_equal

from iohinspector.format import (
    RUNS_SCHEMA,
    TRACE_SCHEMA,
    FormatError,
    ManifestError,
    new_manifest,
    run_ids,
    scan_dataset,
    write_dataset,
)
from iohinspector.format import writer as writer_module

from .synthetic import DATASET_ID, make_dataset, shuffled_chunks

SORTED_TRACE = ["run_id", "row"]


@pytest.fixture
def small_spills(monkeypatch):
    """Make the writer spill and merge after a few hundred rows."""
    monkeypatch.setattr(writer_module, "_MIN_MEMORY_LIMIT", 1)
    monkeypatch.setattr(writer_module, "_MIN_SPILL_ROWS", 1)
    monkeypatch.setattr(writer_module, "_MIN_BATCH_ROWS", 1)
    monkeypatch.setattr(writer_module, "SPILL_BATCH_BYTES", 7 * 64)


def expected_trace(syn):
    """The trace the format requires: run order, then row; with the copied run columns."""
    runs = syn.runs.sort(
        ["function_id", "dimension", "source_file", "source_run_index"]
    )
    order = runs.select(
        "run_id", "function_id", "dimension", pl.int_range(pl.len()).alias("_o")
    )
    out = syn.trace.join(order, on="run_id").sort("_o", "row").drop("_o")
    return out.select(
        "run_id",
        "function_id",
        "dimension",
        *[c for c in syn.trace.columns if c != "run_id"],
    )


def assert_same_values(actual: pl.DataFrame, expected: pl.DataFrame):
    """Equal, with NaN equal to NaN and -0.0 kept distinct from 0.0."""
    assert actual.columns == expected.columns
    assert_frame_equal(actual, expected, check_dtypes=False)
    for col in actual.columns:
        if actual[col].dtype == pl.Float64:
            a = actual[col].to_numpy()
            e = expected[col].to_numpy()
            np.testing.assert_array_equal(np.signbit(a), np.signbit(e))


def test_round_trip(tmp_path):
    syn = make_dataset(seed=1)
    manifest = write_dataset(
        tmp_path / "ds", syn.manifest, syn.runs, syn.trace, descriptors=syn.descriptors
    )
    ds = scan_dataset(tmp_path / "ds")
    assert ds.manifest == manifest
    assert manifest["counts"] == {
        "runs": syn.runs.height,
        "descriptors_rows": syn.descriptors.height + 2 * syn.runs.height,
        "trace_rows": syn.trace.height,
    }
    assert sorted(manifest["descriptors"]) == [
        "algorithm_info",
        "algorithm_name",
        "folder",
        "popsize",
        "seed",
    ]

    runs = ds.runs.collect()
    assert dict(runs.schema) == RUNS_SCHEMA
    assert runs.height == syn.runs.height
    joined = syn.runs.join(runs, on="run_id", suffix="_stored")
    for col in syn.runs.columns:
        if col != "run_id":
            assert (joined[col] == joined[f"{col}_stored"]).all(), col
    assert (runs["suite"] == "").all() and (runs["n_objectives"] == 1).all()
    assert runs["budget"].is_null().all()

    trace = ds.trace.collect()
    assert {k: trace.schema[k] for k in TRACE_SCHEMA} == TRACE_SCHEMA
    assert trace.schema["sigma"] == pl.Float64 and trace.schema["restart"] == pl.Int64
    assert trace.schema["tag"] == pl.String
    assert_same_values(trace, expected_trace(syn))


def test_runs_are_sorted_by_suite_function_dimension_and_source(tmp_path):
    syn = make_dataset(
        functions=(3, 1), dimensions=(10, 2), algorithms=("zeta", "alpha"), seed=2
    )
    write_dataset(tmp_path / "ds", syn.manifest, syn.runs, syn.trace)
    runs = scan_dataset(tmp_path / "ds").runs.collect()
    keys = runs.select(
        "function_id", "dimension", "source_file", "source_run_index"
    ).rows()
    assert keys == sorted(keys)


def test_shuffled_chunks_give_the_same_dataset_as_sorted_input(tmp_path, small_spills):
    syn = make_dataset(
        functions=(1, 2, 3),
        dimensions=(2, 3, 5),
        runs_per_file=3,
        rows=lambda r: r.randint(0, 40),
    )
    write_dataset(tmp_path / "a", syn.manifest, syn.runs, syn.trace)
    write_dataset(
        tmp_path / "b",
        syn.manifest,
        syn.runs,
        shuffled_chunks(syn.trace, 17),
        memory_limit=1,
    )
    a = scan_dataset(tmp_path / "a").trace.collect()
    b = scan_dataset(tmp_path / "b").trace.collect()
    assert_same_values(b, a)
    assert_same_values(a, expected_trace(syn))


def test_external_sort_uses_several_spill_files(tmp_path, small_spills, monkeypatch):
    seen = []
    original = writer_module._merge

    def spy(readers):
        seen.append(len(readers))
        yield from original(readers)

    monkeypatch.setattr(writer_module, "_merge", spy)
    syn = make_dataset(rows=lambda r: r.randint(20, 60))
    write_dataset(
        tmp_path / "ds",
        syn.manifest,
        syn.runs,
        shuffled_chunks(syn.trace, 9),
        memory_limit=1,
    )
    assert seen and seen[0] > 2
    assert_same_values(
        scan_dataset(tmp_path / "ds").trace.collect(), expected_trace(syn)
    )


def test_lazy_and_iterable_inputs(tmp_path):
    syn = make_dataset(seed=3)
    write_dataset(tmp_path / "lazy", syn.manifest, syn.runs.lazy(), syn.trace.lazy())
    write_dataset(
        tmp_path / "iter",
        syn.manifest,
        syn.runs,
        (c.lazy() for c in shuffled_chunks(syn.trace, 4)),
    )
    for name in ("lazy", "iter"):
        assert_same_values(
            scan_dataset(tmp_path / name).trace.collect(), expected_trace(syn)
        )


def test_scanned_dataset_writes_back_identically(tmp_path):
    syn = make_dataset(positions=True, seed=4)
    write_dataset(
        tmp_path / "a",
        syn.manifest,
        syn.runs,
        syn.trace,
        positions=syn.positions,
        descriptors=syn.descriptors,
    )
    ds = scan_dataset(tmp_path / "a")
    write_dataset(
        tmp_path / "b",
        ds.manifest,
        ds.runs,
        ds.trace,
        positions=ds.positions,
        descriptors=ds.descriptors,
    )
    again = scan_dataset(tmp_path / "b")
    assert again.manifest["descriptors"] == ds.manifest["descriptors"]
    for table in ("runs", "descriptors", "trace", "positions"):
        assert_same_values(
            getattr(again, table).collect(), getattr(ds, table).collect()
        )


def test_lossless_values(tmp_path):
    syn = make_dataset(seed=5)
    values = [
        0.0,
        -0.0,
        math.inf,
        -math.inf,
        math.nan,
        5e-324,
        1.7976931348623157e308,
        0.1 + 0.2,
    ]
    run = syn.runs["run_id"][0]
    trace = pl.DataFrame(
        {
            "run_id": [run] * len(values),
            "row": range(len(values)),
            "evaluations": [1] * len(values),
            "raw_y": values,
        }
    )
    write_dataset(tmp_path / "ds", syn.manifest, syn.runs, trace)
    stored = scan_dataset(tmp_path / "ds").trace.collect()
    got = stored["raw_y"].to_list()
    assert [v.hex() for v in got if not math.isnan(v)] == [
        v.hex() for v in values if not math.isnan(v)
    ]
    assert math.isnan(got[4]) and np.signbit(got[1])
    assert stored["evaluations"].to_list() == [1] * len(
        values
    )  # repeated evaluations are kept


def test_runs_without_rows_and_an_empty_trace(tmp_path):
    syn = make_dataset(rows=lambda r: 0, extras=False)
    write_dataset(tmp_path / "ds", syn.manifest, syn.runs, syn.trace)
    ds = scan_dataset(tmp_path / "ds")
    assert ds.runs.collect().height == syn.runs.height
    trace = ds.trace.collect()
    assert trace.height == 0 and list(trace.columns) == list(TRACE_SCHEMA)


def test_positions_round_trip(tmp_path):
    syn = make_dataset(dimensions=(2, 4), positions=True, seed=6)
    m = write_dataset(
        tmp_path / "ds",
        syn.manifest,
        syn.runs,
        syn.trace,
        positions=shuffled_chunks(syn.positions, 3),
    )
    ds = scan_dataset(tmp_path / "ds")
    pos = ds.positions.collect()
    assert m["counts"]["positions_rows"] == syn.positions.height == pos.height
    assert pos.columns == [
        "run_id",
        "function_id",
        "dimension",
        "row",
        "x0",
        "x1",
        "x2",
        "x3",
    ]
    d2 = pos.filter(pl.col("dimension") == 2)
    assert d2["x2"].is_null().all() and d2["x1"].is_not_null().all()
    order = ds.runs.collect().with_row_index("o").select("run_id", "o")
    keys = pos.join(order, on="run_id").select("o", "row").rows()
    assert keys == sorted(keys)


def test_positions_chunks_may_leave_out_high_coordinates(tmp_path):
    syn = make_dataset(dimensions=(2, 3), positions=True, seed=7)
    pos = syn.positions.join(syn.runs.select("run_id", "dimension"), on="run_id")
    low = pos.filter(pl.col("dimension") == 2).select("run_id", "row", "x0", "x1")
    high = pos.filter(pl.col("dimension") == 3).drop("dimension")
    write_dataset(
        tmp_path / "ds", syn.manifest, syn.runs, syn.trace, positions=[low, high]
    )
    assert (
        scan_dataset(tmp_path / "ds").positions.collect().height == syn.positions.height
    )


def test_extra_columns_declared_up_front(tmp_path):
    syn = make_dataset(seed=8)
    first = syn.trace.slice(0, 5).drop("sigma", "tag")
    rest = syn.trace.slice(5)
    extra = {"sigma": pl.Float64, "restart": pl.Int64, "tag": pl.String}
    write_dataset(
        tmp_path / "ds", syn.manifest, syn.runs, [first, rest], extra_columns=extra
    )
    trace = scan_dataset(tmp_path / "ds").trace.collect()
    assert trace.columns[-3:] == ["sigma", "restart", "tag"]
    assert trace.height == syn.trace.height

    with pytest.raises(FormatError, match="extra_columns"):
        write_dataset(tmp_path / "ds2", syn.manifest, syn.runs, [first, rest])


def test_integer_and_float32_inputs_are_widened_without_change(tmp_path):
    syn = make_dataset(extras=False, seed=9)
    trace = syn.trace.with_columns(
        pl.col("row").cast(pl.UInt32),
        pl.col("evaluations").cast(pl.Int32),
        pl.lit(3, pl.Int16).alias("raw_y"),
    )
    runs = syn.runs.with_columns(
        pl.col("function_id").cast(pl.UInt8), pl.col("dimension").cast(pl.Int64)
    )
    write_dataset(tmp_path / "ds", syn.manifest, runs, trace)
    stored = scan_dataset(tmp_path / "ds").trace.collect()
    assert stored.schema["raw_y"] == pl.Float64 and (stored["raw_y"] == 3.0).all()


def test_row_groups_start_at_function_and_dimension_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(writer_module, "MIN_ROW_GROUP_ROWS", 50)
    monkeypatch.setattr(writer_module, "MAX_ROW_GROUP_ROWS", 400)
    monkeypatch.setattr(writer_module, "_MIN_BATCH_ROWS", 1)
    syn = make_dataset(
        functions=(1, 2, 3),
        dimensions=(2, 5),
        runs_per_file=4,
        rows=lambda r: 30,
        extras=False,
    )
    write_dataset(tmp_path / "ds", syn.manifest, syn.runs, syn.trace)
    meta = pq.ParquetFile(tmp_path / "ds" / "trace.parquet").metadata
    assert meta.num_row_groups > 1
    names = [meta.schema.column(i).name for i in range(meta.num_columns)]
    f, d = names.index("function_id"), names.index("dimension")
    for i in range(meta.num_row_groups):
        rg = meta.row_group(i)
        assert rg.num_rows <= 400
        stats = rg.column(f).statistics, rg.column(d).statistics
        assert all(s is not None and s.has_min_max for s in stats)
        assert (
            stats[0].min == stats[0].max and stats[1].min == stats[1].max
        )  # 240 rows per group
        assert rg.column(0).compression == "ZSTD"
    kv = meta.metadata
    assert (
        kv[b"iohinspector.format_version"] == b"1.0"
        and kv[b"iohinspector.table"] == b"trace"
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda s: s.trace.with_columns(pl.col("run_id") + 1), "not in runs"),
        (lambda s: pl.concat([s.trace, s.trace.head(1)]), "twice"),
        (lambda s: s.trace.filter(pl.col("row") != 1), "no gaps"),
        (lambda s: s.trace.with_columns(pl.col("row") + 1), "start at 0"),
        (
            lambda s: s.trace.with_columns(pl.lit(None, pl.Float64).alias("raw_y")),
            "null",
        ),
        (
            lambda s: s.trace.with_columns(pl.col("raw_y").cast(pl.String)),
            "cannot be stored",
        ),
        (lambda s: s.trace.with_columns(pl.lit(2**53 + 1).alias("raw_y")), "exactly"),
        (
            lambda s: s.trace.with_columns(pl.lit(99, pl.Int32).alias("function_id")),
            "differs",
        ),
        (lambda s: s.trace.with_columns(pl.lit(1.0).alias("y0")), "reserved"),
        (lambda s: s.trace.with_columns(pl.lit(1.0).alias("instance")), "reserved"),
        (lambda s: s.trace.with_columns(pl.lit(None).alias("empty")), "no type"),
        (lambda s: s.trace.drop("evaluations"), "missing"),
    ],
)
def test_bad_trace_is_refused(tmp_path, mutate, message):
    syn = make_dataset(rows=lambda r: r.randint(3, 5), seed=10)
    with pytest.raises(FormatError, match=message):
        write_dataset(tmp_path / "ds", syn.manifest, syn.runs, mutate(syn))
    assert list(tmp_path.iterdir()) == []  # the temporary directory is removed


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r.with_columns(pl.lit(1).alias("seed")), "unknown column"),
        (lambda r: r.drop("maximization"), "missing required"),
        (lambda r: r.with_columns(pl.col("run_id") + 1), "stable hash"),
        (lambda r: r.with_columns(pl.lit(None, pl.String).alias("suite")), "null"),
        (lambda r: r.with_columns(pl.lit(2).alias("n_objectives")), "n_objectives"),
        (lambda r: r.with_columns(pl.lit(0).alias("dimension")), "dimension"),
        (lambda r: r.with_columns(pl.lit("{}").alias("attributes")), "descriptors"),
        (lambda r: r.with_columns(pl.lit("abc").alias("source_rows_sha256")), "sha256"),
        (lambda r: r.with_columns(pl.lit(2**40).alias("function_id")), "do not fit"),
        (lambda r: pl.concat([r, r.head(1)]), "not unique"),
    ],
)
def test_bad_runs_are_refused(tmp_path, mutate, message):
    syn = make_dataset(seed=11)
    with pytest.raises(FormatError, match=message):
        write_dataset(tmp_path / "ds", syn.manifest, mutate(syn.runs), syn.trace)


def test_source_file_must_be_in_the_manifest(tmp_path):
    syn = make_dataset(seed=12)
    syn.manifest["sources"] = syn.manifest["sources"][1:]
    with pytest.raises(FormatError, match="not listed in the manifest"):
        write_dataset(tmp_path / "ds", syn.manifest, syn.runs, syn.trace)


def test_invalid_manifest_is_refused_before_writing(tmp_path):
    syn = make_dataset(seed=13)
    syn.manifest["sources"][0]["sha256"] = "nope"
    with pytest.raises(ManifestError):
        write_dataset(tmp_path / "ds", syn.manifest, syn.runs, syn.trace)
    assert not (tmp_path / "ds").exists()


def test_caller_manifest_is_not_changed(tmp_path):
    syn = make_dataset(seed=14)
    before = repr(syn.manifest)
    write_dataset(tmp_path / "ds", syn.manifest, syn.runs, syn.trace)
    assert repr(syn.manifest) == before


def test_existing_dataset_needs_overwrite(tmp_path):
    syn = make_dataset(seed=15)
    write_dataset(tmp_path / "ds", syn.manifest, syn.runs, syn.trace)
    with pytest.raises(FileExistsError):
        write_dataset(tmp_path / "ds", syn.manifest, syn.runs, syn.trace)
    smaller = syn.trace.filter(pl.col("row") < 2)
    write_dataset(tmp_path / "ds", syn.manifest, syn.runs, smaller, overwrite=True)
    assert scan_dataset(tmp_path / "ds").trace.collect().height == smaller.height
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ds"]


def test_failed_overwrite_keeps_the_old_dataset(tmp_path):
    syn = make_dataset(seed=16)
    write_dataset(tmp_path / "ds", syn.manifest, syn.runs, syn.trace)
    with pytest.raises(FormatError):
        write_dataset(
            tmp_path / "ds",
            syn.manifest,
            syn.runs,
            syn.trace.with_columns(pl.col("row") + 1),
            overwrite=True,
        )
    assert scan_dataset(tmp_path / "ds").trace.collect().height == syn.trace.height
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ds"]


def test_overwrite_refuses_a_directory_that_is_not_a_dataset(tmp_path):
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "notes.txt").write_text("keep me")
    syn = make_dataset(seed=17)
    with pytest.raises(FileExistsError, match="not a dataset"):
        write_dataset(
            tmp_path / "ds", syn.manifest, syn.runs, syn.trace, overwrite=True
        )
    assert (tmp_path / "ds" / "notes.txt").read_text() == "keep me"


def test_minimal_runs_get_defaults(tmp_path):
    manifest = new_manifest(
        "minimal",
        "test",
        "0",
        sources=[{"path": "a.dat", "sha256": None, "bytes": None}],
    )
    runs = pl.DataFrame(
        {
            "algorithm_name": ["A"],
            "function_id": [1],
            "dimension": [2],
            "maximization": [False],
            "source_file": ["a.dat"],
            "source_run_index": [0],
        }
    )
    rid = run_ids("minimal", ["a.dat"], [0])[0]
    trace = pl.DataFrame(
        {"run_id": [rid], "row": [0], "evaluations": [1], "raw_y": [1.5]}
    )
    write_dataset(tmp_path / "ds", manifest, runs, trace)
    stored = scan_dataset(tmp_path / "ds").runs.collect().row(0, named=True)
    assert stored["run_id"] == rid and stored["algorithm_info"] == ""
    assert stored["instance"] is None and stored["n_objectives"] == 1


def test_dataset_id_changes_run_ids(tmp_path):
    a = make_dataset(dataset_id="first", seed=18)
    b = make_dataset(dataset_id="second", seed=18)
    assert set(a.runs["run_id"]).isdisjoint(set(b.runs["run_id"]))
    assert DATASET_ID not in ("first", "second")


def test_strings_sort_by_utf8_bytes(tmp_path):
    names = ["b", "B", "é", "a", "Z", "aa", "😀", "￿"]
    syn = make_dataset(
        algorithms=names, functions=(1,), dimensions=(2,), runs_per_file=1
    )
    write_dataset(tmp_path / "ds", syn.manifest, syn.runs, syn.trace)
    stored = scan_dataset(tmp_path / "ds").runs.collect()["algorithm_name"].to_list()
    assert stored == sorted(names, key=lambda s: s.encode("utf-8"))
