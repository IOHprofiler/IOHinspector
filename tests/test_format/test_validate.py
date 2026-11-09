import json
import shutil

import polars as pl
import pyarrow.parquet as pq
import pytest

from iohinspector.format import (
    DatasetValidationError,
    run_id,
    validate_dataset,
    write_dataset,
)

from .synthetic import make_dataset


@pytest.fixture(scope="module")
def valid_dataset(tmp_path_factory):
    syn = make_dataset(
        dimensions=(2, 3), positions=True, rows=lambda r: r.randint(2, 8), seed=20
    )
    path = tmp_path_factory.mktemp("valid") / "ds"
    write_dataset(path, syn.manifest, syn.runs, syn.trace, syn.positions)
    return path


@pytest.fixture
def ds(valid_dataset, tmp_path):
    """A fresh copy of the valid dataset to corrupt."""
    path = tmp_path / "ds"
    shutil.copytree(valid_dataset, path)
    return path


def rewrite(path, table, change):
    """Apply ``change`` to a table and write it back (with polars, zstd)."""
    file = path / f"{table}.parquet"
    df = change(pl.read_parquet(file))
    df.write_parquet(file, compression="zstd")


def edit_manifest(path, change):
    file = path / "manifest.json"
    manifest = json.loads(file.read_text())
    change(manifest)
    file.write_text(json.dumps(manifest))


def assert_error(path, fragment, **kw):
    report = validate_dataset(path, **kw)
    assert not report.ok, "expected an error"
    assert any(fragment in e for e in report.errors), report.errors
    return report


def test_valid_dataset_passes(valid_dataset):
    report = validate_dataset(valid_dataset)
    assert report.ok, report.errors
    assert report.warnings == []
    report.raise_for_errors()


def test_small_batches_give_the_same_result(valid_dataset):
    assert validate_dataset(valid_dataset, batch_rows=3).ok


def test_report_raises_and_prints(ds):
    (ds / "runs.parquet").unlink()
    report = validate_dataset(ds)
    with pytest.raises(DatasetValidationError) as exc:
        report.raise_for_errors()
    assert exc.value.report is report
    assert "INVALID" in str(report) and "runs.parquet is missing" in str(report)


# -- manifest and files ---------------------------------------------------------


def test_missing_manifest(ds):
    (ds / "manifest.json").unlink()
    assert_error(ds, "manifest.json is missing")


def test_manifest_not_json(ds):
    (ds / "manifest.json").write_text("{")
    assert_error(ds, "not valid JSON")


def test_unknown_major_version(ds):
    edit_manifest(ds, lambda m: m.update(format_version="2.0"))
    report = assert_error(ds, "not supported")
    assert len(report.errors) == 1  # nothing else is checked


def test_manifest_schema(ds):
    edit_manifest(ds, lambda m: m["sources"][0].update(sha256="xyz"))
    assert_error(ds, "sources/0/sha256")


@pytest.mark.parametrize("key", ["runs", "trace_rows", "positions_rows"])
def test_manifest_counts(ds, key):
    edit_manifest(ds, lambda m: m["counts"].update({key: m["counts"][key] + 1}))
    assert_error(ds, f"counts.{key}")


def test_positions_count_without_file(ds):
    (ds / "positions.parquet").unlink()
    assert_error(ds, "positions_rows must be present exactly when")


def test_unexpected_file_is_a_warning(ds):
    (ds / "notes.txt").write_text("hi")
    report = validate_dataset(ds)
    assert report.ok and any("notes.txt" in w for w in report.warnings)


def test_newer_minor_version_allows_unknown_columns(ds):
    edit_manifest(ds, lambda m: m.update(format_version="1.4", new_field=1))
    rewrite(ds, "runs", lambda df: df.with_columns(pl.lit(1).alias("future")))
    report = validate_dataset(ds)
    assert report.ok, report.errors
    assert any("future" in w for w in report.warnings)
    edit_manifest(ds, lambda m: m.update(format_version="1.0"))
    assert_error(ds, "unknown column")


# -- runs -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "change, fragment",
    [
        (lambda df: df.drop("suite"), "column 'suite' is missing"),
        (lambda df: df.with_columns(pl.col("function_id").cast(pl.Int64)), "has type"),
        (lambda df: df.with_columns(pl.lit(1).alias("seed")), "unknown column"),
        (lambda df: df.with_columns(pl.lit(None, pl.String).alias("suite")), "null"),
        (lambda df: df.with_columns(pl.col("run_id") + 1), "stable hash"),
        (lambda df: df.reverse(), "run order"),
        (
            lambda df: df.with_columns(pl.lit(2, pl.Int32).alias("n_objectives")),
            "n_objectives",
        ),
        (lambda df: pl.concat([df, df.tail(1)]), "not unique"),
    ],
)
def test_runs_problems(ds, change, fragment):
    rewrite(ds, "runs", change)
    assert_error(ds, fragment)


def test_runs_source_file_not_in_manifest(ds):
    edit_manifest(ds, lambda m: m.update(sources=m["sources"][1:]))
    assert_error(ds, "not listed in the manifest")


def test_runs_with_another_dataset_id(ds):
    edit_manifest(ds, lambda m: m.update(dataset_id="renamed"))
    assert_error(ds, "stable hash")


# -- trace ----------------------------------------------------------------------


def _second_run(df):
    return df["run_id"].unique(maintain_order=True)[1]


@pytest.mark.parametrize(
    "change, fragment",
    [
        (
            lambda df: df.with_columns(pl.col("run_id").shift(1, fill_value=7)),
            "not in runs",
        ),
        (
            lambda df: df.with_columns(pl.lit(9, pl.Int32).alias("dimension")),
            "differs from their run",
        ),
        (lambda df: df.reverse(), "the first row is"),
        (lambda df: df.filter(pl.col("row") != 1), "no gaps"),
        (lambda df: pl.concat([df.head(1), df]), "twice"),
        (lambda df: df.with_columns(pl.col("row") + 1), "start at 0"),
        (
            lambda df: df.with_columns(
                pl.when(pl.col("row") == 1)
                .then(None)
                .otherwise(pl.col("raw_y"))
                .alias("raw_y")
            ),
            "null",
        ),
        (lambda df: df.with_columns(pl.lit(1.0).alias("y0")), "reserved"),
        (
            lambda df: df.with_columns(pl.lit(1.0, pl.Float32).alias("f32")),
            "not allowed",
        ),
        (lambda df: df.drop("raw_y"), "column 'raw_y' is missing"),
        (
            lambda df: df.with_columns(pl.col("evaluations").cast(pl.Float64)),
            "has type",
        ),
    ],
)
def test_trace_problems(ds, change, fragment):
    rewrite(ds, "trace", change)
    assert_error(ds, fragment)


def test_trace_order_across_batches(ds):
    def swap_runs(df):
        first, second = df["run_id"].unique(maintain_order=True)[:2]
        a = df.filter(pl.col("run_id") == first)
        b = df.filter(pl.col("run_id") == second)
        rest = df.filter(~pl.col("run_id").is_in([first, second]))
        return pl.concat([b, a, rest])

    rewrite(ds, "trace", swap_runs)
    for batch_rows in (1, 2, 1000):
        assert_error(ds, "run order", batch_rows=batch_rows)


def test_storage_warnings(ds):
    file = ds / "trace.parquet"
    pq.write_table(
        pq.read_table(file).replace_schema_metadata(None), file, compression="snappy"
    )
    report = validate_dataset(ds)
    assert report.ok, report.errors
    text = "\n".join(report.warnings)
    assert "not zstd" in text and "key-value metadata" in text


def test_missing_statistics_warn(ds):
    file = ds / "trace.parquet"
    pq.write_table(
        pq.read_table(file), file, compression="zstd", write_statistics=False
    )
    report = validate_dataset(ds)
    assert report.ok and any("statistics" in w for w in report.warnings)


# -- positions ------------------------------------------------------------------


@pytest.mark.parametrize(
    "change, fragment",
    [
        (lambda df: df.with_columns(pl.col("row") + 1000), "no trace row"),
        (lambda df: df.with_columns(pl.lit(1.0).alias("x2")), "nulls do not match"),
        (
            lambda df: df.with_columns(pl.lit(None, pl.Float64).alias("x0")),
            "nulls do not match",
        ),
        (lambda df: df.reverse(), "positions: "),
        (lambda df: df.with_columns(pl.lit(1.0).alias("x3")), "unexpected column"),
        (lambda df: df.drop("x2"), "column 'x2' is missing"),
        (lambda df: pl.concat([df.head(1), df]), "twice"),
    ],
)
def test_positions_problems(ds, change, fragment):
    rewrite(ds, "positions", change)
    assert_error(ds, fragment)


def test_error_limit(ds):
    rewrite(
        ds, "trace", lambda df: df.with_columns(pl.lit(9, pl.Int32).alias("dimension"))
    )
    rewrite(
        ds,
        "runs",
        lambda df: df.with_columns(pl.lit(2, pl.Int32).alias("n_objectives")),
    )
    edit_manifest(ds, lambda m: m["counts"].update(runs=0, trace_rows=0))
    report = validate_dataset(ds, max_errors=2)
    assert len(report.errors) == 3 and report.errors[-1] == "stopped after 2 errors"


def test_not_a_directory(tmp_path):
    assert_error(tmp_path / "nothing", "is not a directory")


def test_run_id_check_uses_the_recipe(ds):
    runs = pl.read_parquet(ds / "runs.parquet")
    r = runs.row(0, named=True)
    manifest = json.loads((ds / "manifest.json").read_text())
    assert r["run_id"] == run_id(
        manifest["dataset_id"], r["source_file"], r["source_run_index"]
    )


def test_unreadable_parquet(ds):
    (ds / "trace.parquet").write_bytes(b"not parquet")
    assert_error(ds, "not a readable Parquet file")
