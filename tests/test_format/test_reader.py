import json
import shutil

import polars as pl
import pyarrow.parquet as pq
import pytest
from polars.testing import assert_frame_equal

from iohinspector.format import (
    DatasetWarning,
    FormatWarning,
    ManifestError,
    UnsupportedFormatVersion,
    scan_dataset,
    write_dataset,
)
from iohinspector.format import writer as writer_module

from .synthetic import make_dataset


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    syn = make_dataset(positions=True, seed=30)
    syn.manifest["warnings"] = [
        {
            "code": "limited_precision",
            "message": "values below 5e-11 are written as 0",
            "source": syn.manifest["sources"][0]["path"],
        }
    ]
    path = tmp_path_factory.mktemp("stored") / "ds"
    write_dataset(path, syn.manifest, syn.runs, syn.trace, syn.positions)
    return path


@pytest.fixture
def copy(stored, tmp_path):
    shutil.copytree(stored, tmp_path / "ds")
    return tmp_path / "ds"


def edit_manifest(path, change):
    file = path / "manifest.json"
    m = json.loads(file.read_text())
    change(m)
    file.write_text(json.dumps(m))


def test_scan_returns_lazy_frames(stored):
    ds = scan_dataset(stored, warn=False)
    assert isinstance(ds.runs, pl.LazyFrame) and isinstance(ds.trace, pl.LazyFrame)
    assert isinstance(ds.positions, pl.LazyFrame)
    assert ds.dataset_id == "synthetic-1" and ds.format_version == "1.0"


def test_manifest_warnings_are_issued(stored):
    with pytest.warns(DatasetWarning, match="limited_precision"):
        scan_dataset(stored)


@pytest.mark.parametrize("version", ["2.0", "0.9", "10.1"])
def test_unknown_major_version_is_refused(copy, version):
    edit_manifest(copy, lambda m: m.update(format_version=version))
    with pytest.raises(UnsupportedFormatVersion, match=version):
        scan_dataset(copy)


@pytest.mark.filterwarnings("ignore::iohinspector.format.DatasetWarning")
def test_newer_minor_version_is_read(copy):
    edit_manifest(copy, lambda m: m.update(format_version="1.9", added_later=True))
    with pytest.warns(FormatWarning):
        ds = scan_dataset(copy, warn=True)
    assert ds.trace.select(pl.len()).collect().item() > 0


def test_invalid_manifest_is_refused(copy):
    edit_manifest(copy, lambda m: m.pop("sources"))
    with pytest.raises(ManifestError, match="sources"):
        scan_dataset(copy)


def test_missing_table_is_refused(copy):
    (copy / "trace.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="trace.parquet"):
        scan_dataset(copy, warn=False)


def test_dataset_without_positions(tmp_path):
    syn = make_dataset(seed=31)
    write_dataset(tmp_path / "ds", syn.manifest, syn.runs, syn.trace)
    assert scan_dataset(tmp_path / "ds").positions is None


# -- predicate pushdown ---------------------------------------------------------


@pytest.fixture
def grouped(tmp_path, monkeypatch):
    """A dataset with one row group per (function, dimension), for 4 x 3 groups."""
    monkeypatch.setattr(writer_module, "MIN_ROW_GROUP_ROWS", 10)
    monkeypatch.setattr(writer_module, "MAX_ROW_GROUP_ROWS", 10_000)
    syn = make_dataset(
        functions=(1, 2, 3, 4),
        dimensions=(2, 5, 10),
        runs_per_file=3,
        rows=lambda r: r.randint(5, 30),
        positions=True,
        seed=32,
    )
    write_dataset(tmp_path / "ds", syn.manifest, syn.runs, syn.trace, syn.positions)
    return tmp_path / "ds"


def _row_group_ranges(file, column):
    meta = pq.ParquetFile(file).metadata
    j = [meta.schema.column(i).name for i in range(meta.num_columns)].index(column)
    out = []
    for i in range(meta.num_row_groups):
        stats = meta.row_group(i).column(j).statistics
        out.append((stats.min, stats.max))
    return out


def _corrupt_row_groups(file, keep):
    """Overwrite the data of every row group whose index is not in ``keep``.

    The footer stays intact, so a reader that skips those row groups by their
    statistics still works, and one that reads them fails.
    """
    meta = pq.ParquetFile(file).metadata
    spans = []
    for i in range(meta.num_row_groups):
        if i in keep:
            continue
        rg = meta.row_group(i)
        for j in range(rg.num_columns):
            col = rg.column(j)
            starts = [col.data_page_offset]
            if col.has_dictionary_page and col.dictionary_page_offset is not None:
                starts.append(col.dictionary_page_offset)
            start = min(starts)
            spans.append((start, col.total_compressed_size))
    with open(file, "r+b") as fh:
        for start, size in spans:
            fh.seek(start)
            fh.write(b"\xff" * size)
    return len(spans)


@pytest.mark.parametrize("table", ["trace", "positions"])
def test_filters_on_function_and_dimension_skip_row_groups(grouped, table):
    file = grouped / f"{table}.parquet"
    f_ranges = _row_group_ranges(file, "function_id")
    d_ranges = _row_group_ranges(file, "dimension")
    assert len(f_ranges) == 12  # one row group per (function, dimension)
    match = {
        i
        for i, ((f0, f1), (d0, d1)) in enumerate(zip(f_ranges, d_ranges))
        if f0 <= 3 <= f1 and d0 <= 5 <= d1
    }
    assert len(match) == 1

    predicate = (pl.col("function_id") == 3) & (pl.col("dimension") == 5)
    expected = pl.read_parquet(file).filter(predicate)
    assert expected.height > 0

    assert _corrupt_row_groups(file, keep=match) > 0
    ds = scan_dataset(grouped, warn=False)
    lazy = ds.trace if table == "trace" else ds.positions
    assert_frame_equal(lazy.filter(predicate).collect(), expected)
    with pytest.raises((pl.exceptions.PolarsError, OSError)):  # others are unreadable
        lazy.collect()


def test_filter_on_function_alone_skips_row_groups(grouped):
    file = grouped / "trace.parquet"
    f_ranges = _row_group_ranges(file, "function_id")
    match = {i for i, (f0, f1) in enumerate(f_ranges) if f0 <= 2 <= f1}
    assert len(match) == 3
    expected = pl.read_parquet(file).filter(pl.col("function_id") == 2)
    _corrupt_row_groups(file, keep=match)
    got = scan_dataset(grouped, warn=False).trace.filter(pl.col("function_id") == 2)
    assert_frame_equal(got.collect(), expected)


def test_trace_joins_with_runs(stored):
    ds = scan_dataset(stored, warn=False)
    runs = ds.runs.select("run_id", "algorithm_name", "instance")
    joined = ds.trace.join(runs, on="run_id").collect()
    assert joined.height == ds.trace.collect().height
    assert joined["algorithm_name"].null_count() == 0
