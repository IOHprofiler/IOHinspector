import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from iohinspector.format import scan_dataset

ROWS = 10_000_000
RUNS = 20_000
LIMIT_BYTES = 2 * 2**30


@pytest.mark.slow
def test_ten_million_rows_stay_under_two_gigabytes(tmp_path):
    child = Path(__file__).with_name("memory_child.py")
    out = tmp_path / "big"
    result = subprocess.run(
        [sys.executable, str(child), str(out), str(ROWS), str(RUNS)],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    print(
        f"\nwrite of {ROWS} rows: {report['seconds']:.1f} s, peak RSS {report['peak_rss_bytes'] / 2**20:.0f} MiB"
    )
    assert report["counts"] == {
        "runs": RUNS,
        "descriptors_rows": 2 * RUNS,  # algorithm_name and algorithm_info
        "trace_rows": ROWS,
    }
    assert report["peak_rss_bytes"] < LIMIT_BYTES

    assert pq.ParquetFile(out / "trace.parquet").metadata.num_rows == ROWS
    ds = scan_dataset(out)
    per_run = ds.trace.group_by("run_id").agg(pl.len(), pl.col("row").max()).collect()
    assert per_run.height == RUNS
    assert (per_run["len"] == ROWS // RUNS).all() and (
        per_run["row"] == ROWS // RUNS - 1
    ).all()
