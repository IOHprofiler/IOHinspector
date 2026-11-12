"""Benchmark of the full-bbob ERT table (Principle 4 of the iohinspector plan).

    python scripts/bench_ert.py [--algorithms 258] [--rows 40] [--out DIR]

Writes a synthetic full-bbob set to Parquet: 24 functions, 6 dimensions, 15 runs
per (algorithm, function, dimension), ``--rows`` improvements per run (Δf from
1e3 down to 1e-10 at random, the ``evals_used`` past the last one). Then times
reading it and the ERT table at the 51 bbob targets by algorithm, function and
dimension (``aggregate_running_time``). The target is 5 s on the 64-core server
with 16 threads (``POLARS_MAX_THREADS=16``).
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

from iohinspector.metrics import aggregate_running_time
from iohinspector.targets import bbob_targets

BY = ["algorithm_name", "function_id", "dimension"]


def write(folder: Path, algorithms: int, rows: int, seed: int = 1) -> None:
    """``runs.parquet`` and ``trace.parquet`` of the synthetic set."""
    rng = np.random.default_rng(seed)
    a, f, d, r = np.meshgrid(
        np.arange(algorithms), np.arange(1, 25), [2, 3, 5, 10, 20, 40], np.arange(15)
    )
    n = a.size
    runs = pl.DataFrame(
        {
            "run_id": np.arange(n),
            "algorithm_name": [f"alg{i:03d}" for i in a.ravel()],
            "function_id": f.ravel(),
            "dimension": d.ravel(),
        }
    ).with_columns(pl.col("algorithm_name").cast(pl.Categorical))
    steps = rng.geometric(0.02, size=(n, rows)).cumsum(axis=1)
    lowest = rng.uniform(-10, 2, size=(n, 1))
    log_y = 3 + (lowest - 3) * np.sort(rng.random((n, rows)), axis=1)
    runs = runs.with_columns(evals_used=steps[:, -1] + rng.integers(0, 1000, n))
    trace = pl.DataFrame(
        {
            "run_id": np.repeat(np.arange(n), rows),
            "evaluations": steps.ravel().astype(np.uint64),
            "raw_y": 10.0 ** log_y.ravel(),
        }
    )
    runs.write_parquet(folder / "runs.parquet")
    trace.write_parquet(folder / "trace.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--algorithms", type=int, default=258)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--out", type=Path, help="keep the data here")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        folder = args.out or Path(tmp)
        folder.mkdir(parents=True, exist_ok=True)
        write(folder, args.algorithms, args.rows)
        start = time.perf_counter()
        runs = pl.scan_parquet(folder / "runs.parquet")
        data = pl.scan_parquet(folder / "trace.parquet").join(runs, on="run_id")
        data = data.collect()
        loaded = time.perf_counter()
        table = aggregate_running_time(data, by=BY, targets=bbob_targets())
        done = time.perf_counter()
    print(
        f"{data['run_id'].n_unique()} runs, {data.height} rows, "
        f"{pl.thread_pool_size()} threads: read {loaded - start:.2f} s, "
        f"ERT table ({table.height} rows) {done - loaded:.2f} s, "
        f"total {done - start:.2f} s"
    )


if __name__ == "__main__":
    main()
