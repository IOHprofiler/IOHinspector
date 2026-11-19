"""Benchmark of exact against sampled Shapley values (SH-2, review of I16).

    python scripts/bench_shapley.py [--groups 5 10 15] [--samples 5] [--rows 40]

Writes the synthetic full-bbob set of ``bench_ert.py`` (24 functions, 6 dimensions,
15 runs per algorithm and pair) for the largest number of groups. For each number
of groups n, times ``get_shapley_values`` on the first n algorithms with its default
targets on a log budget axis, exact and sampled (``samples``, seed 1), and prints a
Markdown table with the largest difference between the two and the largest
standard error of the sampled values.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import polars as pl
from bench_ert import write

from iohinspector.metrics import get_shapley_values


def timed(data: pl.DataFrame, **kw) -> tuple[pl.DataFrame, float]:
    start = time.perf_counter()
    result = get_shapley_values(data, scale_eval_log=True, **kw)
    return result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--groups", type=int, nargs="+", default=[5, 10, 12, 14, 15])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--rows", type=int, default=40)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        write(Path(tmp), max(args.groups), args.rows)
        runs = pl.read_parquet(Path(tmp) / "runs.parquet")
        data = pl.read_parquet(Path(tmp) / "trace.parquet").join(runs, on="run_id")
    data = data.with_columns(pl.col("algorithm_name").cast(pl.String))
    print(f"{pl.thread_pool_size()} threads, samples={args.samples}\n")
    print("| groups | exact (s) | sampled (s) | exact / sampled | max diff | max se |")
    print("|---:|---:|---:|---:|---:|---:|")
    for n in args.groups:
        names = [f"alg{i:03d}" for i in range(n)]
        part = data.filter(pl.col("algorithm_name").is_in(names))
        exact, t_exact = timed(part, exact_limit=n)
        sampled, t_sampled = timed(part, exact_limit=0, samples=args.samples, seed=1)
        diff = (sampled["shapley"] - exact["shapley"]).abs().max()
        print(
            f"| {n} | {t_exact:.2f} | {t_sampled:.2f} | {t_exact / t_sampled:.1f} "
            f"| {diff:.1e} | {sampled['se'].max():.1e} |"
        )


if __name__ == "__main__":
    main()
