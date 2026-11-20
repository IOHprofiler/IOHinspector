"""Benchmark of the Shapley computations (SH-2, Q-SH-2, review of I16).

    python scripts/bench_shapley.py [--groups 5 10 15] [--samples 5] [--rows 40]

Writes the synthetic full-bbob set of ``bench_ert.py`` (24 functions, 6 dimensions,
15 runs per algorithm and pair) for the largest number of groups. For each number
of groups n, builds the game of ``get_shapley_values`` on the first n algorithms
(``get_portfolio_ecdf``, default targets, log budget axis) and times the default
value with each ``method``: ``closed`` (the default), ``exact`` (enumeration) and
``sampled`` (``samples``, seed 1). Prints a Markdown table with the largest
difference of each to the closed form and the largest standard error of the
sampled values.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import polars as pl
from bench_ert import write

from iohinspector.metrics import envelope_value, get_portfolio_ecdf, shapley_values


def timed(f, *args, **kw):
    start = time.perf_counter()
    result = f(*args, **kw)
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
    print(
        "| groups | steps | ECDF data (s) | closed form (s) | enumeration (s) "
        "| sampled (s) | enumeration diff | sampled diff | max se |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in args.groups:
        names = [f"alg{i:03d}" for i in range(n)]
        part = data.filter(pl.col("algorithm_name").is_in(names))
        (_, ecdf, width), t_ecdf = timed(get_portfolio_ecdf, part, scale_eval_log=True)
        value = envelope_value(ecdf, width)
        (closed, _), t_closed = timed(shapley_values, n, value, method="closed")
        (exact, _), t_exact = timed(
            shapley_values, n, value, method="exact", exact_limit=n
        )
        (sampled, se), t_sampled = timed(
            shapley_values, n, value, method="sampled", samples=args.samples, seed=1
        )
        print(
            f"| {n} | {ecdf.shape[1]} | {t_ecdf:.2f} | {t_closed:.3f} "
            f"| {t_exact:.2f} | {t_sampled:.2f} | {abs(exact - closed).max():.1e} "
            f"| {abs(sampled - closed).max():.1e} | {se.max():.1e} |"
        )


if __name__ == "__main__":
    main()
