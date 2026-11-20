"""Benchmark harness on full-bbob-sized data (WP I22, Principle 4 of the plan).

    python scripts/bench.py [--algorithms 258] [--rows 40] [--out DIR]
                            [--datasets DIR] [--pair 10 20] [--repeat 3]
                            [--only CASE ...] [--json FILE]

The data is either synthetic (default) or every converted dataset folder in
``--datasets`` (``iohinspector convert``), such as the converted bbob collection.

The synthetic set is the size of full bbob: ``--algorithms`` algorithms, 24
functions, 6 dimensions and 15 runs per (algorithm, function, dimension), with
``--rows`` improvements per run (Δf from 1e3 down to 1e-10 at random) and an
``evals_used`` past the last one. It is written in format v1, one dataset per
algorithm as on the website, to ``--out`` (kept and reused when given) or to a
temporary folder.

Each case reads the datasets with ``DataManager`` as the website does, or runs a
metric on data loaded before. ``all`` cases use every pair, ``pair`` cases the one
(function, dimension) pair ``--pair``. Prints a Markdown table of the best of
``--repeat`` runs in seconds. The targets of the plan, on the 64-core server with
``POLARS_MAX_THREADS=16``: the full ERT table under 5 s, a single-function view
under 0.3 s.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from iohinspector import DataManager
from iohinspector.format import descriptor_frame, new_manifest, run_ids, write_dataset
from iohinspector.metrics import (
    aggregate_convergence,
    aggregate_running_time,
    get_data_ecdf,
    get_ecdf,
    get_ecdf_auc,
    get_fixed_budget_ecdf,
)
from iohinspector.targets import bbob_targets

BY = ["algorithm_name", "function_id", "dimension"]
COLUMNS = [*BY, "suite", "evals_used"]
DIMENSIONS = [2, 3, 5, 10, 20, 40]
SOURCE = {"path": "synthetic.dat", "sha256": "0" * 64, "bytes": 0}


def synthetic(algorithms: int, rows: int, seed: int = 1) -> pl.DataFrame:
    """The synthetic full-bbob set as one frame: ``run_id``, ``algorithm_name``,
    ``function_id``, ``dimension``, ``evals_used``, ``evaluations`` and ``raw_y``."""
    rng = np.random.default_rng(seed)
    a, f, d, _ = np.meshgrid(
        np.arange(algorithms), np.arange(1, 25), DIMENSIONS, np.arange(15)
    )
    n = a.size
    steps = rng.geometric(0.02, size=(n, rows)).cumsum(axis=1)
    lowest = rng.uniform(-10, 2, size=(n, 1))
    log_y = 3 + (lowest - 3) * np.sort(rng.random((n, rows)), axis=1)
    runs = pl.DataFrame(
        {
            "run_id": np.arange(n),
            "algorithm_name": [f"alg{i:03d}" for i in a.ravel()],
            "function_id": f.ravel(),
            "dimension": d.ravel(),
            "evals_used": steps[:, -1] + rng.integers(0, 1000, n),
        }
    )
    trace = pl.DataFrame(
        {
            "run_id": np.repeat(np.arange(n), rows),
            "evaluations": steps.ravel(),
            "raw_y": 10.0 ** log_y.ravel(),
        }
    )
    return trace.join(runs, on="run_id")


def write_datasets(folder: Path, data: pl.DataFrame) -> None:
    """``data`` (``synthetic``) in format v1, one dataset per algorithm."""
    for (name,), part in data.partition_by("algorithm_name", as_dict=True).items():
        runs = (
            part.group_by("run_id", maintain_order=True)
            .agg(pl.exclude("evaluations", "raw_y").first())
            .with_columns(
                suite=pl.lit("bbob"),
                maximization=False,
                source_file=pl.lit(SOURCE["path"]),
                source_run_index=pl.int_range(pl.len()),
            )
        )
        ids = run_ids(str(name), runs["source_file"], runs["source_run_index"])
        runs = runs.with_columns(new_id=ids)
        trace = part.join(runs.select("run_id", "new_id"), on="run_id").select(
            run_id="new_id",
            row=pl.int_range(pl.len()).over("run_id"),
            evaluations="evaluations",
            raw_y="raw_y",
        )
        runs = runs.drop("run_id", "algorithm_name").rename({"new_id": "run_id"})
        descriptors = descriptor_frame(
            (i, "algorithm_name", name, "algorithm") for i in ids
        )
        write_dataset(
            folder / str(name),
            new_manifest(str(name), "synthetic", "0", sources=[SOURCE]),
            runs,
            trace,
            descriptors=descriptors,
        )


def manager(folder: Path) -> DataManager:
    """Every converted dataset in ``folder``."""
    result = DataManager()
    for path in sorted(p.parent for p in folder.glob("*/manifest.json")):
        result.add_dataset(path)
    return result


def quiet(call: Callable, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(*args, **kwargs)


def cases(folder: Path, pair: tuple[int, int]) -> dict[str, Callable[[], object]]:
    """The benchmark cases by name; the data of the metric cases is loaded once.
    The sampled ECDFs use 50 log-spaced budgets up to the largest evaluation in
    the data they get, as a figure does."""
    f, d = pair
    every = quiet(manager(folder).load, include_columns=COLUMNS)
    one = every.filter(function_id=f, dimension=d)
    pairs = every.select("function_id", "dimension").unique()
    targets = pairs.join(pl.DataFrame({"target": bbob_targets()}), how="cross")
    budgets, budgets_one = (
        np.unique(np.geomspace(1, x["evaluations"].max(), 50).astype(int))
        for x in (every, one)
    )

    def select_pair():
        return manager(folder).select(function_ids=[f], dimensions=[d])

    return {
        "all: load": lambda: manager(folder).load(include_columns=COLUMNS),
        "all: ERT table": lambda: aggregate_running_time(
            every, by=BY, targets=bbob_targets()
        ),
        "all: fixed-budget table": lambda: aggregate_convergence(
            every, by=BY, budgets=budgets
        ),
        "all: ECDF per pair, sampled": lambda: get_ecdf(
            every, targets, by=BY, x_values=budgets
        ),
        "all: ECDF per pair, sampled, carry forward": lambda: get_ecdf(
            every, targets, by=BY, x_values=budgets, stopped="carry_forward"
        ),
        "pair: select and load": lambda: select_pair().load(include_columns=COLUMNS),
        "pair: ERT table": lambda: aggregate_running_time(
            one, by=BY, targets=bbob_targets()
        ),
        "pair: fixed-budget table": lambda: aggregate_convergence(
            one, by=BY, budgets=budgets_one
        ),
        "pair: fixed-budget table, restarts": lambda: aggregate_convergence(
            one, by=BY, budgets=budgets_one, stopped="restart"
        ),
        "pair: ECDF": lambda: get_ecdf(one, targets, by=BY),
        "pair: ECDF, sampled": lambda: get_ecdf(
            one, targets, by=BY, x_values=budgets_one
        ),
        "pair: ECDF AUC": lambda: get_ecdf_auc(one, targets, by=BY),
        "pair: fixed-budget ECDF": lambda: get_fixed_budget_ecdf(one, by=BY),
        "pair: data ECDF (0.0.x)": lambda: get_data_ecdf(one, by=BY),
    }


def best_of(repeat: int, call: Callable[[], object]) -> float:
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        quiet(call)
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--algorithms", type=int, default=258)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--out", type=Path, help="keep the synthetic datasets here")
    parser.add_argument("--datasets", type=Path, help="converted datasets instead")
    parser.add_argument("--pair", type=int, nargs=2, default=(10, 20))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--only", nargs="+", help="cases whose name contains these")
    parser.add_argument("--json", type=Path, help="also write the times here")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        folder = args.datasets or args.out or Path(tmp)
        if not args.datasets and not any(folder.glob("*/manifest.json")):
            folder.mkdir(parents=True, exist_ok=True)
            write_datasets(folder, synthetic(args.algorithms, args.rows))
        todo = cases(folder, tuple(args.pair))
        if args.only:
            todo = {k: v for k, v in todo.items() if any(o in k for o in args.only)}
        print(f"{folder}: {pl.thread_pool_size()} threads, polars {pl.__version__}\n")
        print("| case | seconds |\n|---|---:|")
        times = {}
        for name, call in todo.items():
            times[name] = best_of(args.repeat, call)
            print(f"| {name} | {times[name]:.3f} |", flush=True)
    if args.json:
        args.json.write_text(json.dumps(times, indent=1))


if __name__ == "__main__":
    main()
