"""Benchmark of simulated restarts far beyond the runs' budgets (WP I33, review of #48).

    python scripts/bench_restarts.py [--budget 1e6] [--repeat 3]

On the COCO fixtures of ``tests/golden/fixtures``: ``coco`` (BFGS on f1 in 2-D, 15
runs that all reach 1e-8 within 13 to 25 evaluations) and ``coco_old`` (DIRECT on
f6 in 2-D and 3-D, five runs each with 3e4 to 1e5 evaluations, mostly
unsuccessful), with ten budgets log-spaced from 10 to ``--budget``. Times the
fixed-budget ECDF and the fixed-budget summary with restarts, and the target ECDF
with restarts at the 51 bbob targets sampled at the budgets. Prints a Markdown table
of the best of ``--repeat`` runs in seconds.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from iohinspector import DataManager
from iohinspector.metrics import aggregate_convergence, get_ecdf, get_fixed_budget_ecdf
from iohinspector.targets import PAIR, bbob_targets

FIXTURES = Path(__file__).parent.parent / "tests" / "golden" / "fixtures"
COLUMNS = ["function_id", "dimension", "algorithm_name", "evals_used"]


def load(name: str) -> pl.DataFrame:
    manager = DataManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manager.add_folder(str(FIXTURES / name))
    return manager.load(include_columns=COLUMNS)


def best_of(repeat: int, call) -> float:
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--budget", type=float, default=1e6)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    budget = np.unique(np.round(np.geomspace(10, args.budget, 10)).astype(np.int64))
    print("| fixture | function | seconds |\n|---|---|---|")
    for name in ("coco", "coco_old"):
        data = load(name)
        pairs = data.select(PAIR).unique()
        budgets = pairs.join(pl.DataFrame({"budget": budget}), how="cross")
        targets = pairs.join(pl.DataFrame({"target": bbob_targets()}), how="cross")
        calls = {
            "get_fixed_budget_ecdf": (get_fixed_budget_ecdf, budgets, {}),
            "aggregate_convergence": (
                aggregate_convergence,
                None,
                {
                    "by": ["algorithm_name", *PAIR],
                    "budgets": budget,
                    "stopped": "restart",
                },
            ),
            "get_ecdf": (get_ecdf, targets, {"x_values": budget}),
        }
        for function, (call, grid, options) in calls.items():
            args_ = (data,) if grid is None else (data, grid)
            seconds = best_of(args.repeat, lambda: call(*args_, **options))  # noqa: B023
            print(f"| {name} | {function} | {seconds:.3f} |")


if __name__ == "__main__":
    main()
