"""cocopp reference values for the COCO fixture (WP I03).

Usage: python generate.py <fixtures dir> <output dir> [<fixture name>]

Reads a COCO fixture (default ``coco``) with ``cocopp.pproc.DataSetList`` and
writes Parquet files with
  * runtimes per run for the 51 COCO targets 10^(2 - k/5), k = 0..50  (``detEvals``),
  * aRT per target (``detERT``), success counts, mean and median evaluations of successes,
  * aRT for the targets of interest of cocopp's ERT tables (10, 1, 0.1, 1e-3, 1e-5, 1e-7),
  * ECDF values: the fraction of (target, run) pairs whose runtime is <= a budget, the quantity
    cocopp plots in its runtime-distribution figures, without simulated restarts,
  * for ``coco_old``, the same per target with simulated restarts, as ``pprldmany``
    draws them (``toolsstats.drawSP``), at 10 log-spaced budgets from 10 to 1e6 (WP I33).
plus versions.json. Output is deterministic. A fixture with one data set
(``coco``) gets the file names above; a fixture with several (``coco_old``: f6
in 2-D and 3-D) gets one set of files per data set, with the suffix ``_<dim>D``.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import cocopp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from cocopp import pproc, toolsstats

TARGETS_51 = 10 ** (2 - np.arange(51) / 5)
TARGETS_TABLE = np.array([10.0, 1.0, 0.1, 1e-3, 1e-5, 1e-7])
#: budgets and sample size of the simulated restarts; cocopp's own sample size is 100
#: (in a hurry, its default) or 1100 for five runs, the larger one makes the
#: statistical comparison of tests/test_metrics/test_ecdf.py tighter
RESTART_BUDGETS = np.round(10 ** np.linspace(1, 6, 10))
RESTART_SAMPLES = 20000
SEED = 20240423


def write(path: str, columns: dict) -> None:
    table = pa.table({k: pa.array(v) for k, v in columns.items()})
    pq.write_table(table, path, compression="zstd", compression_level=9, use_dictionary=False,
                   version="2.6")


def main(fix_dir: str, out_dir: str, fixture: str = "coco") -> None:
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(fix_dir, fixture, "**", "*.info")
    info = sorted(glob.glob(pattern, recursive=True))
    dsl = sorted(pproc.DataSetList(info), key=lambda d: int(d.dim))
    if fixture == "coco":
        assert len(dsl) == 1, len(dsl)
        write_references(dsl[0], out_dir, "")
        write_versions(dsl[0], out_dir)
        return
    for ds in dsl:
        write_references(ds, out_dir, f"_{int(ds.dim)}D")
        write_restarts(ds, out_dir, f"_{int(ds.dim)}D")
    write_versions_many(dsl, out_dir)


def write_references(ds, out_dir: str, suffix: str) -> None:
    dim = int(ds.dim)

    # runtimes per run and target
    evals = ds.detEvals(TARGETS_51)  # list over targets of arrays over runs (nan = not reached)
    t_col, r_col, i_col, e_col = [], [], [], []
    for t, arr in zip(TARGETS_51, evals):
        for run, (inst, v) in enumerate(zip(ds.instancenumbers, np.asarray(arr, dtype=float))):
            t_col.append(float(t))
            r_col.append(run)
            i_col.append(int(inst))
            e_col.append(float(v))
    write(os.path.join(out_dir, f"runtimes_51targets{suffix}.parquet"),
          {"target": t_col, "run": r_col, "instance": i_col, "evals": e_col})

    def art_table(targets, name):
        ert = ds.detERT(targets)
        succ = ds.detSuccesses(targets)
        rows = {"target": [float(t) for t in targets], "dim": [dim] * len(targets),
                "ert": [float(v) for v in ert], "n_success": [int(v) for v in succ],
                "n_runs": [int(ds.nbRuns())] * len(targets)}
        ev = [np.asarray(x, dtype=float) for x in ds.detEvals(targets)]
        hits = [a[np.isfinite(a)] for a in ev]
        rows["mean_evals_success"] = [float(h.mean()) if h.size else float("nan") for h in hits]
        rows["median_evals_success"] = [
            float(np.median(h)) if h.size else float("nan") for h in hits
        ]
        write(os.path.join(out_dir, name), rows)

    art_table(TARGETS_51, f"art_51targets{suffix}.parquet")
    art_table(TARGETS_TABLE, f"art_table_targets{suffix}.parquet")

    # ECDF over (target, run) pairs for budgets dim * 10^(k/5)
    budgets = dim * 10 ** (np.arange(0, 26) / 5.0)
    all_ev = np.array([np.asarray(a, dtype=float) for a in evals])  # targets x runs
    ecdf = [float(np.mean(np.isfinite(all_ev) & (all_ev <= b))) for b in budgets]
    write(os.path.join(out_dir, f"ecdf_51targets{suffix}.parquet"),
          {"budget": [float(b) for b in budgets],
           "budget_over_dim": [float(b / dim) for b in budgets],
           "ecdf": ecdf})


def write_restarts(ds, out_dir: str, suffix: str) -> None:
    """Simulated restarts per target, as pprldmany.py:800-817 (divisor 1, uniform
    instances); for ``coco_old``, whose runs are partly unsuccessful."""
    np.random.seed(SEED)
    rows = {"target": [], "budget": [], "ecdf": []}
    for t, arr in zip(TARGETS_51, ds.detEvals(TARGETS_51)):
        arr = np.asarray(arr, dtype=float)
        succ, unsucc = arr[~np.isnan(arr)], ds.maxevals[np.isnan(arr)]
        x = np.full(RESTART_SAMPLES, np.inf)
        if succ.size:
            x = np.asarray(toolsstats.drawSP(succ, unsucc, percentiles=[50],
                                             samplesize=RESTART_SAMPLES)[1])
        for b in RESTART_BUDGETS:
            rows["target"].append(float(t))
            rows["budget"].append(float(b))
            rows["ecdf"].append(float(np.mean(x <= b)))
    write(os.path.join(out_dir, f"ecdf_restarts{suffix}.parquet"), rows)


def write_versions(ds, out_dir: str) -> None:
    dim = int(ds.dim)
    meta = {
        "cocopp": cocopp.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pa.__version__,
        "python": sys.version.split()[0],
        "dataset": str(ds),
        "algId": ds.algId, "funcId": int(ds.funcId), "dim": dim,
        "instances": [int(i) for i in ds.instancenumbers],
        "targets": (
            "51 targets 10^(2 - k/5), k = 0..50, as Delta f = f - f_opt "
            "(cocopp compares with its own 1e-9 relative tolerance)"
        ),
        "files": {
            "runtimes_51targets.parquet": (
                "ds.detEvals(targets): runtime per run, NaN when not reached"),
            "art_51targets.parquet": (
                "ds.detERT, ds.detSuccesses; mean/median evaluations of successful runs"),
            "art_table_targets.parquet": "same for the table targets 10, 1, 0.1, 1e-3, 1e-5, 1e-7",
            "ecdf_51targets.parquet": (
                "fraction of (target, run) pairs with runtime <= budget, "
                "budgets dim*10^(k/5), k = 0..25"),
        },
    }
    with open(os.path.join(out_dir, "versions.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")


def write_versions_many(dsl, out_dir: str) -> None:
    meta = {
        "cocopp": cocopp.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pa.__version__,
        "python": sys.version.split()[0],
        "datasets": [
            {
                "dataset": str(ds),
                "algId": ds.algId,
                "funcId": int(ds.funcId),
                "dim": int(ds.dim),
                "instances": [int(i) for i in ds.instancenumbers],
                "suffix": f"_{int(ds.dim)}D",
            }
            for ds in dsl
        ],
        "targets": (
            "51 targets 10^(2 - k/5), k = 0..50, as Delta f = f - f_opt "
            "(cocopp compares with its own 1e-9 relative tolerance)"
        ),
        "files": "as for the coco fixture, one set per data set with its suffix",
        "ecdf_restarts": (
            "per target and budget, the fraction of simulated restart runtimes <= budget: "
            f"toolsstats.drawSP as pprldmany calls it, {RESTART_SAMPLES} samples, "
            f"np.random.seed({SEED}) per data set, budgets round(10^(1 + 5k/9)), k = 0..9"),
    }
    with open(os.path.join(out_dir, "versions.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
