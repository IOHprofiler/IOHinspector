"""cocopp reference values for the COCO fixture (WP I03).

Usage: python generate.py <fixtures dir> <output dir>

Reads the COCO fixture with ``cocopp.pproc.DataSetList`` and writes Parquet files with
  * runtimes per run for the 51 COCO targets 10^(2 - k/5), k = 0..50  (``detEvals``),
  * aRT per target (``detERT``), success counts, mean and median evaluations of successes,
  * aRT for the targets of interest of cocopp's ERT tables (10, 1, 0.1, 1e-3, 1e-5, 1e-7),
  * ECDF values: the fraction of (target, run) pairs whose runtime is <= a budget, the quantity
    cocopp plots in its runtime-distribution figures, without simulated restarts.
plus versions.json. Output is deterministic.
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
import six  # noqa: F401  (cocopp needs it)
from cocopp import pproc

TARGETS_51 = 10 ** (2 - np.arange(51) / 5)
TARGETS_TABLE = np.array([10.0, 1.0, 0.1, 1e-3, 1e-5, 1e-7])


def write(path: str, columns: dict) -> None:
    table = pa.table({k: pa.array(v) for k, v in columns.items()})
    pq.write_table(table, path, compression="zstd", compression_level=9, use_dictionary=False,
                   version="2.6")


def main(fix_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    info = sorted(glob.glob(os.path.join(fix_dir, "coco", "**", "*.info"), recursive=True))
    dsl = pproc.DataSetList(info)
    assert len(dsl) == 1, len(dsl)
    ds = dsl[0]
    dim = int(ds.dim)

    # runtimes per run and target
    evals = ds.detEvals(TARGETS_51)  # list over targets of arrays over runs (nan = not reached)
    t_col, r_col, i_col, e_col = [], [], [], []
    for t, arr in zip(TARGETS_51, evals):
        for run, (inst, v) in enumerate(zip(ds.instancenumbers, np.asarray(arr, dtype=float))):
            t_col.append(float(t)); r_col.append(run); i_col.append(int(inst)); e_col.append(float(v))
    write(os.path.join(out_dir, "runtimes_51targets.parquet"),
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
        rows["median_evals_success"] = [float(np.median(h)) if h.size else float("nan") for h in hits]
        write(os.path.join(out_dir, name), rows)

    art_table(TARGETS_51, "art_51targets.parquet")
    art_table(TARGETS_TABLE, "art_table_targets.parquet")

    # ECDF over (target, run) pairs for budgets dim * 10^(k/5)
    budgets = dim * 10 ** (np.arange(0, 26) / 5.0)
    all_ev = np.array([np.asarray(a, dtype=float) for a in evals])  # targets x runs
    ecdf = [float(np.mean(np.isfinite(all_ev) & (all_ev <= b))) for b in budgets]
    write(os.path.join(out_dir, "ecdf_51targets.parquet"),
          {"budget": [float(b) for b in budgets], "budget_over_dim": [float(b / dim) for b in budgets],
           "ecdf": ecdf})

    meta = {
        "cocopp": cocopp.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pa.__version__,
        "python": sys.version.split()[0],
        "dataset": str(ds),
        "algId": ds.algId, "funcId": int(ds.funcId), "dim": dim,
        "instances": [int(i) for i in ds.instancenumbers],
        "targets": "51 targets 10^(2 - k/5), k = 0..50, as Delta f = f - f_opt (cocopp compares with its own 1e-9 relative tolerance)",
        "files": {
            "runtimes_51targets.parquet": "ds.detEvals(targets): runtime per run, NaN when not reached",
            "art_51targets.parquet": "ds.detERT, ds.detSuccesses; mean/median evaluations of successful runs",
            "art_table_targets.parquet": "same for the table targets 10, 1, 0.1, 1e-3, 1e-5, 1e-7",
            "ecdf_51targets.parquet": "fraction of (target, run) pairs with runtime <= budget, budgets dim*10^(k/5), k = 0..25",
        },
    }
    with open(os.path.join(out_dir, "versions.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
