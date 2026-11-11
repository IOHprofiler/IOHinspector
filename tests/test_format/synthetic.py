"""Synthetic datasets for the format tests."""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl

from iohinspector.format import descriptor_frame, new_manifest, run_ids

DATASET_ID = "synthetic-1"


@dataclass
class Synthetic:
    manifest: dict
    runs: pl.DataFrame
    trace: pl.DataFrame
    positions: pl.DataFrame | None
    descriptors: pl.DataFrame

    def with_algorithms(self, records) -> pl.DataFrame:
        """The runs' ``algorithm_name`` descriptors plus ``descriptor_frame(records)``."""
        names = self.descriptors.filter(pl.col("key") == "algorithm_name")
        return pl.concat([names, descriptor_frame(records)])


def make_dataset(
    functions=(1, 2),
    dimensions=(2, 5),
    algorithms=("algA", "algB"),
    runs_per_file=2,
    rows=lambda rng: rng.randint(0, 6),
    extras=True,
    positions=False,
    seed=0,
    dataset_id=DATASET_ID,
) -> Synthetic:
    """One source file per (algorithm, function, dimension), with ``runs_per_file`` runs.

    ``descriptors`` holds every run's ``algorithm_name`` (origin ``algorithm``) and
    the descriptors ``seed``, ``popsize`` and ``folder``; ``runs`` has no algorithm.

    Values include ties, zeros, inf and nan, so that a lossy writer would be caught.
    """
    # #lizard forgives  (test helper: every option has a default)
    rng = random.Random(seed)  # nosec B311  # reproducible test data, not security
    runs, sources = _runs_and_sources(
        algorithms, functions, dimensions, runs_per_file, dataset_id
    )
    descriptors = _descriptors(runs)
    runs = runs.drop("algorithm")
    trace_rows, pos_rows = [], []
    for run in runs.iter_rows(named=True):
        _add_run_rows(run, rows(rng), rng, extras, positions, trace_rows, pos_rows)
    trace_schema = {
        "run_id": pl.Int64,
        "row": pl.Int64,
        "evaluations": pl.Int64,
        "raw_y": pl.Float64,
    }
    if extras:
        trace_schema |= {"sigma": pl.Float64, "restart": pl.Int64, "tag": pl.String}
    trace = pl.DataFrame(trace_rows, schema=trace_schema)
    pos_df = None
    if positions:
        pos_schema = {"run_id": pl.Int64, "row": pl.Int64} | {
            f"x{j}": pl.Float64 for j in range(max(dimensions))
        }
        pos_df = pl.DataFrame(pos_rows, schema=pos_schema)
    manifest = new_manifest(dataset_id, "synthetic", "0.0", sources=sources)
    return Synthetic(manifest, runs, trace, pos_df, descriptors)


def _runs_and_sources(algorithms, functions, dimensions, runs_per_file, dataset_id):
    """The runs (with an ``algorithm`` column) and the manifest's sources."""
    run_rows, sources = [], []
    for alg in algorithms:
        for f in functions:
            for d in dimensions:
                path = f"{alg}/data_f{f}/IOHprofiler_f{f}_DIM{d}.dat"
                sources.append(
                    {
                        "path": path,
                        "sha256": "0" * 64,
                        "bytes": 0,
                        "kind": "ioh-dat",
                        "data_rows": 0,
                    }
                )
                run_rows += [
                    {
                        "algorithm": alg,
                        "function_id": f,
                        "dimension": d,
                        "instance": i + 1,
                        "maximization": f % 2 == 0,
                        "source_file": path,
                        "run_index": i,
                    }
                    for i in range(runs_per_file)
                ]
    runs = pl.DataFrame(run_rows)
    runs = runs.with_columns(
        run_ids(dataset_id, runs["source_file"], runs["run_index"])
    )
    return runs, sources


def _descriptors(runs: pl.DataFrame) -> pl.DataFrame:
    """``algorithm_name``, ``seed``, ``popsize`` and ``folder`` of every run."""
    records = []
    for run in runs.iter_rows(named=True):
        records.append((run["run_id"], "algorithm_name", run["algorithm"], "algorithm"))
        records.append((run["run_id"], "seed", run["run_index"] * 7 + 1, "run"))
        records.append(
            (run["run_id"], "popsize", 10 * (1 + run["function_id"] % 2), "algorithm")
        )
        records.append((run["run_id"], "folder", run["algorithm"], "path"))
    return descriptor_frame(records)


_SPECIAL = [0.0, float("inf"), float("-inf"), float("nan"), 1e-300, -0.0]


def _add_run_rows(run, n, rng, extras, positions, trace_rows, pos_rows):
    """Append ``n`` trace rows of ``run``, and positions for some of them."""
    evals = 0
    y = rng.uniform(1, 100)
    for r in range(n):
        evals += rng.randint(1, 3)
        if rng.random() < 0.3:
            y = rng.choice(_SPECIAL)
        elif rng.random() < 0.5:
            y = y * rng.uniform(0.5, 1.0)  # else: repeat the previous value
        row = {"run_id": run["run_id"], "row": r, "evaluations": evals, "raw_y": y}
        if extras:
            row["sigma"] = rng.uniform(0, 1) if rng.random() < 0.9 else None
            row["restart"] = rng.randint(0, 3)
            row["tag"] = rng.choice(["a", "b", None])
        trace_rows.append(row)
        if positions and rng.random() < 0.7:
            pos = {"run_id": run["run_id"], "row": r}
            for j in range(run["dimension"]):
                pos[f"x{j}"] = rng.uniform(-5, 5)
            pos_rows.append(pos)


def shuffled_chunks(
    df: pl.DataFrame, n_chunks: int, seed: int = 0
) -> list[pl.DataFrame]:
    """The rows of ``df`` in random order, split into ``n_chunks`` chunks."""
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
    size = max(1, -(-shuffled.height // n_chunks))
    return [shuffled.slice(i, size) for i in range(0, shuffled.height, size)]


def algorithm_descriptors(run_id: pl.Series, names: list[str]) -> pl.DataFrame:
    """The ``algorithm_name`` descriptors of the runs ``run_id``."""
    return descriptor_frame(
        (rid, "algorithm_name", name, "algorithm")
        for rid, name in zip(run_id.to_list(), names, strict=True)
    )


def big_trace_chunks(
    n_rows: int, n_runs: int, chunk_rows: int, dataset_id: str, seed: int = 0
) -> tuple[dict, pl.DataFrame, pl.DataFrame, Iterator[pl.DataFrame]]:
    """A large dataset generated chunk by chunk, with runs in a scrambled order.

    Runs are spread over 24 functions and 6 dimensions. Each chunk holds consecutive
    rows of a few runs, and the runs are visited in a random order, so the writer
    has to sort across spill files.
    """
    rng = np.random.default_rng(seed)
    funcs = np.arange(1, 25, dtype=np.int32)
    dims = np.array([2, 3, 5, 10, 20, 40], dtype=np.int32)
    idx = np.arange(n_runs)
    path = [f"alg{i % 7}/f{funcs[i % 24]}_d{dims[(i // 24) % 6]}.dat" for i in idx]
    runs = pl.DataFrame(
        {
            "function_id": funcs[idx % 24],
            "dimension": dims[(idx // 24) % 6],
            "maximization": np.zeros(n_runs, dtype=bool),
            "source_file": path,
            "run_index": idx // (24 * 6 * 7),
        }
    )
    runs = runs.with_columns(
        run_ids(dataset_id, runs["source_file"], runs["run_index"])
    )
    descriptors = algorithm_descriptors(runs["run_id"], [f"alg{i % 7}" for i in idx])
    sources = [{"path": p, "sha256": None, "bytes": None} for p in sorted(set(path))]
    manifest = new_manifest(dataset_id, "synthetic", "0.0", sources=sources)
    per_run = n_rows // n_runs
    ids = runs["run_id"].to_numpy()

    def chunks() -> Iterator[pl.DataFrame]:
        order = rng.permutation(n_runs)
        runs_per_chunk = max(1, chunk_rows // per_run)
        for start in range(0, n_runs, runs_per_chunk):
            end = start + runs_per_chunk
            sel = order[start:end]
            k = len(sel)
            yield pl.DataFrame(
                {
                    "run_id": np.repeat(ids[sel], per_run),
                    "row": np.tile(np.arange(per_run, dtype=np.int64), k),
                    "evaluations": np.tile(
                        np.arange(1, per_run + 1, dtype=np.int64), k
                    ),
                    "raw_y": rng.random(k * per_run),
                }
            )

    return manifest, runs, descriptors, chunks()
