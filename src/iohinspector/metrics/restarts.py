"""Simulated restarts in the ECDFs (``docs/definitions/ecdf.md``, EC-7): their
expected value, computed from the runs' runtimes instead of simulated.

A run set is the runs of one group of ``by`` on one (function, dimension) pair, n
runs. For a target t, the runs F that do not reach it are continued by runs drawn
uniformly from the set, so a restarted run reaches t within b with probability

    E_t(b) = 1/n sum_{i not in F} 1[T_i(t) <= b] + 1/n sum_{j in F} E_t(b - B_j).

Over the integer evaluations the increments r of E_t solve
r = h + 1/n sum_{j in F} r(. - B_j), h being the increments without restarts.

Targets are compared as levels, larger is better: -f when minimising, f when
maximising, or the normalised quality of the continuum ECDF. A run fails a target
exactly when its final level is below the target's, so F is the k runs of the set
with the lowest final levels, k being the number of final levels below the target.
"""

import math

import numpy as np
import polars as pl

from .utils import runs_of

#: restarts stop where the chance that a chain needs a further one is below this
EPSILON = 1e-17


def restarts(stopped: str) -> bool:
    """Whether ``stopped`` asks for simulated restarts (EC-7, FB-6)."""
    if stopped not in ("restart", "carry_forward"):
        raise ValueError(f"stopped is 'restart' or 'carry_forward', not {stopped!r}")
    return stopped == "restart"


def run_sets(data: pl.DataFrame, by: list[str], eval_var: str, final: pl.Expr):
    """One row per run: ``run_id``, ``run_set`` (its group of ``by``),
    ``evals_used`` (B_i, at least 1) and ``final``, the run's final level (``final``
    aggregated over its rows)."""
    runs = runs_of(data, by, eval_var).select(
        "run_id", pl.col("run_set").cast(pl.Int64), pl.col("evals_used").clip(1)
    )
    finals = data.group_by("run_id").agg(final.alias("final"))
    return runs.join(finals, on="run_id").sort("run_set", "final")


def _below(runs: pl.DataFrame, rows: pl.DataFrame, level: str, strict: bool):
    """Per row of ``rows`` (with ``run_set``): the runs of its set whose final level
    is below its ``level`` (or at most it, unless ``strict``)."""
    finals, levels = runs["final"].to_numpy(), rows[level].to_numpy()
    values, rank = np.unique(np.r_[finals, levels], return_inverse=True)
    width = values.size + 1  # one key per (run set, level)
    keys = runs["run_set"].to_numpy() * width + rank[: finals.size]  # sorted
    sets = rows["run_set"].to_numpy() * width
    side = "left" if strict else "right"
    below = np.searchsorted(keys, sets + rank[finals.size :], side)
    return below - np.searchsorted(keys, sets, "left")


def failed(runs: pl.DataFrame, hits: pl.DataFrame, level: str) -> pl.DataFrame:
    """``hits`` (with ``run_id``) with ``run_set`` and ``k``: the runs of the set
    that do not reach the hit's ``level``."""
    hits = hits.join(runs.select("run_id", "run_set"), on="run_id")
    return hits.with_columns(k=pl.Series(_below(runs, hits, level, True)))


def split(runs: pl.DataFrame, rows: pl.DataFrame) -> pl.DataFrame:
    """``rows``, each the levels (``lo``, ``hi``] that a run reaches at once, cut
    at the final levels of its run set, with ``run_set`` and ``k`` per piece: the
    runs that do not reach the levels of the piece."""
    rows = rows.join(runs.select("run_id", "run_set"), on="run_id")
    first = _below(runs, rows, "lo", False)
    last = _below(runs, rows, "hi", True)
    cuts = runs.select("run_set", "final", k=pl.int_range(pl.len()).over("run_set"))
    rows = rows.with_columns(first=pl.Series(first), last=pl.Series(last))
    k = pl.col("k")
    return (
        rows.with_columns(k=pl.int_ranges("first", pl.col("last") + 1))
        .explode("k")
        .join(cuts.rename({"final": "upper"}), on=["run_set", "k"], how="left")
        .join(
            cuts.select("run_set", lower="final", k=k + 1),
            on=["run_set", "k"],
            how="left",
        )
        .with_columns(
            lo=pl.when(k == pl.col("first")).then("lo").otherwise("lower"),
            hi=pl.when(k == pl.col("last")).then("hi").otherwise("upper"),
        )
        .filter(pl.col("hi") > pl.col("lo"))
        .drop("first", "last", "lower", "upper")
    )


def _renew(h: np.ndarray, budgets: np.ndarray, n: int) -> np.ndarray:
    """r = h + 1/n sum_j r(. - B_j) over the evaluations 0, 1, ..., in blocks of
    the smallest B_j, as r looks back by at least that much."""
    r = h.copy()
    values, counts = np.unique(budgets, return_counts=True)
    step = int(values[0])
    for start in range(step, r.size, step):
        end = min(start + step, r.size)
        for b, c in zip(values[values < end].tolist(), counts.tolist()):
            lo = max(start, b)
            r[lo:end] += c / n * r[lo - b : end - b]
    return r


def _length(budgets: np.ndarray, n: int, start: int, horizon: int) -> int:
    """The evaluations 0..horizon, or fewer: after ``start`` and k restarts, a
    chain has not succeeded with chance at most (|F|/n)^k; it stops below
    ``EPSILON``."""
    k = math.ceil(math.log(EPSILON) / math.log(budgets.size / n))
    return min(horizon, start + k * int(budgets.max())) + 1


def _sets(runs: pl.DataFrame):
    """Per run set: its size and its B_i in the order of its final levels."""
    return {
        key[0]: (part.height, part["evals_used"].to_numpy())
        for key, part in runs.partition_by("run_set", as_dict=True).items()
    }


def continued(
    hits: pl.DataFrame, runs: pl.DataFrame, key: list[str], eval_var: str
) -> pl.DataFrame:
    """The increments that restarts add to ``hits`` (``key``, ``run_set``, ``k``,
    ``eval_var`` the hitting time, ``inc`` and ``horizon``, in evaluations): per
    curve of ``key`` (within one run set) and k, r - h up to the horizon."""
    sets, parts = _sets(runs), []
    for part in hits.filter(pl.col("k") > 0).partition_by([*key, "k"]):
        n, used = sets[part["run_set"][0]]
        budgets, time = used[: part["k"][0]], part[eval_var].to_numpy().astype(int)
        horizon = int(part["horizon"][0])
        if budgets.min() + time.min() > horizon:  # no restart within the horizon
            continue
        size = _length(budgets, n, int(time.max()), horizon)
        h = np.bincount(time[time < size], part["inc"].to_numpy()[time < size], size)
        extra = _renew(h, budgets, n) - h
        at = np.flatnonzero(extra)
        parts.append(
            part.head(1)
            .select(key)
            .join(pl.DataFrame({eval_var: at, "inc": extra[at]}), how="cross")
        )
    schema = {**hits.select(key).schema, eval_var: pl.Int64, "inc": pl.Float64}
    return pl.concat(parts) if parts else pl.DataFrame(schema=schema)


def gained(
    pieces: pl.DataFrame, runs: pl.DataFrame, budgets: dict, eval_var: str
) -> np.ndarray:
    """Per piece (``run_set``, ``k``, ``eval_var`` its hitting time T): what
    restarts add to the number of its pair's budgets b >= T, as the sum over b of
    E's renewal function at b - T, less 1[b >= T]. ``budgets`` maps each run set
    to its pair's budgets."""
    sets, gain = _sets(runs), np.zeros(pieces.height)
    rows = pieces.with_row_index("_row").filter(pl.col("k") > 0)
    for (s, k), part in rows.partition_by(["run_set", "k"], as_dict=True).items():
        n, used = sets[s]
        time = part[eval_var].to_numpy().astype(int)
        wait = budgets[s][None, :] - time[:, None]
        if used[:k].min() > wait.max():  # no restart before the largest budget
            continue
        delta = np.zeros(_length(used[:k], n, 0, int(wait.max())))
        delta[0] = 1.0
        renewal = np.cumsum(_renew(delta, used[:k], n))
        more = renewal[np.clip(wait, 0, renewal.size - 1)] - 1
        gain[part["_row"].to_numpy()] = np.where(wait >= 0, more, 0.0).sum(axis=1)
    return gain
