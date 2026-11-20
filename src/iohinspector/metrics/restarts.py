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


def _renew(r: np.ndarray, values, counts, n: int, first: int = 0) -> np.ndarray:
    """r = h + 1/n sum_j r(. - B_j) over the evaluations 0, 1, ..., in place of h
    (``r``), with the B_j as their distinct ``values`` in ascending order and their
    ``counts``. In blocks of the smallest B_j, as r looks back by at least that
    much. h is zero before ``first``, and so is r: the terms that would add 0 are
    skipped."""
    step = int(values[0])
    pairs = list(zip(values.tolist(), counts.tolist()))
    for start in range(step * max(1, (first + step) // step), r.size, step):
        end = min(start + step, r.size)
        for b, c in pairs:
            lo = max(start, b + first)
            if lo < end:
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


def _failures(runs: pl.DataFrame, run_set: np.ndarray, k: np.ndarray):
    """Per pair of ``run_set`` and ``k`` (arrays): the size n of the run set and the
    B_i of its k runs with the lowest final levels, as their distinct ``values`` in
    ascending order and their ``counts``, one row per pair (padded with count 0).
    ``runs`` is sorted by run set and final level (``run_sets``)."""
    sets, used = runs["run_set"].to_numpy(), runs["evals_used"].to_numpy()
    first = np.searchsorted(sets, run_set, "left")
    n = np.searchsorted(sets, run_set, "right") - first
    column = np.arange(int(k.max()))
    taken = column < k[:, None]
    budgets = used[np.minimum(first[:, None] + column, used.size - 1)]
    budgets = np.sort(np.where(taken, budgets, np.iinfo(np.int64).max), axis=1)
    new = np.ones(budgets.shape, bool)
    new[:, 1:] = budgets[:, 1:] != budgets[:, :-1]
    rows = np.broadcast_to(np.arange(k.size)[:, None], budgets.shape)[taken]
    cell = rows * column.size + (np.cumsum(new, axis=1) - 1)[taken]
    values = np.zeros(budgets.size, np.int64)
    values[cell] = budgets[taken]
    counts = np.bincount(cell, minlength=budgets.size).reshape(budgets.shape)
    return n, values.reshape(budgets.shape), counts


def _lengths(n, values, counts, start, horizon) -> np.ndarray:
    """``_length`` per row of ``_failures``, from evaluation ``start`` (arrays)."""
    k = counts.sum(axis=1)
    pairs, index = np.unique(np.stack([k, n], axis=1), axis=0, return_inverse=True)
    restarts = [
        math.ceil(math.log(EPSILON) / math.log(a / b)) for a, b in pairs.tolist()
    ]
    largest = values[np.arange(k.size), (counts > 0).sum(axis=1) - 1]
    return np.minimum(horizon, start + np.array(restarts)[index.ravel()] * largest) + 1


def _batches(size: np.ndarray, limit: int = 1 << 24):
    """Consecutive slices of ``size`` with at most ``limit`` in total, or one."""
    ends, first = np.cumsum(size), 0
    while first < size.size:
        start = ends[first] - size[first]
        last = max(first + 1, int(np.searchsorted(ends, start + limit, "right")))
        yield slice(first, last)
        first = last


def continued(
    hits: pl.DataFrame, runs: pl.DataFrame, key: list[str], eval_var: str
) -> pl.DataFrame:
    """The increments that restarts add to ``hits`` (``key``, ``run_set``, ``k``,
    ``eval_var`` the hitting time, ``inc`` and ``horizon``, in evaluations): per
    curve of ``key`` (within one run set) and k, r - h up to the horizon. Rows per
    curve and k in the order of their first hit, each in the order of evaluations."""
    hits = hits.filter(pl.col("k") > 0).with_row_index("_row")
    first = pl.col("_row").min().over([*key, "k"]).rank("dense").cast(pl.Int64) - 1
    hits = hits.with_columns(_curve=first)
    schema = {**hits.select(key).schema, eval_var: pl.Int64, "inc": pl.Float64}
    if hits.is_empty():
        return pl.DataFrame(schema=schema)
    curves = hits.group_by("_curve").agg(pl.all().first()).sort("_curve")
    time, curve = hits[eval_var].to_numpy().astype(int), hits["_curve"].to_numpy()
    lowest = np.full(curves.height, time.max())
    latest = np.zeros(curves.height, time.dtype)
    np.minimum.at(lowest, curve, time)
    np.maximum.at(latest, curve, time)
    horizon = curves["horizon"].to_numpy().astype(int)
    n, values, counts = _failures(
        runs, curves["run_set"].to_numpy(), curves["k"].to_numpy()
    )
    keep = values[:, 0] + lowest <= horizon  # else no restart within the horizon
    size = np.where(keep, _lengths(n, values, counts, latest, horizon), 0)
    parts = []
    for part in _batches(size):
        if not size[part].any():
            continue
        offset = np.cumsum(size[part]) - size[part]
        rows = (curve >= part.start) & (curve < part.stop)
        rows &= time < size[curve]
        at = offset[curve[rows] - part.start] + time[rows]
        h = np.bincount(at, hits["inc"].to_numpy()[rows], int(size[part].sum()))
        r = h.copy()
        for g in np.flatnonzero(size[part]):
            c = part.start + g
            d = np.count_nonzero(counts[c])
            sequence = r[offset[g] : offset[g] + size[c]]
            _renew(sequence, values[c, :d], counts[c, :d], int(n[c]), int(lowest[c]))
        extra = r - h
        at = np.flatnonzero(extra)
        which = np.searchsorted(offset, at, "right") - 1
        parts.append(
            curves.select(key)[which + part.start].with_columns(
                pl.Series(eval_var, at - offset[which]), pl.Series("inc", extra[at])
            )
        )
    return pl.concat([pl.DataFrame(schema=schema), *parts])


def gained(
    pieces: pl.DataFrame, runs: pl.DataFrame, budgets: dict, eval_var: str
) -> np.ndarray:
    """Per piece (``run_set``, ``k``, ``eval_var`` its hitting time T) and budget b
    of its run set (``budgets`` maps each run set to its budgets): what restarts
    add to 1[b >= T], E's renewal function at b - T less 1[b >= T]. One row per
    piece, one column per budget (zero beyond a run set's budgets)."""
    width = max((len(b) for b in budgets.values()), default=0)
    sets, gain = _sets(runs), np.zeros((pieces.height, width))
    rows = pieces.with_row_index("_row").filter(pl.col("k") > 0)
    for (s, k), part in rows.partition_by(["run_set", "k"], as_dict=True).items():
        n, used = sets[s]
        time = part[eval_var].to_numpy().astype(int)
        wait = np.asarray(budgets[s])[None, :] - time[:, None]
        if used[:k].min() > wait.max():  # no restart before the largest budget
            continue
        delta = np.zeros(_length(used[:k], n, 0, int(wait.max())))
        delta[0] = 1.0
        renewal = np.cumsum(_renew(delta, *np.unique(used[:k], return_counts=True), n))
        more = renewal[np.clip(wait, 0, renewal.size - 1)] - 1
        gain[part["_row"].to_numpy(), : wait.shape[1]] = np.where(wait >= 0, more, 0.0)
    return gain
