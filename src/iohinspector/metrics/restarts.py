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
#: evaluations per chunk of the renewal pass (``continued``), which bounds its memory
CHUNK = 1 << 20


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


def _renew(r: np.ndarray, terms, first: int, begin: int, end: int, at: int = 0):
    """r = h + sum_j rate_j r(. - B_j) in place over the evaluations begin..end-1,
    r[i] being evaluation i - ``at``; there r holds h, before it r. ``terms`` are
    the (B_j, rate_j) by ascending B_j. In blocks of the smallest B_j, as r looks
    back by at least that much. h is zero before ``first``, and so is r: the terms
    that would add 0 are skipped."""
    step = terms[0][0]
    for block in range(max(begin, first + step), end, step):
        stop = min(block + step, end)
        for b, rate in terms:
            lo = max(block, b + first)
            if lo >= stop:
                break
            r[lo + at : stop + at] += rate * r[lo - b + at : stop - b + at]


def _windows(times, weights, terms, size: int):
    """The increments r - h that restarts add over the evaluations 0..size-1
    (``_renew``, h being ``weights`` at ``times``), in chunks ``(start, array)`` of
    ``CHUNK``: only the last max B_j values of r are kept, so memory does not grow
    with ``size``."""
    first, history = int(times.min()), min(terms[-1][0], size)
    r = np.zeros(history + min(CHUNK, size))  # r[i]: evaluation start - history + i
    for start in range(0, size, CHUNK):
        end = min(start + CHUNK, size)
        inside = (times >= start) & (times < end)
        h = np.bincount(times[inside] - start, weights[inside], end - start)
        r[history : history + h.size] = h
        _renew(r, terms, first, start, end, history - start)
        yield start, r[history : history + h.size] - h
        r[:history] = r[h.size : h.size + history].copy()


def _bins(cuts: np.ndarray, size: int) -> np.ndarray:
    """The first evaluation of each bin of ``cuts`` (ascending, distinct, the
    last at least size - 1) within 0..size-1: bin v holds the evaluations after
    cut v-1 up to cut v, and the bins after the one of size - 1 are empty."""
    return np.concatenate(([0], cuts[cuts < size - 1] + 1))


def _binned(chunks, starts: np.ndarray) -> np.ndarray:
    """The sum of the ``chunks`` (``(start, array)`` over evaluations) per bin, bin
    k from evaluation ``starts[k]`` (ascending, from 0) to the next."""
    sums = np.zeros(starts.size)
    for start, extra in chunks:
        k = int(np.searchsorted(starts, start, "right")) - 1
        j = int(np.searchsorted(starts, start + extra.size))
        sums[k:j] += np.add.reduceat(extra, np.r_[0, starts[k + 1 : j] - start])
    return sums


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


def _batches(size: np.ndarray, limit: int):
    """Consecutive slices of ``size`` with at most ``limit`` in total, or one."""
    ends, first = np.cumsum(size), 0
    while first < size.size:
        start = ends[first] - size[first]
        last = max(first + 1, int(np.searchsorted(ends, start + limit, "right")))
        yield slice(first, last)
        first = last


def continued(  # pylint: disable=R0914
    hits: pl.DataFrame,
    runs: pl.DataFrame,
    key: list[str],
    eval_var: str,
    at: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """The increments that restarts add to ``hits`` (``key``, ``run_set``, ``k``,
    ``eval_var`` the hitting time, ``inc`` and ``horizon``, in evaluations): per
    curve of ``key`` (within one run set) and k, r - h up to the horizon. Rows per
    curve and k in the order of their first hit, each in the order of evaluations.
    With ``at`` (``key`` and ``eval_var``), only their sums up to the evaluations in
    ``at`` of each curve's ``key``: each increment is added at the first of them at
    or after it, and those after the last are left out.

    Curves of up to ``CHUNK`` evaluations are computed together in arrays of about
    that size, longer ones one at a time in windows (``_windows``)."""
    hits = hits.filter(pl.col("k") > 0).with_row_index("_row")
    first = pl.col("_row").min().over([*key, "k"]).rank("dense").cast(pl.Int64) - 1
    hits = hits.with_columns(_curve=first).sort("_curve", maintain_order=True)
    schema = {**hits.select(key).schema, eval_var: pl.Int64, "inc": pl.Float64}
    if hits.is_empty():
        return pl.DataFrame(schema=schema)
    curves = hits.group_by("_curve").agg(pl.all().first()).sort("_curve")
    time, curve = hits[eval_var].to_numpy().astype(int), hits["_curve"].to_numpy()
    weight = hits["inc"].to_numpy()
    bounds = np.searchsorted(curve, np.arange(curves.height + 1))
    lowest = np.minimum.reduceat(time, bounds[:-1])
    latest = np.maximum.reduceat(time, bounds[:-1])
    n, values, counts = _failures(
        runs, curves["run_set"].to_numpy(), curves["k"].to_numpy()
    )
    size = _lengths(n, values, counts, latest, curves["horizon"].to_numpy())
    cuts: list[np.ndarray] = []
    if at is not None:
        at = at.group_by(key).agg(pl.col(eval_var).unique().sort())
        points = {r[:-1]: np.asarray(r[-1], np.int64) for r in at.rows()}
        none = np.zeros(0, np.int64)
        cuts = [points.get(r, none) for r in curves.select(key).rows()]
        size = np.minimum(size, [c[-1] + 1 if c.size else 0 for c in cuts])
    size = np.where(values[:, 0] + lowest < size, size, 0)  # else no restart in it

    def terms(c):
        d = np.count_nonzero(counts[c])
        return list(zip(values[c, :d].tolist(), (counts[c, :d] / n[c]).tolist()))

    parts = []
    small = np.where(size <= CHUNK, size, 0)
    for part in _batches(small, CHUNK):
        live = np.flatnonzero(small[part])
        if not live.size:
            continue
        offset = np.cumsum(small[part]) - small[part]
        rows = (curve >= part.start) & (curve < part.stop) & (time < size[curve])
        place = offset[curve[rows] - part.start] + time[rows]
        h = np.bincount(place, weight[rows], int(small[part].sum()))
        r = h.copy()
        for c in part.start + live:
            first, end = int(lowest[c]), int(size[c])
            _renew(r, terms(c), first, 0, end, int(offset[c - part.start]))
        extra = r - h
        if at is None:
            place = np.flatnonzero(extra)
            g = np.searchsorted(offset, place, "right") - 1
            parts.append((part.start + g, place - offset[g], extra[place]))
            continue
        starts = [_bins(cuts[c], size[c]) for c in part.start + live]
        width = [x.size for x in starts]
        sums = np.add.reduceat(
            extra, np.concatenate(starts) + np.repeat(offset[live], width)
        )
        evaluation = np.concatenate(
            [cuts[c][:w] for c, w in zip(part.start + live, width)]
        )
        parts.append((np.repeat(part.start + live, width), evaluation, sums))
    for c in np.flatnonzero(size > CHUNK):
        span = slice(bounds[c], bounds[c + 1])
        chunks = _windows(time[span], weight[span], terms(c), int(size[c]))
        if at is None:
            for start, extra in chunks:
                place = np.flatnonzero(extra)
                parts.append((np.full(place.size, c), start + place, extra[place]))
        else:
            sums = _binned(chunks, _bins(cuts[c], size[c]))
            parts.append((np.full(sums.size, c), cuts[c][: sums.size], sums))
    if not parts:
        return pl.DataFrame(schema=schema)
    which, evaluation, inc = (np.concatenate(x) for x in zip(*parts))
    keep = np.flatnonzero(inc)
    order = keep[np.argsort(which[keep], kind="stable")]  # the long curves came last
    return curves.select(key)[which[order]].with_columns(
        pl.Series(eval_var, evaluation[order], pl.Int64), pl.Series("inc", inc[order])
    )


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
        late = wait < 0
        if used[:k].min() > wait.max():  # no restart before the largest budget
            continue
        size = _length(used[:k], n, 0, int(wait.max()))
        values, counts = np.unique(used[:k], return_counts=True)
        terms = list(zip(values.tolist(), (counts / n).tolist()))
        wait = np.clip(wait, 0, None)  # (more is 0 where b < T)
        at, index = np.unique(wait, return_inverse=True)
        chunks = _windows(np.zeros(1, np.int64), np.ones(1), terms, size)
        sums = np.cumsum(_binned(chunks, _bins(at, size)))
        more = np.r_[sums, np.full(at.size - sums.size, sums[-1])]
        more = more[index.reshape(wait.shape)]
        gain[part["_row"].to_numpy(), : wait.shape[1]] = np.where(late, 0.0, more)
    return gain
