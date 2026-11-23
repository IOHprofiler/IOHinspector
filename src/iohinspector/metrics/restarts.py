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

By default (``stopped="restart"``) the restarts are sampled, as cocopp's drawSP
does: per run set and k, ``SAMPLES`` draws of the evaluations that the restarts
add before a run that reaches t (``_draws``). The exact renewal
(``stopped="restart_exact"``) costs time in proportion to the horizon
(``exact_cost``).
"""

import math
import warnings
from functools import partial

import numpy as np
import polars as pl

from .utils import runs_of

#: restarts stop where the chance that a chain needs a further one is below this
EPSILON = 1e-17
#: evaluations per chunk of the renewal pass (``continued``), which bounds its memory
CHUNK = 1 << 20
#: draws per run set and number of failed runs of the sampled restarts, and their seed
SAMPLES = 1000
SEED = 20261123
#: ``stopped`` for the exact restarts
EXACT = "restart_exact"
#: the exact restarts warn above this estimate (``exact_cost``)
WARN_SECONDS, WARN_GB = 10.0, 2.0


def restarts(stopped: str) -> bool:
    """Whether ``stopped`` asks for simulated restarts (EC-7, FB-6): ``"restart"``
    (sampled) or ``"restart_exact"``, else ``"carry_forward"``."""
    if stopped not in ("restart", EXACT, "carry_forward"):
        raise ValueError(
            f"stopped is 'restart', {EXACT!r} or 'carry_forward', not {stopped!r}"
        )
    return stopped != "carry_forward"


def run_sets(data: pl.DataFrame, by: list[str], eval_var: str, final: pl.Expr):
    """One row per run: ``run_id``, ``run_set`` (its group of ``by``),
    ``evals_used`` (B_i, at least 1) and ``final``, the run's final level (``final``
    aggregated over its rows), sorted by run set, final level and run id: runs
    that tie on their final level keep one order, so the sampled restarts, which
    index the failed runs, are the same in every process."""
    runs = runs_of(data, by, eval_var).select(
        "run_id", pl.col("run_set").cast(pl.Int64), pl.col("evals_used").clip(1)
    )
    finals = data.group_by("run_id").agg(final.alias("final"))
    return runs.join(finals, on="run_id").sort("run_set", "final", "run_id")


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


def _set_bounds(runs: pl.DataFrame, run_set: np.ndarray):
    """B_i of ``runs`` (sorted by run set, ``run_sets``), and per entry of
    ``run_set`` the position of its set's first run and the set's size n."""
    sets, used = runs["run_set"].to_numpy(), runs["evals_used"].to_numpy()
    first = np.searchsorted(sets, run_set, "left")
    return used, first, np.searchsorted(sets, run_set, "right") - first


def _failures(runs: pl.DataFrame, run_set: np.ndarray, k: np.ndarray):
    """Per pair of ``run_set`` and ``k`` (arrays): the size n of the run set and the
    B_i of its k runs with the lowest final levels, as their distinct ``values`` in
    ascending order and their ``counts``, one row per pair (padded with count 0).
    ``runs`` is sorted by run set and final level (``run_sets``)."""
    used, first, n = _set_bounds(runs, run_set)
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


class RestartCostWarning(UserWarning):
    """Exact simulated restarts (``stopped="restart_exact"``) will be slow or large.
    Raise it as an error to guard against them:
    ``warnings.simplefilter("error", RestartCostWarning)``."""


#: seconds per step of the renewal pass (an evaluation times a failed budget), per
#: evaluation copied between windows, and per evaluation of a windowed curve,
#: measured for get_ecdf on one laptop (Apple M5, ``docs/performance.md``): a
#: rough guide, within a factor of about 2 there
PER_STEP, PER_COPY, PER_WINDOWED = 3.0e-10, 1.5e-10, 1.5e-9


def _cost(size: np.ndarray, values=None, counts=None) -> dict:
    """The estimated cost of the exact renewal pass over curves of ``size``
    evaluations whose failed runs have the distinct budgets ``values`` (rows,
    padded where ``counts`` is 0): ``curves``, ``evaluations``, ``seconds`` and
    ``gigabytes`` (the arrays of the largest windowed curve on top of the data).
    Counted from the curves without computing them."""
    cost = {"curves": int(np.count_nonzero(size)), "evaluations": int(size.sum())}
    if not cost["curves"]:
        return {**cost, "seconds": 0.0, "gigabytes": 0.0}
    used = counts > 0
    steps = np.sum(size * (used & (values < size[:, None])).sum(axis=1))
    history = np.minimum(np.where(used, values, 0).max(axis=1), size)
    long = size > CHUNK
    copied = np.sum(np.ceil(size[long] / CHUNK) * history[long])
    seconds = PER_STEP * steps + PER_COPY * copied + PER_WINDOWED * size[long].sum()
    peak = 4 * history[long].max(initial=0) + 16 * min(CHUNK, int(size.max()))
    return {**cost, "seconds": float(seconds), "gigabytes": float(8 * peak / 2**30)}


def _warn(cost: dict) -> None:
    if cost["seconds"] > WARN_SECONDS or cost["gigabytes"] > WARN_GB:
        warnings.warn(
            f"exact restarts (stopped={EXACT!r}) will take roughly "
            f"{cost['seconds']:.0f} s and {cost['gigabytes']:.1f} GB here: "
            f"{cost['curves']} curves, {cost['evaluations']:.2g} evaluations in "
            "all, which grow with the horizon. Sampled restarts "
            "(stopped='restart') or a smaller horizon cost far less.",
            RestartCostWarning,
            stacklevel=2,
        )


def _draws(first, n, k, used, horizon) -> np.ndarray:
    """``SAMPLES`` draws per row of what restarts add before a run that reaches:
    the sum of M budgets drawn uniformly from the row's k failed runs, M >= 1 with
    P(M = m) = q^(m-1) (1 - q), q = k/n, -1 past the row's ``horizon``. Rows are
    arrays: ``first``, the row's run set's first run in ``used`` (B_i by final
    level, so its failed runs follow), n, k and ``horizon``. As E's renewal
    function less 1 is sum_m q^m P(m budgets sum to at most x), a draw stands for
    the weight k / (n - k) / SAMPLES. The same uniforms serve every row (common
    random numbers): a row's draws depend only on its own runs."""
    rng = np.random.default_rng(SEED)
    count = 1 + np.floor(np.log1p(-rng.random(SAMPLES)) / np.log(k / n)[:, None])
    total = np.zeros(count.shape, np.int64)
    flat, count = total.reshape(-1), count.reshape(-1)
    live, m = np.arange(flat.size), 0
    while live.size:
        m += 1
        u = rng.random(SAMPLES)
        row, column = np.divmod(live, SAMPLES)
        flat[live] += used[first[row] + (u[column] * k[row]).astype(np.int64)]
        live = live[(count[live] > m) & (flat[live] <= horizon[row])]
    return np.where(total <= horizon[:, None], total, -1)


def _shifts(runs: pl.DataFrame, run_set, k, horizon):
    """Per row (arrays ``run_set``, ``k`` and ``horizon``): its distinct draws
    (``_draws``) within the horizon, ascending, at ``values[bounds[i]:bounds[i +
    1]]`` with their ``counts``, and the weight of one draw. ``runs`` is sorted
    by run set and final level (``run_sets``)."""
    used, first, n = _set_bounds(runs, run_set)
    found: list = [], [], []
    step = max(1, CHUNK // SAMPLES)
    for lo in range(0, k.size, step):
        part = slice(lo, lo + step)
        draws = np.sort(_draws(first[part], n[part], k[part], used, horizon[part]))
        row = np.repeat(np.arange(lo, lo + draws.shape[0]), SAMPLES)
        flat = draws.reshape(-1)
        start = np.flatnonzero(
            np.r_[True, (flat[1:] != flat[:-1]) | (row[1:] != row[:-1])]
        )
        keep = flat[start] >= 0
        for out, x in zip(
            found, (row[start], flat[start], np.diff(np.r_[start, flat.size]))
        ):
            out.append(x[keep])
    row, values, counts = (np.concatenate(x) for x in found)
    bounds = np.searchsorted(row, np.arange(k.size + 1))
    return values, counts, bounds, k / (n - k) / SAMPLES


def _sampled(hits, curves, runs, eval_var, cuts, weigh):  # pylint: disable=R0914
    """``continued`` by sampling: a hit of a curve at T with weight w is continued
    at T + s for each distinct draw s of its run set and k (``_shifts``), with w
    times the weight of its draws. (curve, evaluation, increment) arrays, in
    batches of about ``CHUNK`` pairs of a hit and a draw."""
    horizon = curves["horizon"].to_numpy()
    if cuts is not None:
        horizon = np.minimum(horizon, [c[-1] if c.size else -1 for c in cuts])
    rows = curves.select("run_set", "k", horizon=pl.Series(horizon))
    unique = rows.group_by("run_set", "k").agg(pl.col("horizon").max())
    unique = unique.sort("run_set", "k").with_row_index("_u")
    row = rows.join(unique.drop("horizon"), on=["run_set", "k"], maintain_order="left")
    row = row["_u"].to_numpy().astype(np.int64)
    values, counts, sbound, mass = _shifts(
        runs, *(unique[c].to_numpy() for c in ("run_set", "k", "horizon"))
    )
    hits = hits.group_by("_curve", eval_var).agg(pl.col("inc").sum())
    hits = hits.sort("_curve", eval_var)
    curve, time = hits["_curve"].to_numpy(), hits[eval_var].to_numpy().astype(np.int64)
    weight = hits["inc"].to_numpy()
    hbound = np.searchsorted(curve, np.arange(curves.height + 1))
    nh, ns = np.diff(hbound), np.diff(sbound)[row]
    pairs = nh * ns
    if cuts is not None:
        width = np.array([c.size for c in cuts])
        stride = int(max((c[-1] for c in cuts if c.size), default=0)) + 2
        owner = np.repeat(np.arange(len(cuts)), width)
        points = np.concatenate(cuts) if width.sum() else np.zeros(0, np.int64)
        sums = np.zeros(points.size)
    parts = []
    for part in _batches(pairs, CHUNK):
        c = np.repeat(np.arange(part.start, part.stop), pairs[part])
        j = np.arange(c.size) - np.repeat(
            np.cumsum(pairs[part]) - pairs[part], pairs[part]
        )
        hit = hbound[c] + j // ns[c]
        draw = sbound[row[c]] + j % ns[c]
        evaluation = time[hit] + values[draw]
        inc = weight[hit] * counts[draw] * mass[row[c]]
        keep = evaluation <= horizon[c]
        c, evaluation, inc = c[keep], evaluation[keep], inc[keep]
        if cuts is not None:  # each at the first sample at or after it
            at = np.searchsorted(owner * stride + points, c * stride + evaluation)
            sums += np.bincount(at, inc, points.size)
        elif weigh is not None:
            parts.append(weigh(c, evaluation, inc))
        else:
            parts.append((c, evaluation, inc))
    if cuts is not None:
        parts.append((owner, points, sums))
    return parts


def continued(  # pylint: disable=R0913
    hits: pl.DataFrame,
    runs: pl.DataFrame,
    key: list[str],
    eval_var: str,
    at: pl.DataFrame | None = None,
    weighted: pl.Expr | None = None,
    *,
    exact: bool = False,
    cost_only: bool = False,
):
    """The increments that restarts add to ``hits`` (``key``, ``run_set``, ``k``,
    ``eval_var`` the hitting time, ``inc`` and ``horizon``, in evaluations): per
    curve of ``key`` (within one run set) and k, r - h up to the horizon. Rows per
    curve and k in the order of their first hit, each in the order of evaluations.
    With ``at`` (``key`` and ``eval_var``), only their sums up to the evaluations in
    ``at`` of each curve's ``key``: each increment is added at the first of them at
    or after it, and those after the last are left out. Or with ``weighted`` (an
    expression of ``key`` and ``eval_var``), only each curve's sum of the
    increments times it, at evaluation 0 (in rows of a chunk each).

    Sampled (``_sampled``) unless ``exact`` (``_exact``); with ``cost_only`` only
    the estimated cost of the exact computation (``exact_cost``)."""
    hits = hits.filter(pl.col("k") > 0).with_row_index("_row")
    first = pl.col("_row").min().over([*key, "k"]).rank("dense").cast(pl.Int64) - 1
    hits = hits.with_columns(_curve=first).sort("_curve", maintain_order=True)
    schema = {**hits.select(key).schema, eval_var: pl.Int64, "inc": pl.Float64}
    if hits.is_empty():
        return (
            _cost(np.zeros(0, np.int64)) if cost_only else pl.DataFrame(schema=schema)
        )
    curves = hits.group_by("_curve").agg(pl.all().first()).sort("_curve")
    cuts = None
    if at is not None:
        at = at.group_by(key).agg(pl.col(eval_var).unique().sort())
        points = {r[:-1]: np.asarray(r[-1], np.int64) for r in at.rows()}
        none = np.zeros(0, np.int64)
        cuts = [points.get(r, none) for r in curves.select(key).rows()]
    names = curves.select(key)
    weigh = None
    if weighted is not None:
        weigh = partial(_weigh, names, eval_var, weighted)
    if not (exact or cost_only):
        found = _sampled(hits, curves, runs, eval_var, cuts, weigh)
    else:
        found = _exact(hits, curves, runs, eval_var, cuts, weigh, cost_only)
        if cost_only:
            return found
    if not found:
        return pl.DataFrame(schema=schema)
    which, evaluation, inc = (np.concatenate(x) for x in zip(*found))
    keep = np.flatnonzero(inc)
    order = keep[np.argsort(which[keep], kind="stable")]  # the long curves came last
    return names[which[order]].with_columns(
        pl.Series(eval_var, evaluation[order], pl.Int64), pl.Series("inc", inc[order])
    )


def _weigh(names, eval_var, weighted, which, evaluation, inc):
    """Per curve in ``which`` (rows of ``names``, its ``key``), the sum of its
    ``inc`` times ``weighted``, at evaluation 0, so that memory stays bounded."""
    rows = names[which].with_columns(
        pl.Series("_c", which),
        pl.Series(eval_var, evaluation, pl.Int64),
        pl.Series("inc", inc),
    )
    rows = rows.group_by("_c").agg((pl.col("inc") * weighted).sum())
    return (
        rows["_c"].to_numpy(),
        np.zeros(rows.height, np.int64),
        rows["inc"].to_numpy(),
    )


def _exact(  # pylint: disable=R0913,R0914,R0917
    hits, curves, runs, eval_var, cuts, weigh, cost_only
):
    """``continued`` by the renewal pass: (curve, evaluation, increment) arrays,
    or with ``cost_only`` its estimated cost. Curves of up to ``CHUNK``
    evaluations are computed together in arrays of about that size, longer ones
    one at a time in windows (``_windows``)."""
    time, curve = hits[eval_var].to_numpy().astype(int), hits["_curve"].to_numpy()
    weight = hits["inc"].to_numpy()
    bounds = np.searchsorted(curve, np.arange(curves.height + 1))
    lowest = np.minimum.reduceat(time, bounds[:-1])
    latest = np.maximum.reduceat(time, bounds[:-1])
    n, values, counts = _failures(
        runs, curves["run_set"].to_numpy(), curves["k"].to_numpy()
    )
    size = _lengths(n, values, counts, latest, curves["horizon"].to_numpy())
    if cuts is not None:
        size = np.minimum(size, [c[-1] + 1 if c.size else 0 for c in cuts])
    size = np.where(values[:, 0] + lowest < size, size, 0)  # else no restart in it
    cost = _cost(size, values, counts)
    if cost_only:
        return cost
    _warn(cost)

    def terms(c):
        d = np.count_nonzero(counts[c])
        return list(zip(values[c, :d].tolist(), (counts[c, :d] / n[c]).tolist()))

    def add(which, evaluation, inc):
        parts.append(
            (which, evaluation, inc) if weigh is None else weigh(which, evaluation, inc)
        )

    parts: list = []

    small = np.where(size <= CHUNK, size, 0)
    for part in _batches(small, CHUNK):
        live = np.flatnonzero(small[part])
        if not live.size:
            continue
        offset = np.cumsum(small[part]) - small[part]
        rows = (curve >= part.start) & (curve < part.stop) & (time < small[curve])
        place = offset[curve[rows] - part.start] + time[rows]
        h = np.bincount(place, weight[rows], int(small[part].sum()))
        r = h.copy()
        for c in part.start + live:
            first, end = int(lowest[c]), int(size[c])
            _renew(r, terms(c), first, 0, end, int(offset[c - part.start]))
        extra = r - h
        if cuts is None:
            place = np.flatnonzero(extra)
            g = np.searchsorted(offset, place, "right") - 1
            add(part.start + g, place - offset[g], extra[place])
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
        if cuts is None:
            for start, extra in chunks:
                place = np.flatnonzero(extra)
                add(np.full(place.size, c), start + place, extra[place])
        else:
            sums = _binned(chunks, _bins(cuts[c], size[c]))
            parts.append((np.full(sums.size, c), cuts[c][: sums.size], sums))
    return parts


def gained(
    pieces: pl.DataFrame,
    runs: pl.DataFrame,
    budgets: dict,
    eval_var: str,
    *,
    exact: bool = False,
) -> np.ndarray:
    """Per piece (``run_set``, ``k``, ``eval_var`` its hitting time T) and budget b
    of its run set (``budgets`` maps each run set to its budgets): what restarts
    add to 1[b >= T], E's renewal function at b - T less 1[b >= T]. One row per
    piece, one column per budget (zero beyond a run set's budgets). Sampled
    (``_shifts``) unless ``exact``."""
    width = max((len(b) for b in budgets.values()), default=0)
    gain = np.zeros((pieces.height, width))
    rows = pieces.with_row_index("_row").filter(pl.col("k") > 0)
    if rows.is_empty():
        return gain
    table = np.full((rows.height, width), -1, np.int64)  # each row's budgets
    sets = rows["run_set"].to_numpy()
    for s in np.unique(sets):
        b = np.asarray(budgets[s], np.int64)
        table[sets == s, : b.size] = b
    wait = table - rows[eval_var].to_numpy().astype(np.int64)[:, None]
    late = (table < 0) | (wait < 0)
    if exact:
        more = _gained_exact(rows, runs, wait, late)
    else:
        groups = rows.select("run_set", "k").unique().sort("run_set", "k")
        groups = groups.with_columns(_last=pl.Series(_last(rows, wait, groups)))
        group = (
            rows.join(
                groups.with_row_index("_g"), on=["run_set", "k"], maintain_order="left"
            )["_g"]
            .to_numpy()
            .astype(np.int64)
        )
        values, counts, bounds, mass = _shifts(
            runs, *(groups[c].to_numpy() for c in ("run_set", "k", "_last"))
        )
        stride = int(groups["_last"].max()) + 2
        keys = np.repeat(np.arange(groups.height), np.diff(bounds)) * stride + values
        within = np.clip(wait, 0, stride - 1) + group[:, None] * stride
        total = np.r_[0, np.cumsum(counts)]
        found = total[np.searchsorted(keys, within, "right")]
        more = mass[group][:, None] * (found - total[bounds[group]][:, None])
    gain[rows["_row"].to_numpy()] = np.where(late, 0.0, more)
    return gain


def _last(rows: pl.DataFrame, wait: np.ndarray, groups: pl.DataFrame) -> np.ndarray:
    """Per group (``run_set``, ``k``) of ``rows``: the largest ``wait`` of its rows."""
    largest = rows.select("run_set", "k", _w=pl.Series(wait.max(axis=1)))
    largest = largest.group_by("run_set", "k").agg(pl.col("_w").max())
    return groups.join(largest, on=["run_set", "k"], maintain_order="left")[
        "_w"
    ].to_numpy()


def _gained_exact(rows, runs, wait, late) -> np.ndarray:  # pylint: disable=R0914
    """``gained`` by the renewal pass, per run set and k."""
    sets, more = _sets(runs), np.zeros(wait.shape)
    jobs, sizes, budgets = [], [], []
    for (s, k), part in (
        rows.with_row_index("_i").partition_by(["run_set", "k"], as_dict=True).items()
    ):
        n, used = sets[s]
        index = part["_i"].to_numpy()
        largest = int(np.where(late[index], -1, wait[index]).max())
        if used[:k].min() > largest:  # no restart before the largest budget
            continue
        jobs.append((index, n, used[:k]))
        sizes.append(_length(used[:k], n, 0, largest))
        budgets.append(np.unique(used[:k]))
    if not jobs:
        return more
    padded = np.zeros((len(jobs), max(b.size for b in budgets)), np.int64)
    for i, b in enumerate(budgets):
        padded[i, : b.size] = b
    _warn(_cost(np.array(sizes), padded, (padded > 0).astype(int)))
    for (index, n, used), size in zip(jobs, sizes):
        values, counts = np.unique(used, return_counts=True)
        terms = list(zip(values.tolist(), (counts / n).tolist()))
        at, inverse = np.unique(np.clip(wait[index], 0, None), return_inverse=True)
        chunks = _windows(np.zeros(1, np.int64), np.ones(1), terms, size)
        sums = np.cumsum(_binned(chunks, _bins(at, size)))
        sums = np.r_[sums, np.full(at.size - sums.size, sums[-1])]
        more[index] = sums[inverse.reshape(index.size, -1)]
    return more
