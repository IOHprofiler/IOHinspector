"""Statistical comparisons of the groups of ``by``: one-sided two-sample KS tests
with Holm correction, and win fractions (``docs/definitions/comparisons.md``,
CM-1 to CM-5)."""

import math
from collections.abc import Sequence

import numpy as np
import polars as pl

from ..targets import PAIR, get_target_dt
from .distributions import _split
from .fixed_budget import aggregate_convergence
from .fixed_target import aggregate_running_time

__all__ = ["MEASURES", "get_ks_tests", "get_win_fractions", "holm", "ks_test"]

#: R's rule for an exact p-value (``ks.test``): n m below this (CM-1).
EXACT_BELOW = 10000


def _smirnov_upper(q: float, m: int, n: int, tie: list[bool]) -> float:
    """P(D+ >= q) of the one-sided two-sample Smirnov statistic for sizes m and n,
    given the tie pattern of the pooled sample: ``tie[k]`` is True when the k-th
    and (k+1)-th smallest values are equal (k = 1 .. m + n - 1). A line-by-line
    port of R 4.3.3's ``psmirnov_exact_ties_upper`` (``stats/src/ks.c``, the
    recursion of Viehmann 2021)."""
    u = [0.0] * (n + 1)
    for j in range(1, n + 1):
        u[j] = 1.0 if (0.0 - j / n >= q and not tie[j]) else u[j - 1]
    for i in range(1, m + 1):
        r = i / m
        if r >= q and not tie[i]:
            u[0] = 1.0
        for j in range(1, n + 1):
            if r - j / n >= q and not tie[i + j]:
                u[j] = 1.0
            else:
                u[j] = i / (i + j) * u[j] + j / (i + j) * u[j - 1]
    return u[n]


def ks_test(x, y, *, exact: bool | None = None) -> tuple[float, float, str]:
    """One-sided two-sample Kolmogorov-Smirnov test of H1 "x is better than y":
    the values of x are stochastically smaller (CM-1).

    The statistic is D+ = sup (F_x - F_y) over the empirical CDFs. NaN values are
    left out; ``inf`` values (failed runs) count as the largest values, and tie
    with each other. The p-value is exact (``exact=True``) or asymptotic,
    exp(-2 nm/(n+m) D+^2); by default exact when n m < 10000. This is R's
    ``ks.test(x, y, alternative = "greater")``, with one difference: R's exact
    p-value treats equal ``inf`` values as distinct, here they are ties.

    Returns:
        (D+, p, method): ``method`` is ``"exact"`` or ``"asymptotic"``.
    """
    x, y = (np.sort(np.asarray(v, dtype=float)) for v in (x, y))
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    m, n = len(x), len(y)
    if not m or not n:
        raise ValueError("ks_test needs at least one value in each sample")
    pooled = np.sort(np.concatenate([x, y]))
    gap = (
        np.searchsorted(x, pooled, "right") / m
        - np.searchsorted(y, pooled, "right") / n
    )
    d = float(gap.max())  # 0 at the largest value, so D+ >= 0
    if exact is None:
        exact = m * n < EXACT_BELOW
    if not exact:
        p = 1 - (1 - math.exp(-2 * (m * n / (m + n)) * d * d))  # R's psmirnov_asymp
        return d, min(1.0, max(0.0, p)), "asymptotic"
    # R adds 0.5/mn so that rounding does not turn an equality into an inequality
    q = (0.5 + math.floor(d * m * n - 1e-7)) / (m * n)
    tie = [False, *(pooled[1:] == pooled[:-1]).tolist(), False]
    return d, min(1.0, max(0.0, _smirnov_upper(q, m, n, tie))), "exact"


def holm(p) -> np.ndarray:
    """Holm's step-down adjustment of the p-values ``p`` (R's
    ``p.adjust(p, "holm")``). NaN values stay NaN and do not count towards the
    family size."""
    p = np.asarray(p, dtype=float)
    out = p.copy()
    ok = ~np.isnan(p)
    q = p[ok]
    n = len(q)
    if n > 1:
        order = np.argsort(q, kind="stable")
        adjusted = np.minimum(1, np.maximum.accumulate((n - np.arange(n)) * q[order]))
        q = np.empty(n)
        q[order] = adjusted
        out[ok] = q
    return out


def _sample(frame: pl.DataFrame, value: str, sign: float, drop_failures: bool):
    v = frame[value].cast(pl.Float64).drop_nulls().to_numpy() * sign
    v = v[~np.isnan(v)]
    return v[np.isfinite(v)] if drop_failures else v


def get_ks_tests(  # pylint: disable=too-many-arguments,too-many-locals
    sample: pl.DataFrame,
    value: str,
    by: Sequence[str] = ("algorithm_name",),
    *,
    view: Sequence[str] = (),
    higher_is_better: bool = False,
    exact: bool | None = None,
    alpha: float = 0.01,
    drop_failures: bool = False,
) -> pl.DataFrame:
    """Pairwise one-sided KS tests between the groups of ``by``, Holm-corrected,
    per ``view`` (CM-1 to CM-4).

    ``sample`` is the output of ``get_runtime_sample`` (``value="runtime"``,
    ``view=["target"]``) or ``get_value_sample`` (``value="value"``,
    ``view=["budget"]``), or any table with one value per run. Each ``view`` (for
    example a target of one function and dimension) is one family of tests: every
    ordered pair (a, c) of groups is tested for "a is better than c", and the
    p-values of the family are Holm-adjusted together.

    A failed run's runtime is ``inf`` and counts as the worst value (CM-2);
    ``drop_failures=True`` leaves those values out instead, as IOHanalyzer does.
    Null values (no value yet at a budget) are left out. When a has no values and
    c has, p_ac = 1 and p_ca = 0; when neither has, both are null.

    Args:
        sample: one value per run, with the ``by`` and ``view`` columns.
        value: the column of values.
        by: the columns that make a group.
        view: the columns of one family of tests.
        higher_is_better: larger values are better (fixed-budget values under
            maximisation, CM-3); the values are negated.
        exact: the p-value method; None for exact when n m < 10000.
        alpha: the significance level of ``better``.
        drop_failures: leave out infinite values (IOHanalyzer's behaviour).

    Returns:
        pl.DataFrame: one row per view and ordered pair: the ``view`` columns, the
            ``by`` columns of a, the ``by`` columns of c with the suffix ``_vs``,
            ``n`` and ``m`` (the sample sizes), ``d_plus``, ``p_value``,
            ``method``, ``p_adjusted``, ``better`` (p_adjusted <= ``alpha``: the
            dominance graph has an edge a -> c) and ``decision``, the heat map
            value: 1 when a is better than c, -1 when c is better than a, else 0.
    """
    by, view = list(by), list(view)
    sign = -1.0 if higher_is_better else 1.0
    rows = []
    for key, frame in _split(sample, view):
        groups = [
            (g, _sample(f, value, sign, drop_failures)) for g, f in _split(frame, by)
        ]
        tests = {}
        for ga, xa in groups:
            for gc, xc in groups:
                if ga == gc:
                    continue
                if len(xa) and len(xc):
                    tests[ga, gc] = (len(xa), len(xc), *ks_test(xa, xc, exact=exact))
                else:  # IOHanalyzer's rule for empty samples
                    p = None if len(xa) == len(xc) == 0 else float(len(xa) == 0)
                    tests[ga, gc] = (len(xa), len(xc), None, p, None)
        p = holm([np.nan if t[3] is None else t[3] for t in tests.values()])
        adjusted = {
            pair: None if np.isnan(q) else q for pair, q in zip(tests, p.tolist())
        }
        better = {pair: q is not None and q <= alpha for pair, q in adjusted.items()}
        for (ga, gc), test in tests.items():
            decision = int(better[ga, gc]) - int(better[gc, ga])
            rows.append(
                (*key, *ga, *gc, *test, adjusted[ga, gc], better[ga, gc], decision)
            )
    vs = [f"{c}_vs" for c in by]
    numbers = [
        "n",
        "m",
        "d_plus",
        "p_value",
        "method",
        "p_adjusted",
        "better",
        "decision",
    ]
    out = pl.DataFrame(rows, schema=[*view, *by, *vs, *numbers], orient="row")
    types = {
        **sample.select(*view, *by).schema,
        **dict(zip(vs, sample.select(by).dtypes)),
    }
    types |= {"n": pl.Int64, "m": pl.Int64, "d_plus": pl.Float64, "p_value": pl.Float64}
    types |= {"method": pl.String, "p_adjusted": pl.Float64, "better": pl.Boolean}
    return out.cast(types | {"decision": pl.Int8})


#: The performance measures of the win fractions per perspective (CM-5): the
#: column of ``aggregate_running_time`` (fixed target) or ``aggregate_convergence``
#: (fixed budget) that is compared. The first is the default; ``{k}`` is PAR-k's k.
MEASURES = {
    "fixed_target": {
        "ert": "ERT",
        "mean": "mean",
        "par_k": "par_{k:g}",
        "median": "median",
    },
    "fixed_budget": {"mean": "mean", "median": "median", "best": "best"},
}


def _summaries(data, targets, by, column, fixed_target, **options) -> pl.DataFrame:
    """``column`` of the summary of each group of ``by`` per (function, dimension)
    pair at the pair's ``targets``, as ``x`` and ``measure``."""
    summary, x, at = (
        (aggregate_running_time, options["fval_var"], "targets")
        if fixed_target
        else (aggregate_convergence, options["eval_var"], "budgets")
    )
    out = []
    for key, pair in data.partition_by(PAIR, as_dict=True).items():
        values = targets.filter(*(pl.col(c) == v for c, v in zip(PAIR, key)))["target"]
        if values.is_empty():
            continue
        options[at] = values if fixed_target else values.cast(pl.Int64)
        frame = summary(pair, by=by, **options)
        names = (pl.lit(v).alias(c) for c, v in zip(PAIR, key))
        out.append(
            frame.select(
                *names, *by, pl.col(x).alias("x"), pl.col(column).alias("measure")
            )
        )
    return pl.concat(out)


def get_win_fractions(  # pylint: disable=too-many-arguments
    data: pl.DataFrame,
    targets: pl.DataFrame | None = None,
    by: Sequence[str] = ("algorithm_name",),
    *,
    perspective: str = "fixed_target",
    measure: str | None = None,
    k: float = 10,
    maximization: bool = False,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
) -> pl.DataFrame:
    """The fraction of targets (or budgets) on which each group of ``by`` beats
    each other group by a performance measure (CM-5).

    Per (function, dimension) pair and target, a beats c when its ``measure`` is
    better: a win scores 1, a tie (including two undefined measures) 0.5, a loss 0.
    An undefined measure, such as the mean runtime without a success, is the worst
    value. W_ac is the mean score over the targets of a pair, then over the pairs
    both groups have, with equal weight.

    Args:
        data: the runs, with ``run_id``, ``function_id``, ``dimension``,
            ``eval_var``, ``fval_var`` and ``by`` (``evals_used`` for the runs'
            budgets, as ``aggregate_running_time``).
        targets: targets (budgets) per pair: ``function_id``, ``dimension`` and
            ``target``; one per pair from ``get_target_dt`` when None.
        by: the columns that make a group.
        perspective: ``"fixed_target"`` or ``"fixed_budget"``.
        measure: a key of ``MEASURES[perspective]``; fixed target ``ert``
            (default), ``mean`` (of the successful runs, IOHanalyzer's), ``par_k``
            or ``median``; fixed budget ``mean`` (default), ``median`` or ``best``.
        k: the PAR-k factor.
        maximization: whether larger function values are better.
        eval_var: the column of evaluation numbers.
        fval_var: the column of function values.

    Returns:
        pl.DataFrame: one row per ordered pair of groups (a = c included, 0.5):
            the ``by`` columns of a, those of c with the suffix ``_vs``,
            ``win_fraction`` (W_ac) and ``n_pairs``, the (function, dimension)
            pairs it averages.
    """
    if perspective not in MEASURES:
        raise ValueError(f"perspective {perspective!r}: fixed_target or fixed_budget")
    measures = MEASURES[perspective]
    measure = measure or next(iter(measures))
    if measure not in measures:
        raise ValueError(
            f"measure {measure!r} for {perspective}: one of {list(measures)}"
        )
    by = list(by)
    if targets is None:
        targets = get_target_dt(
            data,
            perspective,
            maximization=maximization,
            fval_var=fval_var,
            eval_var=eval_var,
            by=by,
        )
    fixed_target = perspective == "fixed_target"
    higher = maximization and not fixed_target
    summary = _summaries(
        data,
        targets,
        by,
        measures[measure].format(k=k),
        fixed_target,
        maximization=maximization,
        eval_var=eval_var,
        fval_var=fval_var,
        **({"k": k} if fixed_target else {}),
    )
    worst = -np.inf if higher else np.inf
    v = pl.col("measure").cast(pl.Float64).fill_nan(None).fill_null(worst)
    summary = summary.with_columns(v)
    vs = {c: f"{c}_vs" for c in by}
    other = summary.rename({**vs, "measure": "measure_vs"})
    a, c = pl.col("measure"), pl.col("measure_vs")
    win = (a > c) if higher else (a < c)
    scores = summary.join(other, on=[*PAIR, "x"]).with_columns(
        score=win.cast(pl.Float64) + 0.5 * (a == c).cast(pl.Float64)
    )
    groups = [*by, *vs.values()]
    per_pair = scores.group_by(*groups, *PAIR).agg(pl.col("score").mean())
    out = per_pair.group_by(groups).agg(
        win_fraction=pl.col("score").mean(), n_pairs=pl.len().cast(pl.Int64)
    )
    return out.sort(groups)
