"""Cumulative difference plot (``docs/definitions/cdp.md``, CD-1 to CD-4): a port of
RVCompare 0.1.8's ``get_Y_AB_bounds_bootstrap`` (CC0) with a seed."""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..targets import PAIR
from .distributions import get_runtime_sample, get_value_sample

__all__ = ["CumulativeDifference", "cumulative_difference", "get_cumulative_difference"]

POINTS = 2000  # RVCompare's nOfEstimationPoints
CURVES = ["y_a", "y_b", "diff"]


def _densities(a: np.ndarray, b: np.ndarray, epsilon: float) -> np.ndarray:
    """Per row of the samples ``a`` and ``b`` (equal size n): the fraction of A in
    the tie group of each position of the sorted pooled sample (RVCompare's
    ``rank_interval_multA``). Values closer than ``epsilon``, or both infinite, tie."""
    rows, n = a.shape
    pooled = np.concatenate([a, b], axis=1)
    order = np.argsort(pooled, axis=1, kind="stable")
    values = np.take_along_axis(pooled, order, axis=1)
    with np.errstate(invalid="ignore"):  # inf - inf
        tie = (values[:, 1:] == values[:, :-1]) | (np.diff(values, axis=1) < epsilon)
    group = np.cumsum(np.c_[np.ones(rows, bool), ~tie], axis=1) - 1
    group += np.arange(rows)[:, None] * 2 * n  # one set of groups per row
    from_a = (order < n).ravel()
    size = np.bincount(group.ravel())
    return (np.bincount(group.ravel(), weights=from_a) / np.maximum(size, 1))[group]


def _on_grid(mult: np.ndarray) -> np.ndarray:
    """RVCompare's ``helper_from_ranks_to_integrable_values`` and
    ``helperTrapezoidRule`` per row: the density on ``POINTS`` points of [0, 1],
    normalised, integrated with the trapezoid rule from 0 to each point."""
    ranks = mult.shape[1]  # r_max + 1 = 2n
    j_max = POINTS - 1
    p = np.arange(POINTS) / j_max
    r = np.searchsorted(np.arange(ranks) / ranks, p, side="right") - 1  # last k/2n <= p
    steps, at = np.unique(r, return_inverse=True)
    previous = np.r_[-1, steps[:-1]]
    total = np.c_[np.zeros(len(mult)), np.cumsum(mult, axis=1)]
    # the mean of mult over the ranks passed since the previous step
    density = ((total[:, steps + 1] - total[:, previous + 1]) / (steps - previous))[
        :, at
    ]
    density = density / density.sum(axis=1, keepdims=True) * j_max
    trapezoid = np.c_[np.zeros(len(mult)), (density[:, :-1] + density[:, 1:]) / 2]
    return np.cumsum(trapezoid * (1 / j_max), axis=1)


def cumulative_difference(  # pylint: disable=too-many-arguments
    x_a: Sequence[float] | np.ndarray,
    x_b: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    epsilon: float = 1e-80,
    n_bootstrap: int = 1000,
    seed: int | np.random.Generator | None = None,
) -> pl.DataFrame:
    """The cumulative distributions of the quantile random variables Y_A and Y_B of
    two samples (smaller is better) and their difference, with bootstrap bands
    [Arza2023] (``docs/definitions/cdp.md``, CD-1).

    The pooled sample is ranked; values closer than ``epsilon`` (or both infinite)
    share a rank. The density of Y_A at position p of [0, 1] is the fraction of A
    among the observations at rank ⌊2n p⌋, so the densities of Y_A and Y_B sum to 2.
    As RVCompare 0.1.8, it is evaluated on 2000 points, normalised and integrated
    with the trapezoid rule. The first of ``n_bootstrap`` samples is the observed
    one; the others draw n values with replacement from each sample. Each band is
    the α'/2 and 1 − α'/2 quantile (type 7) of the bootstrap curves, with
    α' = 1 − sqrt(1 − α).

    Args:
        x_a, x_b: the samples, of equal size, oriented so that smaller is better;
            ``inf`` is the worst value. NaN is rejected.
        alpha: 1 minus the joint confidence of the two bands.
        epsilon: values closer than this are equal (CD-4).
        n_bootstrap: bootstrap samples, the observed one included.
        seed: the random seed, or a numpy Generator.

    Returns:
        pl.DataFrame: ``p`` and, for each of ``y_a``, ``y_b`` and ``diff``
        (y_a − y_b), the estimate and its ``_lower`` and ``_upper`` band.
    """
    a, b = np.asarray(x_a, dtype=float), np.asarray(x_b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError(
            f"the samples must be non-empty and of equal size, not {len(a)} and {len(b)}"
        )
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("the samples contain NaN")
    if not 0 < epsilon <= 0.1 or not 0 < alpha <= 1 or n_bootstrap < 1:
        raise ValueError("need 0 < epsilon <= 0.1, 0 < alpha <= 1 and n_bootstrap >= 1")
    rng = np.random.default_rng(seed)
    n = len(a)
    draw = rng.integers(0, n, (2, n_bootstrap - 1, n))
    a = np.r_[a[None], a[draw[0]]]
    b = np.r_[b[None], b[draw[1]]]
    mult = _densities(a, b, epsilon)
    y_a, y_b = _on_grid(mult), _on_grid(1 - mult)
    level = 1 - np.sqrt(1 - alpha)
    columns = {"p": np.arange(POINTS) / (POINTS - 1)}
    for name, curves in zip(CURVES, [y_a, y_b, y_a - y_b]):
        lower, upper = np.quantile(curves, [level / 2, 1 - level / 2], axis=0)
        columns |= {name: curves[0], f"{name}_lower": lower, f"{name}_upper": upper}
    return pl.DataFrame(columns)


@dataclass(frozen=True)
class CumulativeDifference:
    """The result of ``get_cumulative_difference``.

    Attributes:
        curves: the output of ``cumulative_difference`` for groups A and B.
        runs: one row per run of the two groups: ``run_id``, ``by``, ``group``
            (``"A"`` or ``"B"``), ``value`` (the sample value, smaller is better,
            ``inf`` for a failed run) and ``used`` (false for a run left out by the
            subsample, CD-3).
    """

    curves: pl.DataFrame
    runs: pl.DataFrame

    @property
    def subsampled(self) -> bool:
        """Whether runs were left out to make the samples equally large (CD-3)."""
        return not self.runs["used"].all()


def _sample(data, target, budget, by, maximization, fval_var, eval_var):  # pylint: disable=R0913,R0917
    """One value per run, smaller is better: the runtime to ``target`` (``inf`` when
    not reached, CD-2) or the best value at ``budget`` (negated when maximising;
    ``inf`` before the first evaluation)."""
    options = {"by": by, "fval_var": fval_var, "eval_var": eval_var}
    if target is not None:
        sample = get_runtime_sample(
            data, [target], maximization=maximization, **options
        )
        return sample.select("run_id", *by, value=pl.col("runtime").cast(pl.Float64))
    sample = get_value_sample(data, [budget], maximization=maximization, **options)
    value = pl.col("value").cast(pl.Float64)
    value = (-value if maximization else value).fill_null(float("inf"))
    return sample.select("run_id", *by, value=value)


def get_cumulative_difference(  # pylint: disable=too-many-arguments,too-many-locals
    data: pl.DataFrame,
    *,
    target: float | None = None,
    budget: int | None = None,
    by: Sequence[str] = ("algorithm_name",),
    maximization: bool = False,
    subsample: bool = True,
    seed: int | None = None,
    alpha: float = 0.05,
    epsilon: float = 1e-80,
    n_bootstrap: int = 1000,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
) -> CumulativeDifference:
    """The cumulative difference plot of two groups of runs on one (function,
    dimension) pair (``docs/definitions/cdp.md``).

    Give ``target`` for the fixed-target view (runtimes; a run that never reaches
    the target counts as ``inf``, CD-2) or ``budget`` for the fixed-budget view (the
    best value at the budget; negated when maximising, so smaller is better).
    ``by`` must make exactly two groups: A is the first in sorted order, B the
    second. When they have different numbers of runs, the larger group is
    subsampled without replacement to the size of the smaller, with ``seed``, and a
    warning says so (CD-3); ``subsample=False`` raises instead. ``result.runs``
    lists every run and whether it was used.

    Args:
        data: the logged rows of one (function, dimension) pair.
        target, budget: exactly one of them.
        by: the columns that make the two groups (D17).
        maximization: larger values are better.
        subsample: subsample the larger group (default) or raise.
        seed: the random seed of the subsample and the bootstrap.
        alpha, epsilon, n_bootstrap: as in ``cumulative_difference``.
        fval_var, eval_var: the value and evaluations columns.

    Returns:
        CumulativeDifference: ``curves`` and ``runs``.
    """
    if (target is None) == (budget is None):
        raise ValueError("give either target or budget")
    by = list(by)
    pairs = [c for c in PAIR if c in data.columns]
    if pairs and data.select(pairs).n_unique() > 1:
        raise ValueError("use one (function, dimension) pair at a time")
    runs = _sample(data, target, budget, by, maximization, fval_var, eval_var)
    groups = runs.select(by).unique().sort(by, nulls_last=True)
    if groups.height != 2:
        raise ValueError(f"by must make exactly two groups, not {groups.height}")
    groups = groups.with_columns(group=pl.Series(["A", "B"]))
    runs = runs.join(groups, on=by, nulls_equal=True).sort("group", "run_id")
    sizes = dict(runs.group_by("group").len().iter_rows())
    n = min(sizes.values())
    rng = np.random.default_rng(seed)
    used = pl.Series("used", np.ones(runs.height, dtype=bool))
    if sizes["A"] != sizes["B"]:
        if not subsample:
            raise ValueError(
                f"the groups have {sizes['A']} and {sizes['B']} runs; pass "
                "subsample=True to subsample the larger one"
            )
        large = "A" if sizes["A"] > n else "B"
        offset = 0 if large == "A" else sizes["A"]
        keep = rng.choice(sizes[large], n, replace=False)
        mask = np.ones(runs.height, dtype=bool)
        mask[offset : offset + sizes[large]] = False
        mask[offset + keep] = True
        used = pl.Series("used", mask)
        warnings.warn(
            f"group {large} has {sizes[large]} runs and the other {n}: "
            f"{sizes[large] - n} runs of {large} were left out at random (seed {seed}); "
            "see result.runs",
            stacklevel=2,
        )
    runs = runs.with_columns(used)
    chosen = runs.filter("used")
    curves = cumulative_difference(
        chosen.filter(group="A")["value"].to_numpy(),
        chosen.filter(group="B")["value"].to_numpy(),
        alpha=alpha,
        epsilon=epsilon,
        n_bootstrap=n_bootstrap,
        seed=rng,
    )
    return CumulativeDifference(
        curves, runs.select("run_id", *by, "group", "value", "used")
    )
