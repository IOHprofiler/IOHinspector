"""Runtime and value samples, histograms and violin data
(``docs/definitions/distributions.md``, DI-1 to DI-4)."""

import math
from collections.abc import Iterable, Sequence

import numpy as np
import polars as pl

from ..align import align_data

__all__ = [
    "get_histogram",
    "get_runtime_sample",
    "get_value_sample",
    "get_violin",
    "nclass_fd",
    "pretty",
]

EPS = np.finfo(float).eps


def get_runtime_sample(  # pylint: disable=too-many-arguments
    data: pl.DataFrame,
    targets: Iterable[float],
    by: Sequence[str] = ("algorithm_name",),
    *,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    maximization: bool = False,
) -> pl.DataFrame:
    """The runtime of every run at every target: ``by`` columns, ``run_id``,
    ``target`` and ``runtime`` (``inf`` when the run never reached the target).
    Use one function and dimension at a time."""
    aligned = align_data(
        data,
        targets,
        group_cols=["run_id", *by],
        x_col=fval_var,
        y_col=eval_var,
        maximization=maximization,
    )
    return aligned.rename({fval_var: "target", eval_var: "runtime"}).sort(
        [*by, "run_id", "target"]
    )


def get_value_sample(  # pylint: disable=too-many-arguments
    data: pl.DataFrame,
    budgets: Iterable[int],
    by: Sequence[str] = ("algorithm_name",),
    *,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    maximization: bool = False,
) -> pl.DataFrame:
    """The best value of every run at every budget: ``by`` columns, ``run_id``,
    ``budget`` and ``value`` (null before the run's first logged evaluation,
    AL-5). Use one function and dimension at a time."""
    aligned = align_data(
        data.cast({eval_var: pl.Int64}),
        budgets,
        group_cols=["run_id", *by],
        x_col=eval_var,
        y_col=fval_var,
        maximization=maximization,
    )
    return aligned.rename({eval_var: "budget", fval_var: "value"}).sort(
        [*by, "run_id", "budget"]
    )


def nclass_fd(x: np.ndarray) -> int:
    """Number of histogram classes by R's ``nclass.FD`` (DI-1), with Scott's rule
    when the interquartile range is zero at every level (DI-1, step 1)."""
    n = len(x)
    s = np.sort([float(f"{v:.5g}") for v in x])  # signif(x, 5)
    h = 2 * np.subtract(*np.quantile(s, [0.75, 0.25]))
    a = 1 / 4
    while h == 0 and a / 2 >= 1 / 512:
        a /= 2
        h = np.subtract(*np.quantile(s, [1 - a, a])) / (1 - 2 * a)
    if h == 0 and n > 1:
        h = 3.5 * np.std(x, ddof=1)
    if h <= 0:
        return 1
    return max(1, math.ceil((x.max() - x.min()) / (h * n ** (-1 / 3))))


def pretty(  # pylint: disable=too-many-locals
    lo: float, up: float, n: int, min_n: int = 1
) -> np.ndarray:
    """R's ``pretty(c(lo, up), n, min.n)`` (the C routine ``R_pretty``, R 4.3.3)."""
    # #lizard forgives  (a line-by-line port of R's C routine)
    h, h5, eps = 1.5, 0.5 + 1.5 * 1.5, 1e-10
    dx = up - lo
    cell = 1.0 if dx == 0 and up == 0 else max(abs(lo), abs(up))
    u = 1 + (1 / (1 + h) if h5 >= 1.5 * h + 0.5 else 1.5 / (1 + h5))
    small = (dx == 0 and up == 0) or dx < cell * u * max(1, n) * EPS * 3
    if small:
        cell = (9 + cell / 10 if cell > 10 else cell) * 0.75
        cell = cell / min_n if min_n > 1 else cell
    else:
        cell = dx / n if n > 1 else dx
    cell = max(cell, 20 * np.finfo(float).tiny)
    base = 10.0 ** math.floor(math.log10(cell))
    unit = base
    if 2 * base - cell < h * (cell - unit):
        unit = 2 * base
        if 5 * base - cell < h5 * (cell - unit):
            unit = 5 * base
            if 10 * base - cell < h * (cell - unit):
                unit = 10 * base
    ns, nu = math.floor(lo / unit + eps), math.ceil(up / unit - eps)
    while ns * unit > lo + eps * unit:
        ns -= 1
    while nu * unit < up - eps * unit:
        nu += 1
    k = int(0.5 + nu - ns)
    if k < min_n:
        k = min_n - k
        if ns >= 0:
            nu, ns = nu + k // 2, ns - k // 2 - k % 2
        else:
            ns, nu = ns - k // 2, nu + k // 2 + k % 2
    breaks = np.linspace(ns * unit, nu * unit, nu - ns + 1)
    return np.where(np.abs(breaks) < 1e-14 * unit, 0.0, breaks)


def _breaks(x: np.ndarray) -> np.ndarray:
    return pretty(x.min(), x.max(), nclass_fd(x))


def _counts(x: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """Counts in the bins (a, b], the lowest closed, with R's fuzz of 1e-7
    (``hist.default``)."""
    nb = len(breaks)
    steps = np.diff(breaks)
    if nb > 5:
        fuzz = np.median(steps)
    else:
        fuzz = np.ptp(x) if nb <= 3 else steps.min()
    edges = breaks + 1e-7 * fuzz * np.r_[-1, np.ones(nb - 1)]
    index = np.clip(np.searchsorted(edges, x, side="left") - 1, 0, nb - 2)
    return np.bincount(index, minlength=nb - 1)


def _split(frame: pl.DataFrame, cols: Sequence[str]) -> list:
    """(key, sub-frame) pairs of ``frame`` split on ``cols``; one pair when empty."""
    if not cols:
        return [((), frame)]
    return list(frame.partition_by(cols, as_dict=True).items())


def _finite(sample: pl.DataFrame, value: str) -> pl.DataFrame:
    return sample.filter(pl.col(value).is_finite())  # drops null (undefined) too


def get_histogram(  # pylint: disable=too-many-locals
    sample: pl.DataFrame,
    value: str,
    by: Sequence[str] = ("algorithm_name",),
    view: Sequence[str] = (),
    bins: str = "equal",
) -> pl.DataFrame:
    """Histogram of ``value`` per group of ``by`` and per ``view`` (for example
    ``["target"]``), from the output of ``get_runtime_sample`` or
    ``get_value_sample``. Only defined values count (failed runs have an infinite
    runtime); ``n_failed`` is the number of runs left out. ``bins="equal"`` (DI-4)
    takes the breaks from the pooled values of the groups of one view;
    ``"per_algorithm"`` takes them from each group. A group with fewer than two
    values has no histogram.

    Returns one row per bin: ``view`` and ``by`` columns, ``left``, ``right``,
    ``mid``, ``width``, ``count``, ``n`` and ``n_failed``."""
    if bins not in ("equal", "per_algorithm"):
        raise ValueError(f"bins {bins!r}: equal or per_algorithm")
    by, view = list(by), list(view)
    rows = []
    for key, frame in _split(sample, view):
        pooled = _finite(frame, value)[value].to_numpy()
        for gkey, group in _split(frame, by):
            x = _finite(group, value)[value].to_numpy()
            if len(x) < 2:
                continue
            edges = _breaks(pooled if bins == "equal" else x)
            counts = _counts(x, edges)
            n_failed = len(group) - len(x)
            for i, c in enumerate(counts):
                rows.append((*key, *gkey, edges[i], edges[i + 1], c, len(x), n_failed))
    names = [*view, *by, "left", "right", "count", "n", "n_failed"]
    out = pl.DataFrame(rows, schema=names, orient="row")
    return out.with_columns(
        mid=(pl.col("left") + pl.col("right")) / 2,
        width=pl.col("right") - pl.col("left"),
    ).select(*view, *by, "left", "right", "mid", "width", "count", "n", "n_failed")


def _violin(x: np.ndarray) -> dict:
    """Box and density of one sample (DI-2): Gaussian kernel, Silverman's
    bandwidth, hard span, step at most bw / 3, density scaled by the size."""
    n = len(x)
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    sd = np.std(x, ddof=1) if n > 1 else 0.0
    bw = 1.059 * min(sd, (q3 - q1) / 1.349) * n ** (-0.2)
    if bw > 0:
        grid = np.linspace(x.min(), x.max(), math.ceil(np.ptp(x) / (bw / 3)) + 1)
        z = (grid[:, None] - x[None, :]) / bw
        density = np.exp(-0.5 * z**2).sum(axis=1) / (bw * math.sqrt(2 * math.pi))
    else:  # no spread: a zero-width violin at the value
        grid, density = x[:1], np.zeros(1)
    return {
        "n": n,
        "q1": q1,
        "median": med,
        "q3": q3,
        "bandwidth": bw,
        "x": grid.tolist(),
        "density": density.tolist(),
    }


def get_violin(
    sample: pl.DataFrame,
    value: str,
    by: Sequence[str] = ("algorithm_name",),
    view: Sequence[str] = (),
) -> pl.DataFrame:
    """Violin and box data of ``value`` per group of ``by`` and per ``view``: one
    row with ``n``, ``n_failed``, the quartiles and median (type 7), the
    ``bandwidth`` and the lists ``x`` (the grid) and ``density`` (scaled by
    ``n``). Only defined values count."""
    keys = [*view, *by]
    rows = []
    for key, group in _split(sample, keys):
        x = _finite(group, value)[value].to_numpy()
        if len(x):
            rows.append({**dict(zip(keys, key)), **_violin(x)})
            rows[-1]["n_failed"] = len(group) - len(x)
    return pl.DataFrame(rows)
