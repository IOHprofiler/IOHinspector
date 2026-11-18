"""Target and budget grids (``docs/definitions/targets.md``, TG-*).

Ports of IOHanalyzer's ``seq_FV``, ``seq_RT``, ``get_ECDF_targets`` and
``get_target_dt``, plus the COCO ``bbob`` grid and the default bounds of the
IOHexperimenter suite registry (TG-8).
"""

import warnings
from collections.abc import Iterable, Sequence
from decimal import Decimal, localcontext

import numpy as np
import polars as pl

__all__ = [
    "bbob_targets",
    "get_ecdf_budgets",
    "get_ecdf_targets",
    "get_target_dt",
    "seq_fv",
    "seq_rt",
]

PAIR = ["function_id", "dimension"]
#: Lower end of log-spaced targets when the data has no positive value (TG-3).
LOG_FLOOR = 1e-12
#: Default target bounds (lower, upper) per suite, upper-case name (TG-8): the
#: ``target_bounds`` of the IOHexperimenter suite registry (ioh 0.4.2) for the
#: suites that give both. They are log-scale precisions f - f_opt, which the
#: IOHexperimenter and COCO readers store in ``raw_y``. ``tests/test_targets.py``
#: checks this table against the registry file.
SUITE_BOUNDS = {"BBOB": (1e-8, 1e2), "SBOX": (1e-8, 1e2)}


def spaced(lower: float, upper: float, n: int, log: bool = False) -> np.ndarray:
    """``n`` values evenly spaced from ``lower`` to ``upper`` (in log10 when ``log``),
    both ends included, sorted, without duplicates."""
    if log and (lower <= 0 or upper <= 0):
        raise ValueError(f"log spacing needs positive bounds, got {lower}, {upper}")
    ends = np.log10([lower, upper]) if log else (lower, upper)
    values = np.linspace(*ends, n)
    values = 10.0**values if log else values
    if n > 1:  # the ends exactly: 10 ** log10(x) can be 1 ulp off
        values[[0, -1]] = lower, upper
    return np.unique(values)


def seq_fv(
    values: Iterable[float],
    n: int = 10,
    *,
    log: bool = True,
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    """``n`` targets from the smallest to the largest finite value of ``values``
    (TG-1), evenly spaced in log10 or linearly. ``lower`` and ``upper`` replace the
    observed ends. Under log spacing a smallest value <= 0 becomes the smallest
    positive value, or 1e-12 with a warning when there is none (TG-3)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    lo = v.min() if lower is None else lower
    hi = v.max() if upper is None else upper
    if log and lower is None and lo <= 0:
        positive = v[v > 0]
        if positive.size == 0:
            warnings.warn("no positive value: log-spaced targets start at 1e-12")
        lo = positive.min() if positive.size else LOG_FLOOR
    if log and upper is None and hi <= 0:
        hi = lo
    return spaced(lo, hi, n, log)


def seq_rt(
    evaluations: Iterable[int],
    n: int = 50,
    *,
    log: bool = True,
    lower: int | None = None,
    upper: int | None = None,
) -> np.ndarray:
    """``n`` budgets from the smallest to the largest of ``evaluations``, rounded
    to the nearest integer (half to even) without duplicates (TG-5)."""
    e = np.asarray(evaluations, dtype=float)
    lo = e.min() if lower is None else lower
    hi = e.max() if upper is None else upper
    return np.unique(np.round(spaced(lo, hi, n, log)).astype(np.int64))


def bbob_targets() -> np.ndarray:
    """The 51 COCO targets 10^(2 - k/5), k = 0..50, from 1e2 down to 1e-8.

    Computed in decimal and rounded once to float, so every target is the
    correctly rounded value on every platform (numpy's power can be 1 ulp off).
    """
    with localcontext() as ctx:
        ctx.prec = 40
        exponents = (Decimal(10 - k) / 5 for k in range(51))
        return np.array([float(Decimal(10) ** e) for e in exponents])


def _suite_bounds(suites: Iterable[str | None]) -> tuple[float | None, float | None]:
    """The registry's bounds of a pair whose runs are all of one suite in
    ``SUITE_BOUNDS`` (TG-8), else none."""
    names = {str(s).upper() for s in suites}
    if len(names) != 1:
        return None, None
    return SUITE_BOUNDS.get(names.pop(), (None, None))


def get_ecdf_targets(  # pylint: disable=too-many-arguments
    data: pl.DataFrame,
    kind: str = "log-linear",
    *,
    n: int = 10,
    lower: float | None = None,
    upper: float | None = None,
    fval_var: str = "raw_y",
) -> pl.DataFrame:
    """Targets per (function, dimension) pair (TG-1, TG-2): ``kind``
    ``"log-linear"`` or ``"linear"`` (``n`` targets over the pair's observed
    values), or ``"bbob"`` (the 51 COCO targets).

    A pair whose runs are all of a suite with registry bounds (``SUITE_BOUNDS``:
    BBOB and SBOX, 1e-8 to 1e2, matched without regard to case) spans those
    instead of its observed values (TG-8); the suite comes from a ``suite`` column
    of ``data``. ``lower`` and ``upper`` replace either bound for every pair.
    """
    if kind not in ("log-linear", "linear", "bbob"):
        raise ValueError(f"kind {kind!r}: log-linear, linear or bbob")
    if "suite" not in data.columns:
        data = data.with_columns(suite=pl.lit(None, pl.String))
    grouped = data.group_by(PAIR, maintain_order=True).agg(
        pl.col(fval_var), pl.col("suite").unique()
    )
    log = kind == "log-linear"
    targets = []
    for values, suites in zip(grouped[fval_var], grouped["suite"]):
        lo, hi = _suite_bounds(suites)
        lo = lo if lower is None else lower
        hi = hi if upper is None else upper
        targets.append(
            bbob_targets()
            if kind == "bbob"
            else seq_fv(values, n, log=log, lower=lo, upper=hi)
        )
    targets = grouped.select(PAIR).with_columns(target=pl.Series(targets))
    return targets.explode("target").sort(PAIR, maintain_order=True)


def get_ecdf_budgets(
    data: pl.DataFrame,
    n: int = 10,
    *,
    log: bool = True,
    eval_var: str = "evaluations",
) -> pl.DataFrame:
    """Budgets per (function, dimension) pair for the fixed-budget ECDF (EC-9):
    ``n`` from the pair's smallest to its largest logged evaluation, log-spaced
    or linear, rounded as ``seq_rt`` (TG-5). Columns ``function_id``,
    ``dimension`` and ``budget``."""
    grouped = data.group_by(PAIR, maintain_order=True).agg(pl.col(eval_var))
    budgets = [seq_rt(e, n, log=log) for e in grouped[eval_var]]
    budgets = grouped.drop(eval_var).with_columns(budget=pl.Series(budgets))
    return budgets.explode("budget").sort(PAIR, maintain_order=True)


def get_target_dt(  # pylint: disable=too-many-arguments
    data: pl.DataFrame,
    perspective: str,
    *,
    maximization: bool = False,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    by: Sequence[str] = ("algorithm_name",),
) -> pl.DataFrame:
    """One default target per (function, dimension) pair (TG-4, TG-6).

    ``perspective="fixed_target"``: per group of ``by``, the 2 % quantile (type 7)
    of the runs' final best values, 98 % when maximising; then the best of these.
    ``"fixed_budget"``: the largest logged evaluation of the pair.
    """
    if perspective == "fixed_budget":
        target = pl.col(eval_var).max().cast(pl.Float64)
        return data.group_by(PAIR).agg(target.alias("target")).sort(PAIR)
    if perspective != "fixed_target":
        raise ValueError(f"perspective {perspective!r}: fixed_target or fixed_budget")
    best = pl.max if maximization else pl.min
    quantile = pl.col("final").quantile(0.98 if maximization else 0.02, "linear")
    return (
        data.group_by([*PAIR, *by, "run_id"])
        .agg(best(fval_var).alias("final"))
        .group_by([*PAIR, *by])
        .agg(quantile.alias("target"))
        .group_by(PAIR)
        .agg(best("target"))
        .sort(PAIR)
    )
