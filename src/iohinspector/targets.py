"""Target and budget grids (``docs/definitions/targets.md``, TG-*).

Ports of IOHanalyzer's ``seq_FV``, ``seq_RT``, ``get_ECDF_targets`` and
``get_target_dt``, plus the COCO ``bbob`` grid.
"""

import warnings
from collections.abc import Iterable, Sequence

import numpy as np
import polars as pl

__all__ = ["bbob_targets", "get_ecdf_targets", "get_target_dt", "seq_fv", "seq_rt"]

PAIR = ("function_id", "dimension")
#: Lower end of log-spaced targets when the data has no positive value (TG-3).
LOG_FLOOR = 1e-12


def spaced(lower: float, upper: float, n: int, log: bool = False) -> np.ndarray:
    """``n`` values evenly spaced from ``lower`` to ``upper`` (in log10 when ``log``),
    both ends included, sorted, without duplicates."""
    if log:
        if lower <= 0 or upper <= 0:
            raise ValueError(f"log spacing needs positive bounds, got {lower}, {upper}")
        lower, upper = np.log10(lower), np.log10(upper)
    values = np.unique(np.linspace(lower, upper, n))
    return 10.0**values if log else values


def seq_fv(
    values: Iterable[float],
    n: int = 10,
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
    """The 51 COCO targets 10^(2 - k/5), k = 0..50, from 1e2 down to 1e-8."""
    return 10.0 ** (2 - np.arange(51) / 5)


def get_ecdf_targets(
    data: pl.DataFrame,
    kind: str = "log-linear",
    n: int = 10,
    fval_var: str = "raw_y",
    pair: Sequence[str] = PAIR,
) -> pl.DataFrame:
    """Targets per (function, dimension) pair (TG-1, TG-2): ``kind``
    ``"log-linear"`` or ``"linear"`` (``n`` targets over the pair's observed
    values), or ``"bbob"`` (the 51 COCO targets)."""
    pair = list(pair)

    def targets(values: list[float]) -> list[float]:
        if kind == "bbob":
            return bbob_targets().tolist()
        if kind not in ("log-linear", "linear"):
            raise ValueError(f"kind {kind!r}: log-linear, linear or bbob")
        return seq_fv(values, n, kind == "log-linear").tolist()

    grouped = data.group_by(pair, maintain_order=True).agg(pl.col(fval_var))
    return pl.DataFrame(
        [(*key, t) for *key, values in grouped.iter_rows() for t in targets(values)],
        schema=[*pair, "target"],
        orient="row",
    ).sort(pair, maintain_order=True)


def get_target_dt(  # pylint: disable=too-many-arguments
    data: pl.DataFrame,
    perspective: str,
    maximization: bool = False,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    by: Sequence[str] = ("algorithm_name",),
    pair: Sequence[str] = PAIR,
) -> pl.DataFrame:
    """One default target per (function, dimension) pair (TG-4, TG-6).

    ``perspective="fixed_target"``: per group of ``by``, the 2 % quantile (type 7)
    of the runs' final best values, 98 % when maximising; then the best of these.
    ``"fixed_budget"``: the largest logged evaluation of the pair.
    """
    pair = list(pair)
    if perspective == "fixed_budget":
        target = pl.col(eval_var).max().cast(pl.Float64)
        return data.group_by(pair).agg(target.alias("target")).sort(pair)
    if perspective != "fixed_target":
        raise ValueError(f"perspective {perspective!r}: fixed_target or fixed_budget")
    y = pl.col(fval_var)
    final = y.max() if maximization else y.min()
    quantile = pl.col("final").quantile(0.98 if maximization else 0.02, "linear")
    best = pl.col("target").max() if maximization else pl.col("target").min()
    return (
        data.group_by([*pair, *by, "run_id"])
        .agg(final.alias("final"))
        .group_by([*pair, *by])
        .agg(quantile.alias("target"))
        .group_by(pair)
        .agg(best)
        .sort(pair)
    )
