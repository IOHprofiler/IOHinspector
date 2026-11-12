import warnings
from collections.abc import Sequence

import polars as pl

from .._upgrade import upgrade_hints
from ..targets import PAIR
from .ecdf import get_ecdf_auc


@upgrade_hints
def get_aocc(
    data: pl.DataFrame,
    eval_max: int,
    fval_var: str = "eaf",
    by: Sequence[str] = ("function_name", "algorithm_name"),
    scale_eval_log: bool = False,
) -> pl.DataFrame:
    """Deprecated in 0.2.0: use ``get_ecdf_auc(data, continuous=True)`` on the raw
    function values, which normalises them per (function, dimension) pair (EC-5).

    The mean over the runs of each group of ``by`` of the normalised area under
    ``fval_var``, a quality in [0, 1] such as the ``eaf`` of ``transform_fval``,
    over the evaluations [1, ``eval_max``] (log10 with ``scale_eval_log``), in the
    column ``AOCC``. Since 0.2.0 this is ``get_ecdf_auc``, so unlike 0.0.8 the area
    starts at evaluation 1 (not 0) and is divided by ``eval_max - 1``; it is of the
    best-so-far quality, clipped to [0, 1]; a run has 0 before its first logged
    evaluation (not its smallest value); and an empty range raises ``ValueError``.
    """
    warnings.warn(
        "get_aocc is deprecated since 0.2.0; use get_ecdf_auc(data, continuous=True)",
        DeprecationWarning,
        stacklevel=3,
    )
    # one pair unless by names it: the mean over runs, as before
    data = data.with_columns(pl.lit(0).alias(c) for c in PAIR if c not in by)
    bounds = pl.DataFrame({"target": [0.0, 1.0]})
    auc = get_ecdf_auc(
        data,
        data.select(PAIR).unique().join(bounds, how="cross"),
        by,
        b_hi=eval_max,
        scale_eval_log=scale_eval_log,
        continuous=True,
        scale_f_log=False,
        maximization=True,
        fval_var=fval_var,
    )
    return auc.rename({"auc": "AOCC"})
