from collections.abc import Iterable

import numpy as np
import polars as pl
from moocore import (
    epsilon_additive,
    epsilon_mult,
    filter_dominated,
    hypervolume,
    igd_plus,
)

from .._upgrade import removed_names


def get_reference_set(
    data: pl.DataFrame, obj_cols: Iterable[str], max_size: int = 1000
) -> np.ndarray:
    """Get a (subsampled) reference set from a set of data

    Args:
        data (pl.DataFrame): The dataframe from which to extract the objective values
        obj_cols (Iterable[str]): The names of the objectives in 'data'
        max_size (int, optional): The maximum number of points in the reference set. Defaults to 1000.

    Returns:
        np.ndarray: The filtered reference set
    """
    obj_vals = np.array(data[obj_cols])
    ref_set = filter_dominated(obj_vals)
    if len(ref_set) < max_size:
        return ref_set
    return ref_set[np.random.choice(len(ref_set), size=max_size)]


def _r2(weights: np.ndarray, ideal: np.ndarray, points: np.ndarray) -> float:
    """The R2 indicator: the mean over the weight vectors of the smallest weighted
    Tchebycheff distance of a point to the ideal point (inf without points)."""
    if len(points) == 0:
        return np.inf
    distance = np.abs(np.asarray(points) - ideal)[None, :, :] * weights[:, None, :]
    total = 0.0
    for smallest in distance.max(axis=2).min(axis=1):  # summed in order
        total += smallest
    return total / len(weights)


class _Anytime:
    """An indicator over time, an input to ``add_indicator``: called with one run
    (``group``), its objective columns and the evaluations ``evals``, it adds a
    column ``var_name`` with the indicator of the points logged up to each of
    ``evals`` (inf before the first) and drops the objective columns.

    Subclasses give ``_value(points)``; ``clip`` clips the points to
    ``reference_point`` first, ``nondominated`` keeps only their nondominated
    ones."""

    var_name = ""
    minimize = True
    clip = False
    nondominated = True

    def _value(self, points: np.ndarray) -> float:
        raise NotImplementedError

    def __call__(
        self, group: pl.DataFrame, obj_vars: Iterable, evals: Iterable[int]
    ) -> pl.DataFrame:
        """
        Args:
            group (pl.DataFrame): The DataFrame on which the indicator will be added (should be 1 optimization run only)
            obj_vars (Iterable): Which columns are the objectives
            evals (Iterable[int]): At which evaluations the operation should be performed.
            Note that using more evaluations will make the code slower.

        Returns:
            pl.DataFrame: a new DataFrame with columns of 'evals' and the indicator
        """
        points = np.array(group[obj_vars])
        if self.clip:
            points = np.clip(points, None, self.reference_point)
        logged = np.sort(group["evaluations"].to_numpy())
        counts = np.searchsorted(logged, np.asarray(evals), "right")
        values = [
            self._value(
                filter_dominated(points[:n]) if self.nondominated else points[:n]
            )
            for n in counts.tolist()
        ]
        return (
            pl.DataFrame(
                [
                    pl.Series(name="evaluations", values=evals, dtype=pl.UInt64),
                    pl.Series(name=self.var_name, values=values),
                ]
            )
            .join_asof(group.sort("evaluations"), on="evaluations", strategy="backward")
            .fill_null(np.inf)
            .drop(obj_vars)
        )


class HyperVolume(_Anytime):
    """The hypervolume over time, of the points clipped to the reference point."""

    var_name = "HyperVolume"
    minimize = False
    clip = True
    nondominated = False

    def __init__(self, reference_point: np.ndarray):
        """
        Args:
            reference_point (np.ndarray): The reference point for the HV calculation.
        """
        self.reference_point = reference_point

    def _value(self, points: np.ndarray) -> float:
        return hypervolume(points, ref=self.reference_point)


class Epsilon(_Anytime):
    """The additive or multiplicative epsilon indicator over time."""

    clip = True

    def __init__(self, reference_point: np.ndarray, version: str = "additive"):
        """
        Args:
            reference_point (np.ndarray): The reference point (set) of the indicator.
            version (str): Whether to use the additive or multiplicative version of this indicator.
        """
        self.reference_point = reference_point
        if version == "additive":
            self.indicator, self.var_name = epsilon_additive, "Epsilon_Additive"
        else:
            self.indicator, self.var_name = epsilon_mult, "Epsilon_Mult"

    def _value(self, points: np.ndarray) -> float:
        return self.indicator(points, ref=self.reference_point)


class IGDPlus(_Anytime):
    """The IGD+ indicator over time."""

    var_name = "IGD+"

    def __init__(self, reference_set: np.ndarray):
        """
        Args:
            reference_set (np.ndarray): The reference set for the IGD+ calculation.
                Note that larger sets make the calculation slower
        """
        self.reference_set = reference_set

    def _value(self, points: np.ndarray) -> float:
        return igd_plus(points, ref=self.reference_set)


try:
    from pymoo.util.ref_dirs import get_reference_directions

    class R2(_Anytime):
        """The R2 indicator over time."""

        var_name = "R2"

        def __init__(self, n_ref_dirs: int, ideal_point: np.ndarray):
            """
            Args:
                n_ref_dirs (int): How many reference directions to use. Reference directions are generated based on pymoo's 'energy' method.
                ideal_point (np.ndarray): The ideal point for the R2 calculations
            """
            self.ref_dirs = get_reference_directions(
                "energy", len(ideal_point), n_ref_dirs
            )
            self.ideal_point = ideal_point

        def _value(self, points: np.ndarray) -> float:
            return _r2(self.ref_dirs, self.ideal_point, points)

except ImportError:

    class R2:
        def __init__(self, *args, **kwargs):
            import warnings

            warnings.warn("R2 indicator is N/A without pymoo installed")


__getattr__ = removed_names(__name__)
