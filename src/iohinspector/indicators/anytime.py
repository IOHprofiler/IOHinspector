from typing import Iterable

import numpy as np
import polars as pl

from moocore import (
    hypervolume,
    igd_plus,
    epsilon_additive,
    epsilon_mult,
    filter_dominated,
)


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


def _tchebycheff(weight_vec, ideal_point, point):
    tch_val = -np.inf
    for w, z, f in zip(weight_vec, ideal_point, point):
        tch_val = max(tch_val, w * abs(f - z))
    return tch_val


def _r2(weight_vec_set, ideal_point, point_set):
    sum_tch = 0
    for w in weight_vec_set:
        min_tch = np.inf
        for p in point_set:
            tch = _tchebycheff(w, ideal_point, p)
            min_tch = min(min_tch, tch)
        sum_tch += min_tch
    return sum_tch / len(weight_vec_set)


class NonDominated:
    def __call__(self, group: pl.DataFrame, objective_columns: Iterable):
        objectives = np.array(group[objective_columns])
        is_efficient = np.ones(objectives.shape[0], dtype=bool)
        for i, c in enumerate(objectives[1:]):
            if is_efficient[i + 1]:
                is_efficient[i + 1 :][is_efficient[i + 1 :]] = np.any(
                    objectives[i + 1 :][is_efficient[i + 1 :]] < c, axis=1
                )
                is_efficient[i + 1] = True
        group = group.with_columns(pl.Series(name="nondominated", values=is_efficient))
        return group


class HyperVolume:
    def __init__(self, reference_point: np.ndarray):
        """Function to calculate the Hypervolume metric over time. Used as an input to the 'add_indicator' function.

        Args:
            reference_set (np.ndarray): The reference point for the HV calculation.
        """
        self.reference_point = reference_point

    @property
    def var_name(self):
        return "HyperVolume"

    @property
    def minimize(self):
        return False

    def __call__(
        self, group: pl.DataFrame, objective_columns: Iterable, evals: Iterable[int]
    ) -> pl.DataFrame:
        """
        Args:
            group (pl.DataFrame): The DataFrame on which the indicator will be added (should be 1 optimization run only)
            objective_columns (Iterable): Which columns are the objectives
            evals (Iterable[int]): At which evaluations the operation should be performed.
            Note that using more evaluations will make the code slower.

        Returns:
            pl.DataFrame: a new DataFrame with columns of 'evals' and corresponding IGD+
        """
        obj_vals = np.clip(
            np.array(group[objective_columns]), None, self.reference_point
        )
        evals_dt = group["evaluations"]
        hvs = [
            hypervolume(obj_vals[: (evals_dt <= eval).sum()], ref=self.reference_point)
            for eval in evals
        ]
        return (
            pl.DataFrame(
                [
                    pl.Series(name="evaluations", values=evals, dtype=pl.UInt64),
                    pl.Series(name="HyperVolume", values=hvs),
                ]
            )
            .join_asof(group.sort("evaluations"), on="evaluations", strategy="backward")
            .fill_null(np.inf)
            .drop(objective_columns)
        )


class Epsilon:
    def __init__(self, reference_point: np.ndarray, version: str = "additive"):
        """Function to calculate the Epsilon indicator over time. Used as an input to the 'add_indicator' function.

        Args:
            reference_set (np.ndarray): The reference point for the HV calculation.
            version (str): Whether to use the additive or multiplicative version of this indicator.
        """
        self.reference_point = reference_point
        if version == "additive":
            self.indicator = epsilon_additive
            self._var_name = "Epsilon_Additive"
        else:
            self.indicator = epsilon_mult
            self._var_name = "Epsilon_Mult"

    @property
    def var_name(self):
        return self._var_name

    @property
    def minimize(self):
        return True

    def __call__(
        self, group: pl.DataFrame, objective_columns: Iterable, evals: Iterable[int]
    ) -> pl.DataFrame:
        """
        Args:
            group (pl.DataFrame): The DataFrame on which the indicator will be added (should be 1 optimization run only)
            objective_columns (Iterable): Which columns are the objectives
            evals (Iterable[int]): At which evaluations the operation should be performed.
            Note that using more evaluations will make the code slower.

        Returns:
            pl.DataFrame: a new DataFrame with columns of 'evals' and corresponding IGD+
        """
        obj_vals = np.clip(
            np.array(group[objective_columns]), None, self.reference_point
        )
        evals_dt = group["evaluations"]
        hvs = [
            self.indicator(
                filter_dominated(obj_vals[: (evals_dt <= eval).sum()]),
                ref=self.reference_point,
            )
            for eval in evals
        ]
        return (
            pl.DataFrame(
                [
                    pl.Series(name="evaluations", values=evals, dtype=pl.UInt64),
                    pl.Series(name=self._var_name, values=hvs),
                ]
            )
            .join_asof(group.sort("evaluations"), on="evaluations", strategy="backward")
            .fill_null(np.inf)
            .drop(objective_columns)
        )


class IGDPlus:
    def __init__(self, reference_set: np.ndarray):
        """Function to calculate the IGD+ metric over time. Used as an input to the 'add_indicator' function.

        Args:
            reference_set (np.ndarray): The reference set for the IGD+ calculation. Note that larger sets make
            the calculation slower
        """
        self.reference_set = reference_set

    @property
    def minimize(self):
        return True

    @property
    def var_name(self):
        return "IGD+"

    def __call__(
        self, group: pl.DataFrame, objective_columns: Iterable, evals: Iterable[int]
    ) -> pl.DataFrame:
        """

        Args:
            group (pl.DataFrame): The DataFrame on which the indicator will be added (should be 1 optimization run only)
            objective_columns (Iterable): Which columns are the objectives
            evals (Iterable[int]): At which evaluations the operation should be performed.
            Note that using more evaluations will make the code slower.

        Returns:
            pl.DataFrame: a new DataFrame with columns of 'evals' and corresponding IGD+
        """
        obj_vals = np.array(group[objective_columns])
        evals_dt = group["evaluations"]
        igds = [
            igd_plus(
                filter_dominated(obj_vals[: (evals_dt <= eval).sum()]),
                ref=self.reference_set,
            )
            for eval in evals
        ]
        return (
            pl.DataFrame(
                [
                    pl.Series(name="evaluations", values=evals, dtype=pl.UInt64),
                    pl.Series(name="IGD+", values=igds),
                ]
            )
            .join_asof(group.sort("evaluations"), on="evaluations", strategy="backward")
            .fill_null(np.inf)
            .drop(objective_columns)
        )

try:
    from pymoo.util.ref_dirs import get_reference_directions
    class R2:
        def __init__(self, n_ref_dirs: int, ideal_point: np.ndarray):
            """Function to calculate the R2 indicator over time. Used as an input to the 'add_indicator' function.

            Args:
                n_ref_dirs (int): How many reference directions to use. Reference directions are generated based on pymoo's 'energy' method.
                ideal_point (np.ndarray): The ideal point for the R2 calculations
            """
            self.ref_dirs = get_reference_directions("energy", len(ideal_point), n_ref_dirs)
            self.ideal_point = ideal_point

        @property
        def var_name(self):
            return "R2"

        @property
        def minimize(self):
            return True

        def __call__(
            self, group: pl.DataFrame, objective_columns: Iterable, evals: Iterable[int]
        ) -> pl.DataFrame:
            """

            Args:
                group (pl.DataFrame): The DataFrame on which the indicator will be added (should be 1 optimization run only)
                objective_columns (Iterable): Which columns are the objectives
                evals (Iterable[int]): At which evaluations the operation should be performed.
                Note that using more evaluations will make the code slower.

            Returns:
                pl.DataFrame: a new DataFrame with columns of 'evals' and corresponding IGD+
            """
            obj_vals = np.array(group[objective_columns])
            evals_dt = group["evaluations"]
            igds = [
                _r2(
                    self.ref_dirs,
                    self.ideal_point,
                    filter_dominated(obj_vals[: (evals_dt <= eval).sum()]),
                )
                for eval in evals
            ]
            return (
                pl.DataFrame(
                    [
                        pl.Series(name="evaluations", values=evals, dtype=pl.UInt64),
                        pl.Series(name=self.var_name, values=igds),
                    ]
                )
                .join_asof(group.sort("evaluations"), on="evaluations", strategy="backward")
                .fill_null(np.inf)
                .drop(objective_columns)
            )

except ImportError:
    class R2:
        def __init__(self, *args, **kwargs):
            import warnings
            warnings.warn("R2 indicator is N/A without pymoo installed")