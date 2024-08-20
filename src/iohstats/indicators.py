from moocore import hypervolume
from scipy.spatial.distance import cdist

import polars as pl
import numpy as np

from typing import Iterable, Callable

from functools import partial

class Anytime_IGD:
    def __init__(self, reference_set: np.ndarray):
        self.reference_set = reference_set

    def __call__(
        self, group: pl.DataFrame, objective_columns: Iterable
    ) -> pl.DataFrame:
        points = np.array(group[objective_columns])
        group = group.with_columns(
            pl.Series(
                name="igd",
                values=np.mean(
                    np.minimum.accumulate(cdist(self.reference_set, points), axis=1),
                    axis=0,
                ),
            )
        )
        return group


class Anytime_IGDPlus:
    def __init__(self, reference_set: np.ndarray):
        self.reference_set = reference_set

    def __call__(
        self, group: pl.DataFrame, objective_columns: Iterable
    ) -> pl.DataFrame:
        points = np.array(group[objective_columns])
        group = group.with_columns(
            pl.Series(
                name="igd+",
                values=np.mean(
                    np.minimum.accumulate(
                        cdist(
                            self.reference_set,
                            points,
                            metric=lambda x, y: np.sqrt(
                                np.clip(y - x, 0, None) ** 2
                            ).sum(),
                        ),
                        axis=1,
                    ),
                    axis=0,
                ),
            )
        )
        return group


class Anytime_HyperVolume:
    def __init__(self, reference_point: np.ndarray):
        self.reference_point = reference_point

    def __call__(
        self, group: pl.DataFrame, objective_columns: Iterable
    ) -> pl.DataFrame:
        # clip is here to avoid negative values; note that this assumes minimization for all objectives
        obj_vals = np.clip(
            np.array(group[objective_columns]), None, self.reference_point
        )
        if len(objective_columns) == 2:
            hvs = self._incremental_hv(obj_vals)
        else:
            hvs = [
                hypervolume(obj_vals[:i], ref=self.reference_point)
                for i in range(1, len(group) + 1)
            ]
        group = group.with_columns(pl.Series(name="hv", values=hvs))
        return group

    def _incremental_hv(self, points):
        sorted_array = [
            [-np.inf, self.reference_point[1]],
            [self.reference_point[0], -np.inf],
        ]
        current_hypervolume = 0.0
        all_hv = []
        for point in np.array(points):
            dominated_idxs = [
                i
                for i, p in enumerate(sorted_array)
                if point[0] <= p[0]
                and point[1] <= p[1]
                and (point[0] < p[0] or point[1] < p[1])
            ]
            point = tuple(point)
            if len(dominated_idxs) > 0:
                index = min(dominated_idxs)
                for _ in dominated_idxs:
                    dom_point = sorted_array[index]
                    left_neighbor = sorted_array[index - 1]
                    right_neighbor = sorted_array[index + 1]
                    current_hypervolume -= (left_neighbor[1] - dom_point[1]) * (
                        right_neighbor[0] - dom_point[0]
                    )
                    sorted_array = [p for p in sorted_array if p is not dom_point]

            index = next(
                (
                    i
                    for i, p in enumerate(sorted_array)
                    if p[0] > point[0] or (p[0] == point[0] and p[1] < point[1])
                ),
                len(sorted_array),
            )
            sorted_array.insert(index, point)
            left_neighbor = sorted_array[index - 1]
            right_neighbor = sorted_array[index + 1]
            current_hypervolume += (left_neighbor[1] - point[1]) * (
                right_neighbor[0] - point[0]
            )
            all_hv.append(current_hypervolume)
        return all_hv


class Anytime_NonDominated:
    def __call__(self, group: pl.DataFrame, objective_columns: Iterable):
        objectives = np.array(group[objective_columns])
        is_efficient = np.ones(objectives.shape[0], dtype=bool)
        for i, c in enumerate(objectives[1:]):
            if is_efficient[i + 1]:
                is_efficient[i + 1 :][is_efficient[i + 1 :]] = np.any(
                    objectives[i + 1 :][is_efficient[i + 1 :]] < c, axis=1
                )  # Keep any later point with a lower cost
                is_efficient[i + 1] = True  # And keep self
        group = group.with_columns(pl.Series(name="nondominated", values=is_efficient))
        return group


class Final_NonDominated:
    def __call__(self, group: pl.DataFrame, objective_columns: Iterable):
        objectives = np.array(group[objective_columns])
        is_efficient = np.ones(objectives.shape[0], dtype=bool)
        for i, c in enumerate(objectives):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(
                    objectives[is_efficient] < c, axis=1
                )  # Keep any later point with a lower cost
                is_efficient[i] = True  # And keep self
        group = group.with_columns(
            pl.Series(name="final_nondominated", values=is_efficient)
        )
        return group


def add_indicator(df: pl.DataFrame, indicator: object, objective_columns: Iterable):
    indicator_callable = partial(indicator, objective_columns=objective_columns)
    return df.group_by("data_id").map_groups(indicator_callable)
