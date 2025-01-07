from typing import Iterable

import polars as pl
import numpy as np
from moocore import is_nondominated


class NonDominated:
    def __call__(self, group: pl.DataFrame, objective_columns: Iterable):
        objectives = np.array(group[objective_columns])
        return group.with_columns(
            pl.Series(name="final_nondominated", values=is_nondominated(objectives))
        )
