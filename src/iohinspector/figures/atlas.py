"""The Atlas: one small panel per (function, dimension) pair, as a grid."""

import dataclasses

import polars as pl

from .spec import FigureSpec, Layout


def atlas(spec: FigureSpec, runs: pl.DataFrame) -> FigureSpec:
    """``spec`` with its first panel drawn once for each (function, dimension)
    pair of ``runs`` (those its ``data.where`` keeps): a row per function and a
    column per dimension, shared axes and one legend below. Render it with
    ``runs``, or give ``panel_frames`` its result."""
    base = spec.panels[0]
    for column, values in base.data.where.items():
        runs = runs.filter(pl.col(column).is_in(values))
    keys = (
        runs.select("function_id", "dimension")
        .unique()
        .sort("function_id", "dimension")
    )
    panels = [
        dataclasses.replace(
            base,
            data=dataclasses.replace(
                base.data,
                where={**base.data.where, "function_id": [f], "dimension": [d]},
            ),
            title=f"f{f} d{d}",
        )
        for f, d in keys.iter_rows()
    ]
    return dataclasses.replace(
        spec,
        panels=panels,
        layout=Layout(columns=keys["dimension"].n_unique(), share_x=True, share_y=True),
        legend=dataclasses.replace(spec.legend, mode="box", shared=True),
    )
