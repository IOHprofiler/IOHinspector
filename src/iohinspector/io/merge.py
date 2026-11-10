"""The identical-row rule: merging files that log the same runs.

Some loggers write one run to several files, each with its own trigger (COCO's
``.dat``, ``.tdat`` and ``.rdat``). Format v1 keeps every row of every file, and
merges two rows only when they describe the same evaluation with identical
values in every column (``docs/format.md``, plan principle 1). Within one file
nothing is merged: rows a file repeats stay.

Two columns describe the merge:

- ``logged_in``: the kinds of the files that hold the row, joined by ``+`` in the
  given kind order, for example ``dat+tdat``;
- ``merge_conflict``: true for every row of an evaluation that the files log
  with different values (more than one distinct row for the evaluation, from
  more than one file). Rows a file repeats are not conflicts.

Rows are ordered by evaluation count, then by kind, then by position in the
file. When a file's evaluation counts decrease within a run, that order would
change the file's own order, so the files' rows follow each other per run
instead, nothing is merged, and ``Merged.unsorted`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

#: Helper columns of the input frames; every other column is a value column.
KEYS = ("run_index", "file_row")


@dataclass
class Merged:
    """The merged rows, with ``run_index``, ``row``, the value columns,
    ``logged_in`` and ``merge_conflict``."""

    trace: pl.DataFrame
    conflicts: int = 0
    unsorted: bool = False


def _with_rows(trace: pl.DataFrame) -> pl.DataFrame:
    return trace.with_columns(
        row=pl.int_range(pl.len(), dtype=pl.Int64).over("run_index")
    ).select("run_index", "row", pl.exclude("run_index", "row"))


def _labels(kinds: list[str]) -> dict[int, str]:
    """``logged_in`` text of every bit mask over ``kinds``."""
    return {
        mask: "+".join(k for i, k in enumerate(kinds) if mask >> i & 1)
        for mask in range(1, 1 << len(kinds))
    }


def _is_sorted(stacked: pl.DataFrame) -> bool:
    if stacked.is_empty():
        return True
    steps = (
        stacked.sort("_kind", "file_row")
        .group_by("_kind", "run_index")
        .agg(ok=(pl.col("evaluations").diff().fill_null(0) >= 0).all())
    )
    return bool(steps["ok"].all())


def merge_files(files: list[tuple[str, pl.DataFrame]]) -> Merged:
    """Merge frames of the same runs, given as ``(kind, frame)`` in kind order.

    Each frame has ``run_index``, ``file_row`` (the row's position in its file),
    ``evaluations`` and further value columns. A single frame comes back with
    ``row`` numbers and without the two merge columns.
    """
    if len(files) == 1:
        return Merged(_with_rows(files[0][1].sort("file_row").drop("file_row")))
    kinds = [kind for kind, _ in files]
    stacked = pl.concat(
        [
            frame.with_columns(_kind=pl.lit(i, dtype=pl.Int64))
            for i, (_, frame) in enumerate(files)
        ],
        how="diagonal",
    )
    values = [c for c in stacked.columns if c not in (*KEYS, "_kind")]
    if not _is_sorted(stacked):
        trace = stacked.sort("run_index", "_kind", "file_row").with_columns(
            logged_in=pl.col("_kind").replace_strict(
                dict(enumerate(kinds)), return_dtype=pl.String
            ),
            merge_conflict=pl.lit(False),
        )
        return Merged(_with_rows(trace.drop("_kind", "file_row")), unsorted=True)
    return _merge_sorted(stacked, values, kinds)


def _merge_sorted(stacked: pl.DataFrame, values: list[str], kinds: list[str]) -> Merged:
    keys = ["run_index", *values]
    stacked = stacked.with_columns(
        _occ=pl.int_range(pl.len()).over([*keys, "_kind"]),
        _bit=pl.lit(2, dtype=pl.Int64).pow(pl.col("_kind")),
    )
    merged = (
        stacked.group_by([*keys, "_occ"])
        .agg(
            _mask=pl.col("_bit").sum(),
            _first=pl.col("_kind").min(),
            _pos=pl.col("file_row").sort_by("_kind").first(),
        )
        .sort("run_index", "evaluations", "_first", "_pos")
    )
    same_eval = ["run_index", "evaluations"]
    union = pl.sum_horizontal(
        ((pl.col("_mask") // (1 << i)) % 2).max().over(same_eval) * (1 << i)
        for i in range(len(kinds))
    )
    distinct = (pl.col("_occ") == 0).sum().over(same_eval)
    merged = merged.with_columns(_union=union).with_columns(
        logged_in=pl.col("_mask").replace_strict(
            _labels(kinds), return_dtype=pl.String
        ),
        merge_conflict=(distinct > 1)
        & ((pl.col("_union") & (pl.col("_union") - 1)) != 0),
    )
    conflicts = int(merged["merge_conflict"].sum())
    trace = merged.select("run_index", *values, "logged_in", "merge_conflict")
    return Merged(_with_rows(trace), conflicts=conflicts)
