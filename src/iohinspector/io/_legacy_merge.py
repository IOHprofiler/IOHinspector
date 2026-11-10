"""Merging the data files of one legacy run set: ``.dat``, ``.tdat``, ``.cdat`` etc.

The legacy logger writes the same run to up to four files, each with its own
trigger. A row that two files log for the same evaluation is stored once when
every column is identical (the identical-row rule of ``docs/format.md``);
otherwise both rows are kept and flagged. Within one file nothing is merged:
repeated rows stay.

Two columns describe the merge when a run set has more than one file:

- ``logged_in``: the file kinds that hold the row, joined by ``+`` in the order
  ``dat``, ``tdat``, ``cdat``, ``idat`` (for example ``dat+cdat``);
- ``merge_conflict``: true for every row of an evaluation that the files log
  with different values (more than one distinct row for the evaluation, from
  more than one file). Rows a file repeats are not conflicts.

Rows are ordered by evaluation count, then by file kind, then by position in the
file. When a file's evaluation counts decrease within a run, that order would
change the file's own order, so nothing is merged for the run set: the files'
rows follow each other per run, and an ``unsorted_evaluations`` warning is added.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ._legacy_dat import DataFile

#: File kinds in their merge order.
MERGE_ORDER = ("dat", "tdat", "cdat", "idat")
_HELPERS = ("run_index", "file_row", "line", "_kind")


@dataclass
class Merged:
    """The merged rows of a run set, with ``run_index`` and ``row``."""

    trace: pl.DataFrame
    conflicts: int = 0
    unsorted: bool = False


def _value_columns(frames: list[pl.DataFrame]) -> list[str]:
    out: list[str] = []
    for frame in frames:
        out += [c for c in frame.columns if c not in _HELPERS and c not in out]
    return out


def _with_rows(trace: pl.DataFrame) -> pl.DataFrame:
    return trace.with_columns(
        row=pl.int_range(pl.len(), dtype=pl.Int64).over("run_index")
    )


def _is_sorted(stacked: pl.DataFrame) -> bool:
    if stacked.is_empty():
        return True
    steps = (
        stacked.sort("_kind", "file_row")
        .group_by("_kind", "run_index")
        .agg(ok=(pl.col("evaluations").diff().fill_null(0) >= 0).all())
    )
    return bool(steps["ok"].all())


def _labels(kinds: list[str]) -> dict[int, str]:
    """``logged_in`` text of every bit mask over ``kinds``."""
    out = {}
    for mask in range(1, 1 << len(kinds)):
        out[mask] = "+".join(k for i, k in enumerate(kinds) if mask >> i & 1)
    return out


def merge_files(files: list[DataFile]) -> Merged:
    """Merge the parsed files of one run set; they must have the same number of runs."""
    if len(files) == 1:
        trace = files[0].trace.drop("file_row", "line")
        return Merged(_with_rows(trace))
    files = sorted(files, key=lambda f: MERGE_ORDER.index(f.kind))
    kinds = [f.kind for f in files]
    stacked = pl.concat(
        [
            f.trace.drop("line").with_columns(_kind=pl.lit(i, dtype=pl.Int64))
            for i, f in enumerate(files)
        ],
        how="diagonal",
    )
    values = _value_columns([f.trace for f in files])
    if not _is_sorted(stacked):
        trace = stacked.sort("run_index", "_kind", "file_row").with_columns(
            logged_in=pl.col("_kind").replace_strict(dict(enumerate(kinds))),
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
