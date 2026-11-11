"""``DataManager``: the one way in for Python users.

A manager holds datasets of format v1 (``docs/format.md``) and shows them as one set
of the four tables (``runs``, ``descriptors``, ``trace``, ``positions``), as polars
LazyFrames. ``select`` narrows the runs; ``load`` builds the wide frame the metrics
and plots take. Runs are keyed by ``run_id``, a stable hash.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import polars as pl

from ._upgrade import removed_names, upgrade_hints
from .align import turbo_align
from .convert import read_dataset
from .format import (
    DESCRIPTORS_SCHEMA,
    RUNS_SCHEMA,
    TRACE_SCHEMA,
    StoredDataset,
    descriptor_columns,
    scan_dataset,
)
from .format.schema import ALGORITHM_DESCRIPTORS

_EMPTY = {"runs": RUNS_SCHEMA, "descriptors": DESCRIPTORS_SCHEMA, "trace": TRACE_SCHEMA}
_TABLES = ("runs", "descriptors", "trace", "positions")
_removed = removed_names("DataManager", AttributeError)


@dataclass
class DataManager:
    """Datasets read from sources or opened from converted folders."""

    datasets: list[StoredDataset] = field(default_factory=list, repr=False)

    def __getattr__(self, name: str) -> Any:
        return _removed(name)

    def add_folder(self, path: str | os.PathLike[str], **options: Any) -> None:
        """Add a source in any format: a file, a folder or an archive.

        ``options`` go to ``iohinspector.convert.convert``: ``format`` (default:
        detected), ``mapping`` for custom CSV, ``dataset_id`` (default: the name of
        ``path``), ``tags`` and ``tags_from_path``.
        """
        self._add(read_dataset(path, **options))

    def add_folders(self, paths: Iterable[str | os.PathLike[str]], **options: Any):
        """``add_folder`` for each path."""
        for path in paths:
            self.add_folder(path, **options)

    def add_dataset(self, path: str | os.PathLike[str]) -> None:
        """Add a converted dataset folder (``iohinspector convert``)."""
        self._add(scan_dataset(path))

    def _add(self, dataset: StoredDataset) -> None:
        if any(d.dataset_id == dataset.dataset_id for d in self.datasets):
            raise ValueError(
                f"a dataset with id {dataset.dataset_id!r} is already loaded; "
                "give another dataset_id"
            )
        self.datasets.append(dataset)

    def __add__(self, other: DataManager) -> DataManager:
        return DataManager(self.datasets + other.datasets)

    # ----------------------------------------------------------------- tables

    def _table(self, name: str) -> pl.LazyFrame:
        frames = [getattr(d, name) for d in self.datasets]
        frames = [f for f in frames if f is not None]
        if frames:
            return pl.concat(frames, how="diagonal_relaxed")
        return pl.LazyFrame(schema=_EMPTY[name])

    @property
    def runs(self) -> pl.LazyFrame:
        return self._table("runs")

    @property
    def descriptors(self) -> pl.LazyFrame:
        return self._table("descriptors")

    @property
    def trace(self) -> pl.LazyFrame:
        return self._table("trace")

    @property
    def positions(self) -> pl.LazyFrame | None:
        """The positions, or None when no dataset has them."""
        if all(d.positions is None for d in self.datasets):
            return None
        return self._table("positions")

    @property
    def overview(self) -> pl.DataFrame:
        """One row per run: the runs columns and every descriptor as a column."""
        keys = self.descriptor_keys
        runs = self.runs
        if keys:
            columns = descriptor_columns(self.descriptors, keys)
            runs = runs.join(columns, on="run_id", how="left", maintain_order="left")
        df = runs.collect()
        front = [c for c in ("run_id", *ALGORITHM_DESCRIPTORS) if c in df.columns]
        return df.select(*front, pl.exclude(front))

    # ----------------------------------------------------------------- selection

    @upgrade_hints
    def select(  # pylint: disable=too-many-arguments
        self,
        *,
        function_ids: Sequence[int] | None = None,
        algorithms: Sequence[str] | None = None,
        dimensions: Sequence[int] | None = None,
        instances: Sequence[int] | None = None,
        run_ids: Sequence[int] | None = None,
        **columns: Sequence[Any],
    ) -> DataManager:
        """The runs whose values are in the given lists, as a new manager.

        ``columns`` names any other runs column or descriptor, such as
        ``suite=["BBOB"]`` or ``budget=[1000]``.
        """
        named = zip(
            ("function_id", "algorithm_name", "dimension", "instance", "run_id"),
            (function_ids, algorithms, dimensions, instances, run_ids),
        )
        filters = {c: v for c, v in named if v is not None} | columns
        overview = self.overview
        unknown = sorted(set(filters) - set(overview.columns))
        if unknown:
            raise ValueError(f"no runs column or descriptor is named {unknown}")
        keep = overview.filter(
            pl.lit(True), *(pl.col(c).is_in(list(v)) for c, v in filters.items())
        )["run_id"]
        return DataManager([_only(d, keep) for d in self.datasets])

    def _distinct(self, column: str) -> tuple[Any, ...]:
        overview = self.overview
        if column not in overview.columns:
            return ()
        return tuple(overview[column].drop_nulls().unique().sort())

    @property
    def functions(self) -> tuple[int, ...]:
        return self._distinct("function_id")

    @property
    def algorithms(self) -> tuple[str, ...]:
        return self._distinct("algorithm_name")

    @property
    def dimensions(self) -> tuple[int, ...]:
        return self._distinct("dimension")

    @property
    def instances(self) -> tuple[int, ...]:
        return self._distinct("instance")

    @property
    def descriptor_keys(self) -> tuple[str, ...]:
        keys = self.descriptors.select(pl.col("key").unique()).collect()["key"]
        return tuple(keys.sort())

    @property
    def n_runs(self) -> int:
        return self.runs.select(pl.len()).collect().item()

    @property
    def any(self) -> bool:
        return self.n_runs > 0

    # ----------------------------------------------------------------- loading

    def load(
        self,
        monotonic: bool = True,
        include_meta_data: bool = False,
        include_columns: Sequence[str] | None = None,
        x_values: Iterable[int | float] | None = None,
    ) -> pl.DataFrame:
        """The trace of the selected runs in one frame, positions ``x0`` ... joined.

        ``monotonic`` makes ``raw_y`` the best-so-far value and keeps, per run, the
        rows where it changes and the final row. ``x_values`` aligns each run to
        those evaluations (``turbo_align``; implies ``monotonic``).
        ``include_meta_data`` puts every ``overview`` column in front,
        ``include_columns`` only the ones named.
        """
        if not self.any:
            return pl.DataFrame()
        lf = self.trace.drop("function_id", "dimension")
        positions = self.positions
        if positions is not None:
            positions = positions.drop("function_id", "dimension")
            lf = lf.join(
                positions, on=["run_id", "row"], how="left", maintain_order="left"
            )
        if monotonic or x_values is not None:
            lf = _best_so_far(lf, self.runs.select("run_id", "maximization"))
        lf = lf.drop("row").with_columns(pl.col("evaluations").cast(pl.UInt64))
        df = lf.collect()
        if x_values is not None:
            df = turbo_align(df, x_values)
        if include_meta_data or include_columns is not None:
            meta = self.overview
            if include_columns is not None:
                names = [c for c in include_columns if c != "run_id"]
                meta = meta.select("run_id", *names)
            df = meta.join(df, on="run_id", maintain_order="right")
        return df


def _only(dataset: StoredDataset, run_ids: pl.Series) -> StoredDataset:
    """``dataset`` with only the runs in ``run_ids``."""
    keep = pl.col("run_id").is_in(run_ids.implode())
    tables = {t: getattr(dataset, t) for t in _TABLES}
    return replace(
        dataset, **{t: lf.filter(keep) for t, lf in tables.items() if lf is not None}
    )


def _best_so_far(lf: pl.LazyFrame, maximization: pl.LazyFrame) -> pl.LazyFrame:
    """Per run, ``raw_y`` as the best-so-far value, on the rows where it changes
    and the final row."""
    y = pl.col("raw_y")
    best = (
        pl.when(pl.col("maximization"))
        .then(y.cum_max().over("run_id"))
        .otherwise(y.cum_min().over("run_id"))
    )
    lf = lf.join(maximization, on="run_id", how="left", maintain_order="left")
    lf = lf.with_columns(raw_y=best).drop("maximization")
    changed = (y != y.shift(1)).over("run_id").fill_null(True)
    last = (pl.int_range(pl.len()) == pl.len() - 1).over("run_id")
    return lf.filter(changed | last)


__all__ = ["DataManager"]
