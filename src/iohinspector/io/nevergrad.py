"""Reader for Nevergrad benchmark CSV files (fixed budget only).

Nevergrad's ``allxps`` files have one row per finished experiment, so one row is
one run with one trace row: the best loss ``loss`` at the end of the budget
``budget``. Mapping to the storage format:

- ``loss`` is ``raw_y`` and ``budget`` is the row's evaluations and the run's budget;
- ``optimizer_name`` is the algorithm name and ``dimension`` the dimension;
- the function is ``name`` (``function_class`` in older files), with ``_rescaled``
  added when ``rescale`` (``transform`` in older files) is true, as IOHanalyzer
  does. Functions get ids 1, 2, ... in the order of their sorted names, recorded
  in ``manifest.extra.function_ids``;
- every other column becomes a descriptor with the origin ``experiment`` and its
  own column name. Its type is that of the whole column (bool, int, float or
  string); an empty cell gives no descriptor;
- the objective is minimised, and the suite is ``NEVERGRAD``.

Rows of crashed runs have an empty ``loss`` (and an ``error`` column naming the
exception). They are skipped, as IOHanalyzer reads them as NA: the count per
algorithm is in ``manifest.extra.skipped_rows`` and a warning
``empty_value_rows`` is recorded. ``data_rows`` and the run indexes count only
the rows kept. A ``loss`` that is not empty and not a number is still an error.

Gzip-compressed files (``.csv.gz``, as published) are read directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._csv import CsvMapping, MappingError, Source, read_csv, read_header
from .base import ReadResult

READER_NAME = "nevergrad-csv"
REQUIRED = ("loss", "budget", "dimension", "optimizer_name")


def is_nevergrad_csv(path: str | Path) -> bool:
    """True if the file's header looks like a Nevergrad result file."""
    try:
        header = read_header(Path(path))
    except (OSError, ValueError):
        return False
    return _problem(header) is None


def _problem(header: list[str]) -> str | None:
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        return f"missing column(s) {missing}"
    if "name" not in header and "function_class" not in header:
        return "neither column 'name' nor 'function_class' exists"
    return None


def nevergrad_mapping(header: list[str]) -> CsvMapping:
    """The column mapping for a Nevergrad file with these columns."""
    problem = _problem(header)
    if problem:
        raise MappingError(f"not a Nevergrad result file: {problem}")
    name_col = "name" if "name" in header else "function_class"
    scale_col = next((c for c in ("rescale", "transform") if c in header), None)

    def function_name(get: Any) -> str:
        name = get(name_col)
        if scale_col and get(scale_col).lower() == "true":
            return f"{name}_rescaled"
        return name

    return CsvMapping(
        evaluations="budget",
        value="loss",
        fields={
            "algorithm_name": Source(column="optimizer_name"),
            "dimension": Source(column="dimension"),
            "budget": Source(column="budget"),
            "maximization": Source(constant=False),
            "suite": Source(constant="NEVERGRAD"),
            "function_name": Source(fn=function_name),
        },
        descriptors="rest",
        skip_empty_value=True,
    )


def read_nevergrad(source: str | Path, dataset_id: str) -> ReadResult:
    """Read a Nevergrad CSV file (or the only CSV file in a folder)."""
    return read_csv(
        source,
        dataset_id,
        nevergrad_mapping,
        reader_name=READER_NAME,
        origin="experiment",
    )
