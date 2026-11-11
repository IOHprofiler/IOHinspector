"""Upgrade hints for the names removed in 0.1.0 (``CHANGELOG.md``).

``HINTS`` is the one table. ``upgrade_hints`` decorates a function so that a removed
keyword argument, or a removed column name such as ``data_id``, fails at once with
its hint. ``removed_names`` makes the ``__getattr__`` of a module or class that
fails the same way for a removed attribute.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import polars as pl

_BY = "use by=[...], a list of columns (runs columns or descriptor keys)"
_DATA = (
    "the old data model is gone; use DataManager (add_folder, add_dataset, select, "
    "load) or iohinspector.format.scan_dataset"
)
_ADD = "use add_folder(path); it takes a file, folder or archive in any format"

HINTS: dict[str, str] = {
    "free_vars": _BY,
    "free_variables": _BY,
    "free_var": _BY,
    "alg_vars": _BY,
    "return_as_pandas": "results are polars DataFrames; call .to_pandas() on them",
    "data_id": "runs are keyed by run_id, a stable hash",
    "data_ids": "use run_ids=[...]",
    "Dataset": _DATA,
    "Scenario": _DATA,
    "Run": _DATA,
    "Function": _DATA,
    "Algorithm": _DATA,
    "Solution": _DATA,
    "METADATA_SCHEMA": _DATA,
    "add_json": _ADD,
    "add_coco_info": _ADD,
    "add_data_set": "use add_folder(path) or add_dataset(path)",
    "extend_overview": "use add_folder(path) or add_dataset(path)",
    "data_sets": "use datasets",
    "select_indexes": "use select(run_ids=[...])",
    "experiment_attributes": "use descriptor_keys, and select(key=[value])",
    "data_attributes": "use trace.collect_schema().names()",
    "NonDominated": "use iohinspector.indicators.NonDominated (final_nondominated)",
}


def _message(name: str) -> str:
    return f"{name!r} was removed in iohinspector 0.1.0: {HINTS[name]}"


def _names(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, pl.DataFrame):
        return [] if "run_id" in value.columns else value.columns
    return list(value) if isinstance(value, (list, tuple)) else []


def upgrade_hints(func: Callable[..., Any]) -> Callable[..., Any]:
    """Fail with a hint on a removed keyword argument or a ``data_id`` column."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for key in kwargs:
            if key in HINTS:
                raise TypeError(_message(key))
        for value in (*args, *kwargs.values()):
            if "data_id" in _names(value):
                raise ValueError(_message("data_id"))
        return func(*args, **kwargs)

    return wrapper


def removed_names(
    owner: str, error: type[Exception] = ImportError
) -> Callable[..., Any]:
    """A ``__getattr__`` for ``owner`` that raises ``error`` with a removed name's hint.

    Modules raise ``ImportError``, so that ``from module import name`` keeps the
    hint; a class passes ``AttributeError``.
    """

    def __getattr__(name: str) -> Any:
        if name in HINTS:
            raise error(_message(name))
        raise AttributeError(f"{owner!r} has no attribute {name!r}")

    return __getattr__
