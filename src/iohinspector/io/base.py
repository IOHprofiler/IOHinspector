"""Parts shared by the readers in ``iohinspector.io``.

A reader turns source files into the tables of dataset format v1
(``docs/format.md``) and returns them as a ``ReadResult``, which ``write`` stores
with ``iohinspector.format.write_dataset``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ..format import check_source_path, write_dataset
from ..format.schema import descriptor_key_problem, is_reserved_trace_name

#: A number token: decimal, optionally with an exponent, or inf / nan in any
#: letter case, each with an optional sign (``docs/format.md``, "Value rules").
NUMBER_PATTERN = (
    r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    r"|[+-]?(?:[iI][nN][fF]|[nN][aA][nN])"
)
INTEGER_PATTERN = r"[+-]?[0-9]+"
NUMBER = re.compile(f"(?:{NUMBER_PATTERN})")


class ReaderError(ValueError):
    """Malformed input. The message starts with the file name and, when known, the line."""

    def __init__(
        self,
        message: str,
        path: str | os.PathLike[str] | None = None,
        line: int | None = None,
    ):
        self.path = None if path is None else str(path)
        self.line = line
        where = ""
        if self.path is not None:
            where = f"{self.path}:{line}: " if line is not None else f"{self.path}: "
        super().__init__(where + message)


@dataclass
class ReadResult:
    """The tables a reader produced, ready for ``write_dataset``.

    ``trace`` and ``positions`` hold ``run_id`` and ``row``; the writer adds
    ``function_id`` and ``dimension`` and sorts every table.
    """

    manifest: dict[str, Any]
    runs: pl.DataFrame
    trace: pl.DataFrame
    descriptors: pl.DataFrame
    positions: pl.DataFrame | None = None

    def write(self, path: str | os.PathLike[str], **kwargs: Any) -> dict[str, Any]:
        """Write the dataset to the directory ``path``; returns the manifest written."""
        return write_dataset(
            path,
            self.manifest,
            self.runs,
            self.trace,
            positions=self.positions,
            descriptors=self.descriptors,
            **kwargs,
        )


@dataclass(frozen=True)
class Reader:
    """One entry of the reader table: ``detect(path)`` and ``read(path, dataset_id)``."""

    name: str
    description: str
    detect: Callable[[Path], bool]
    read: Callable[..., ReadResult]


def source_path(file: Path, root: Path) -> str:
    """The path of ``file`` relative to ``root``, in the syntax of ``docs/format.md``."""
    try:
        rel = file.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ReaderError(f"is outside the input folder {root}", file) from exc
    path = rel.as_posix()
    try:
        return check_source_path(path)
    except ValueError as exc:
        raise ReaderError(str(exc), file) from exc


def first_false(mask: pl.Series) -> int | None:
    """Index of the first false (or null) value of a boolean series, or None."""
    bad = (~mask.fill_null(False)).arg_true()
    return None if bad.is_empty() else int(bad[0])


def parse_floats(
    tokens: pl.Series, column: str, null_token: str | None = None
) -> pl.Series | None:
    """float64 values of number tokens, or None when some token is not a number.

    ``null_token`` (such as ``None``) becomes null; without it every token must be
    a number. Use ``check_floats`` to fail instead of returning None.
    """
    valid = tokens.str.contains(f"^(?:{NUMBER_PATTERN})$")
    if null_token is not None:
        valid = valid | (tokens == null_token)
    if first_false(valid) is not None:
        return None
    values = tokens.str.to_lowercase()
    if null_token is not None:
        values = pl.select(
            pl.when(tokens == null_token).then(None).otherwise(values)
        ).to_series()
    return values.cast(pl.Float64).alias(column)


def check_floats(
    tokens: pl.Series, lines: Sequence[int], path: str, column: str
) -> pl.Series:
    """float64 values of number tokens; fails with file:line at the first other token."""
    i = first_false(tokens.str.contains(f"^(?:{NUMBER_PATTERN})$"))
    if i is not None:
        raise ReaderError(
            f"column {column!r}: {tokens[i]!r} is not a number", path, lines[i]
        )
    return tokens.str.to_lowercase().cast(pl.Float64).alias(column)


def check_integers(
    tokens: pl.Series, lines: Sequence[int], path: str, column: str
) -> pl.Series:
    """int64 values of integer tokens; fails with file:line at the first other token."""
    i = first_false(tokens.str.contains(f"^(?:{INTEGER_PATTERN})$"))
    if i is not None:
        raise ReaderError(
            f"column {column!r}: {tokens[i]!r} is not an integer", path, lines[i]
        )
    values = tokens.cast(pl.Int64, strict=False)
    i = first_false(values.is_not_null())
    if i is not None:
        raise ReaderError(
            f"column {column!r}: {tokens[i]} does not fit in int64", path, lines[i]
        )
    return values.alias(column)


class Renamer:
    """Renames source names that the format reserves or that are already taken.

    A clash gets ``<name>_<suffix>``, then ``<name>_<suffix>2`` and so on. Every
    rename is recorded once per (source, table) for ``manifest.renamed_columns``.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, str]] = []
        self._seen: set[tuple[str, str, str, str]] = set()

    def descriptor_key(
        self, key: str, taken: Iterable[str], suffix: str, source: str
    ) -> str:
        return self._rename(
            key,
            set(taken),
            suffix,
            source,
            "descriptors",
            lambda k: descriptor_key_problem(k) is not None,
        )

    def trace_column(self, name: str, taken: Iterable[str], source: str) -> str:
        return self._rename(
            name, set(taken), "logged", source, "trace", is_reserved_trace_name
        )

    def _rename(
        self,
        name: str,
        taken: set[str],
        suffix: str,
        source: str,
        table: str,
        reserved: Callable[[str], bool],
    ) -> str:
        if not reserved(name) and name not in taken:
            return name
        n = 1
        new = f"{name}_{suffix}"
        while reserved(new) or new in taken:
            n += 1
            new = f"{name}_{suffix}{n}"
        key = (name, new, source, table)
        if key not in self._seen:
            self._seen.add(key)
            self.records.append(
                {"from": name, "to": new, "source": source, "table": table}
            )
        return new
