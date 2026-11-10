"""Stand-ins for ``iohinspector.io.base`` until the IOHexperimenter reader (I05) lands.

The legacy reader was written in parallel with I05, which adds ``io/base.py``.
These definitions have the same names and signatures as there. When both are
merged, this module is deleted and ``legacy.py`` imports them from ``.base``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ..format import write_dataset

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
    """The tables a reader produced, ready for ``write_dataset``."""

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
