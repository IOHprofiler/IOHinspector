"""Readers that turn benchmark logs into dataset format v1 (``docs/format.md``).

Every reader is a line of ``READERS``: ``read(path, dataset_id, **options)``
returns a ``ReadResult``, written with ``ReadResult.write``. ``detect_format``
picks the reader for a path; ``iohinspector.convert`` adds archives and tags.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import Reader, ReaderError, ReadResult
from .coco import detect_coco, read_coco
from .custom_csv import read_custom_csv
from .ioh import detect_ioh, read_ioh
from .nevergrad import is_nevergrad_csv, read_nevergrad

#: The readers, in the order ``detect_format`` tries them. Custom CSV needs a
#: column mapping (option ``mapping``), so it is never detected.
READERS: dict[str, Reader] = {
    r.name: r
    for r in (
        Reader("ioh", "IOHexperimenter JSON + .dat (0.3, 0.4)", detect_ioh, read_ioh),
        Reader("coco", "COCO single-objective .info + .dat", detect_coco, read_coco),
        Reader(
            "nevergrad-csv",
            "one Nevergrad CSV file",
            lambda p: p.is_file() and is_nevergrad_csv(p),
            read_nevergrad,
        ),
        Reader(
            "custom-csv", "one CSV file with a column mapping", None, read_custom_csv
        ),
    )
}


def detect_format(path: str | os.PathLike[str]) -> str:
    """The name of the first reader in ``READERS`` that recognises ``path``."""
    path = Path(path)
    for name, reader in READERS.items():
        if reader.detect is not None and reader.detect(path):
            return name
    raise ReaderError(
        f"no reader recognises this input; give the format ({', '.join(READERS)}), "
        "and a mapping for custom CSV",
        path,
    )


def read(  # pylint: disable=redefined-builtin
    path: str | os.PathLike[str],
    dataset_id: str,
    *,
    format: str | None = None,  # public name, as in the reader table
    **options: Any,
) -> ReadResult:
    """Read ``path`` with the reader named ``format``, or the detected one;
    ``options`` go to the reader (``mapping`` for custom CSV)."""
    name = format or detect_format(path)
    if name not in READERS:
        raise ValueError(f"unknown format {name!r}; known: {sorted(READERS)}")
    return READERS[name].read(Path(path), dataset_id, **options)


__all__ = [
    "READERS",
    "ReadResult",
    "Reader",
    "ReaderError",
    "detect_format",
    "read",
    "read_coco",
    "read_custom_csv",
    "read_ioh",
    "read_nevergrad",
]
