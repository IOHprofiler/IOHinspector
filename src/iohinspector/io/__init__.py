"""Readers that turn benchmark logs into dataset format v1 (``docs/format.md``).

Every reader is registered in ``READERS``. ``detect_format`` picks the reader for
a path, and ``read`` runs it; the ``ReadResult`` it returns is written with
``ReadResult.write``.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import Reader, ReaderError, ReadResult
from .coco import detect_coco, read_coco
from .ioh import detect_ioh, read_ioh

#: One line per reader, in the order ``detect_format`` tries them.
READERS: dict[str, Reader] = {
    "ioh": Reader(
        "ioh", "IOHexperimenter JSON + .dat (0.3, 0.4)", detect_ioh, read_ioh
    ),
    "coco": Reader("coco", "COCO single-objective .info + .dat", detect_coco, read_coco),
}


def detect_format(path: str | os.PathLike[str]) -> str:
    """The name of the first reader in ``READERS`` that recognises ``path``."""
    path = Path(path)
    for name, reader in READERS.items():
        if reader.detect(path):
            return name
    raise ReaderError("no reader recognises this input", path)


def read(  # pylint: disable=redefined-builtin
    path: str | os.PathLike[str],
    dataset_id: str,
    *,
    format: str | None = None,  # public name, as in the reader table
) -> ReadResult:
    """Read ``path`` with the reader named ``format``, or the detected one."""
    name = format or detect_format(path)
    if name not in READERS:
        raise ValueError(f"unknown format {name!r}; known: {sorted(READERS)}")
    return READERS[name].read(Path(path), dataset_id)


__all__ = [
    "READERS",
    "ReadResult",
    "Reader",
    "ReaderError",
    "detect_format",
    "read",
    "read_coco",
    "read_ioh",
]
