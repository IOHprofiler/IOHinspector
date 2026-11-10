"""Readers that turn benchmark logs into dataset format v1 (``docs/format.md``).

Every reader is registered in ``READERS``; ``detect_format`` picks the reader for
a path, and ``read`` runs it.
"""

from __future__ import annotations

import os
from pathlib import Path

from ._compat import Reader, ReaderError, ReadResult
from .legacy import detect_legacy, read_legacy

#: One line per reader, in the order ``detect_format`` tries them.
READERS: dict[str, Reader] = {
    "ioh-legacy": Reader(
        "ioh-legacy",
        "legacy IOHprofiler .info + .dat/.cdat/.idat/.tdat",
        detect_legacy,
        read_legacy,
    ),
}


def detect_format(path: str | os.PathLike[str]) -> str:
    """The name of the first reader in ``READERS`` that recognises ``path``."""
    path = Path(path)
    for name, reader in READERS.items():
        if reader.detect(path):
            return name
    raise ReaderError("no reader recognises this input", path)


def read(
    path: str | os.PathLike[str], dataset_id: str, *, format: str | None = None
) -> ReadResult:  # pylint: disable=redefined-builtin
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
    "read_legacy",
]
