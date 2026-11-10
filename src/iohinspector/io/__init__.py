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
from .custom_csv import convert_custom_csv
from .ioh import detect_ioh, read_ioh
from .legacy import detect_legacy, read_legacy
from .nevergrad import convert_nevergrad

#: One line per reader, in the order ``detect_format`` tries them.
READERS: dict[str, Reader] = {
    "ioh": Reader(
        "ioh", "IOHexperimenter JSON + .dat (0.3, 0.4)", detect_ioh, read_ioh
    ),
    "ioh-legacy": Reader(
        "ioh-legacy",
        "legacy IOHprofiler .info + .dat/.cdat/.idat/.tdat",
        detect_legacy,
        read_legacy,
    ),
    "coco": Reader(
        "coco", "COCO single-objective .info + .dat", detect_coco, read_coco
    ),
}

#: Streaming converters that write a dataset themselves:
#: ``(source, output, ...) -> manifest``. The custom CSV converter needs a column
#: mapping, so neither is tried by ``detect_format``; I09's convert command brings
#: both tables together.
CONVERTERS = {
    "nevergrad-csv": convert_nevergrad,
    "custom-csv": convert_custom_csv,
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
    "CONVERTERS",
    "READERS",
    "ReadResult",
    "Reader",
    "ReaderError",
    "convert_custom_csv",
    "convert_nevergrad",
    "detect_format",
    "read",
    "read_coco",
    "read_ioh",
    "read_legacy",
]
