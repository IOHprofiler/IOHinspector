"""Readers: one per source format, registered in ``READERS``.

Each reader converts a source file or folder to a dataset in the storage format
(``docs/format.md``). Add one line per reader to the table below.
"""

from .custom_csv import convert_custom_csv
from .nevergrad import convert_nevergrad

#: Reader name -> conversion function ``(source, output, ...) -> manifest``.
READERS = {
    "nevergrad-csv": convert_nevergrad,
    "custom-csv": convert_custom_csv,
}

__all__ = ["READERS", "convert_custom_csv", "convert_nevergrad"]
