"""Dataset ids, source paths and stable run ids (``docs/format.md``, "Identifiers")."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import polars as pl

RUN_ID_PREFIX = "ioh-run-v1"
_RUN_ID_MASK = (1 << 63) - 1

DATASET_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_BAD_PATH_CHARS = re.compile(r"[\\\x00-\x1f]")


def check_dataset_id(dataset_id: str) -> str:
    """Return ``dataset_id`` if it is valid, else raise ``ValueError``."""
    if (
        not isinstance(dataset_id, str)
        or DATASET_ID_PATTERN.fullmatch(dataset_id) is None
    ):
        raise ValueError(
            f"invalid dataset id {dataset_id!r}: it must match "
            f"{DATASET_ID_PATTERN.pattern} (1 to 128 characters)"
        )
    return dataset_id


def source_path_problem(path: object) -> str | None:
    """Why ``path`` is not a valid source path, or None if it is valid."""
    if not isinstance(path, str) or not path:
        return "it must be a non-empty string"
    if _BAD_PATH_CHARS.search(path):
        return "it contains a backslash or a control character"
    if path.startswith("/"):
        return "it must be relative (no leading '/')"
    for part in path.split("/"):
        if part in ("", ".", ".."):
            return "it has an empty, '.' or '..' part"
    return None


def check_source_path(path: str) -> str:
    """Return ``path`` if it is a valid source path, else raise ``ValueError``."""
    problem = source_path_problem(path)
    if problem is not None:
        raise ValueError(f"invalid source path {path!r}: {problem}")
    return path


def run_id(dataset_id: str, source_file: str, run_index: int) -> int:
    """The stable run id of run ``run_index`` (0-based) in ``source_file``.

    The first 63 bits of SHA-256 over ``"ioh-run-v1\\n<dataset_id>\\n<source_file>\\n<index>"``.
    """
    if isinstance(run_index, bool) or not isinstance(run_index, int):
        raise TypeError("run_index must be an int")
    if run_index < 0:
        raise ValueError("run_index must be >= 0")
    text = f"{RUN_ID_PREFIX}\n{dataset_id}\n{source_file}\n{run_index}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _RUN_ID_MASK


def run_ids(
    dataset_id: str, source_files: Iterable[str], source_run_indices: Iterable[int]
) -> pl.Series:
    """Run ids for paired ``source_files`` and ``source_run_indices``, as an Int64 series."""
    ids = [
        run_id(dataset_id, f, int(i))
        for f, i in zip(source_files, source_run_indices, strict=True)
    ]
    return pl.Series("run_id", ids, dtype=pl.Int64)
