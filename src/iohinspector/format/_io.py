"""Writing small tables (runs, descriptors). Not public API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from .schema import Column, arrow_schema

PARQUET_OPTIONS: dict[str, Any] = {
    "compression": "zstd",
    "compression_level": 3,
    "write_statistics": True,
}


def write_table(
    df: pl.DataFrame, columns: list[Column], table: str, file: Path
) -> None:
    """Write ``df`` with the format's Arrow schema and key-value metadata."""
    schema = arrow_schema(columns, table)
    frame = df.select([c.name for c in columns])
    pq.write_table(frame.to_arrow().cast(schema), file, **PARQUET_OPTIONS)


def replace_table(
    df: pl.DataFrame, columns: list[Column], table: str, file: Path
) -> None:
    """Write ``df`` next to ``file`` and then move it over ``file``."""
    tmp = file.with_name(f".{file.name}.tmp")
    write_table(df, columns, table, tmp)
    os.replace(tmp, file)
