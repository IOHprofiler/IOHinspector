"""Dataset storage format v1: schemas, writing, scanning, manifests, run ids and validation.

The specification is ``docs/format.md``.
"""

from ._frames import FormatError, attributes_json
from .ids import check_dataset_id, check_source_path, run_id, run_ids
from .manifest import (
    MANIFEST_SCHEMA,
    DatasetWarning,
    FormatWarning,
    ManifestError,
    UnsupportedFormatVersion,
    check_format_version,
    file_entry,
    new_manifest,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from .reader import StoredDataset, scan_dataset
from .schema import (
    FORMAT_VERSION,
    POSITIONS_KEY_SCHEMA,
    RUNS_SCHEMA,
    RUNS_SORT_KEYS,
    TRACE_SCHEMA,
    is_reserved_trace_name,
    position_columns,
)
from .writer import write_dataset

__all__ = [
    "FORMAT_VERSION",
    "MANIFEST_SCHEMA",
    "POSITIONS_KEY_SCHEMA",
    "RUNS_SCHEMA",
    "RUNS_SORT_KEYS",
    "TRACE_SCHEMA",
    "DatasetWarning",
    "FormatError",
    "FormatWarning",
    "ManifestError",
    "StoredDataset",
    "UnsupportedFormatVersion",
    "attributes_json",
    "check_dataset_id",
    "check_format_version",
    "check_source_path",
    "file_entry",
    "is_reserved_trace_name",
    "new_manifest",
    "position_columns",
    "read_manifest",
    "run_id",
    "run_ids",
    "scan_dataset",
    "validate_manifest",
    "write_dataset",
    "write_manifest",
]
