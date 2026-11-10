"""Dataset storage format v1: schemas, writing, scanning, manifests, run ids and validation.

The specification is ``docs/format.md``.
"""

from ._frames import FormatError
from .descriptors import (
    decode_descriptor,
    descriptor_columns,
    descriptor_frame,
    encode_descriptor,
    update_descriptors,
)
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
    DESCRIPTORS_SCHEMA,
    FORMAT_VERSION,
    POSITIONS_KEY_SCHEMA,
    RUNS_SCHEMA,
    RUNS_SORT_KEYS,
    TRACE_SCHEMA,
    is_reserved_trace_name,
    objective_columns,
    position_columns,
)
from .validate import DatasetValidationError, ValidationReport, validate_dataset
from .writer import write_dataset

__all__ = [
    "DESCRIPTORS_SCHEMA",
    "FORMAT_VERSION",
    "MANIFEST_SCHEMA",
    "POSITIONS_KEY_SCHEMA",
    "RUNS_SCHEMA",
    "RUNS_SORT_KEYS",
    "TRACE_SCHEMA",
    "DatasetValidationError",
    "DatasetWarning",
    "FormatError",
    "FormatWarning",
    "ManifestError",
    "StoredDataset",
    "UnsupportedFormatVersion",
    "ValidationReport",
    "check_dataset_id",
    "check_format_version",
    "check_source_path",
    "decode_descriptor",
    "descriptor_columns",
    "descriptor_frame",
    "encode_descriptor",
    "file_entry",
    "is_reserved_trace_name",
    "new_manifest",
    "objective_columns",
    "position_columns",
    "read_manifest",
    "run_id",
    "run_ids",
    "scan_dataset",
    "update_descriptors",
    "validate_dataset",
    "validate_manifest",
    "write_dataset",
    "write_manifest",
]
