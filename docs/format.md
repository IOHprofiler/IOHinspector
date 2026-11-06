# iohinspector dataset format, version 1

Status: draft, waiting for sign-off at checkpoint H2. The decisions that need sign-off are listed under [Open questions for H2](#open-questions-for-h2).

This page specifies how iohinspector stores benchmark data on disk. Every reader in `iohinspector.io` writes this format, the website stores it, and other tools write it without iohinspector: the R scripts that migrate the `.rds` repository (S05) and the IOHexperimenter Parquet logger (E04). The page is therefore written so that it can be implemented from the text alone.

The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.

## Overview

One dataset is one directory:

| File | Required | Contents |
|---|---|---|
| `manifest.json` | yes | Format version, dataset id, the source files with their sha256 and byte size, the reader and writer, row counts, logger triggers, precision and warnings |
| `runs.parquet` | yes | One row per run |
| `trace.parquet` | yes | One row per logged evaluation, in source order |
| `positions.parquet` | no | The search-space position of logged evaluations. Kept apart from the trace because it is large |

Readers ignore other files in the directory, and `validate_dataset` reports them as warnings.

The format is lossless. Every data row of a source file is stored in `trace.parquet`, including rows that repeat a value, rows of a run that did not improve, and the final evaluation. Best-so-far values, deduplication and alignment are computed during analysis and are never applied when the dataset is written.

## Versioning

`format_version` in the manifest is a string `"MAJOR.MINOR"`. This page defines version `1.0`.

- A minor version only adds things: optional manifest fields, optional columns, or new allowed values of an enumeration. A reader that supports `1.x` reads every `1.y`, and ignores fields and columns it does not know. iohinspector warns when it reads a newer minor version.
- Every other change is a new major version.
- Readers MUST refuse a major version they do not know. iohinspector raises `UnsupportedFormatVersion`.

## Value rules

- Floating-point values are IEEE 754 float64. When the source is text, the token is parsed as float64. `inf`, `-inf` and `nan` (in any letter case, with an optional sign) map to their IEEE values. Any other token that does not parse is an error, reported with the file name and line number. A reader never replaces a value.
- Integers are signed: `int32` for small counts and ids, `int64` for evaluations, row numbers and run ids. The format uses no unsigned types, so that R, Java and JavaScript read every column without loss.
- Strings are UTF-8.
- The precision of the source is recorded in the manifest (`sources[].precision`), and a warning is added when it loses information. For example, IOHexperimenter writes values with `{:.10f}`, so every value below 5e-11 is written as 0. Nothing is corrected.

## Identifiers

### Dataset id

`dataset_id` matches `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`. The program that creates the dataset chooses it. Because run ids depend on it, converting the same sources with the same dataset id always gives the same run ids.

### Source paths

`source_file` in `runs.parquet` and `path` in the manifest name a source file by its path relative to the root of the input (the folder or archive given to the converter):

- `/` separates parts; there is no leading `/`, no empty part, and no `.` or `..` part;
- no backslash and no control character (U+0000 to U+001F);
- a file inside an archive is named `<archive path>/<member path>`, for example `algA.zip/data_f1/IOHprofiler_f1_DIM5.dat`, so paths stay unique when two archives contain the same member;
- a writer without source files (such as a logger writing the dataset directly) uses a stable logical name in the same syntax, and lists it in the manifest with `sha256` and `bytes` set to null.

### Run id

`run_id` is a stable hash of the dataset id, the source file and the run's index in that file:

1. Build the UTF-8 string `"ioh-run-v1\n" + dataset_id + "\n" + source_file + "\n" + decimal(source_run_index)`, where `\n` is U+000A and `decimal` writes the integer in base 10 without sign or leading zeros.
2. Take the SHA-256 digest of that string.
3. Read the first 8 bytes of the digest as an unsigned big-endian integer, and clear its highest bit.

The result is in [0, 2^63) and is stored as `int64`. Test vector: dataset id `example-dataset`, source file `algA/data_f1/IOHprofiler_f1_DIM5.dat` and run index `0` give run id `7884767605179371010`.

`source_run_index` counts from 0, in the order the runs appear in the source file. Run ids MUST be unique within a dataset. With 63 bits, the chance of any collision among ten million runs is about 5e-6, and `validate_dataset` reports one if it happens.

## manifest.json

A UTF-8 JSON object. Writers SHOULD write it with sorted keys and an indent of 2, so that two conversions of the same input differ only in `created`.

| Field | Required | Meaning |
|---|---|---|
| `format` | yes | Always `"iohinspector-dataset"` |
| `format_version` | yes | `"1.0"` for this page |
| `dataset_id` | yes | See [Dataset id](#dataset-id) |
| `created` | yes | Conversion time in UTC, RFC 3339 with a `Z` suffix, for example `2026-11-06T12:00:00Z` |
| `reader` | yes | The reader that parsed the sources: `name`, `version`, and optionally the `options` it ran with |
| `writer` | yes | The program that wrote the Parquet files: `name` and `version`. iohinspector fills this in |
| `sources` | yes | One entry per source file, possibly empty. See below |
| `archives` | no | Archives the sources were extracted from: `path`, `sha256`, `bytes`, and `archive` when nested |
| `counts` | yes | `runs`, `trace_rows`, and `positions_rows` exactly when `positions.parquet` exists. They MUST equal the row counts of the Parquet files |
| `renamed_columns` | no | Extra trace columns the reader renamed because their source name is reserved: `from`, `to`, and optionally `source` |
| `warnings` | no | Problems found in the sources: `code` (lower case with underscores, for example `limited_precision`), `message`, and optionally `source` |
| `variable_types` | no | Reserved for mixed variable types. MUST be null or absent in version 1 |
| `extra` | no | A free-form object for reader-specific details |

A `sources` entry has:

| Field | Required | Meaning |
|---|---|---|
| `path` | yes | See [Source paths](#source-paths) |
| `sha256` | yes | Lower-case hex SHA-256 of the file's bytes, or null for a logical source |
| `bytes` | yes | File size in bytes, or null for a logical source |
| `kind` | no | The reader's name for the file type, for example `ioh-json`, `ioh-dat`, `coco-info`, `coco-tdat` |
| `archive` | no | The `archives[].path` the file was extracted from |
| `data_rows` | no | Number of data rows read from the file. When rows of two files are merged (COCO `.dat` and `.tdat`), this counts the rows in the file, before merging |
| `logger` | no | The logging software named in the file: `name` and `version` |
| `triggers` | no | How the logger chose which evaluations to write. A list of objects with `kind` (`all`, `improvement`, `fixed_budget`, `fixed_target` or `other`) and optional `params`. Absent means unknown |
| `precision` | no | How numbers were written: `notation` (`fixed`, `scientific`, `general`, `roundtrip` or `unknown`) and `digits`. IOHexperimenter's `{:.10f}` is `{"notation": "fixed", "digits": 10}` |

### JSON Schema

The schema below is normative. iohinspector ships the same schema as `iohinspector.format.MANIFEST_SCHEMA`, and a test keeps the two identical. When a dataset has a newer minor version than the reader, the reader validates it with `additionalProperties` relaxed.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "urn:iohinspector:dataset-manifest:1",
  "title": "iohinspector dataset manifest, format version 1",
  "type": "object",
  "additionalProperties": false,
  "required": ["format", "format_version", "dataset_id", "created", "reader", "writer", "sources", "counts"],
  "properties": {
    "format": {"const": "iohinspector-dataset"},
    "format_version": {"type": "string", "pattern": "^1\\.(0|[1-9][0-9]*)$"},
    "dataset_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"},
    "created": {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]{1,9})?Z$"},
    "reader": {"$ref": "#/$defs/software"},
    "writer": {"$ref": "#/$defs/software"},
    "sources": {"type": "array", "items": {"$ref": "#/$defs/source"}},
    "archives": {"type": "array", "items": {"$ref": "#/$defs/archive"}},
    "counts": {
      "type": "object",
      "additionalProperties": false,
      "required": ["runs", "trace_rows"],
      "properties": {
        "runs": {"type": "integer", "minimum": 0},
        "trace_rows": {"type": "integer", "minimum": 0},
        "positions_rows": {"type": "integer", "minimum": 0}
      }
    },
    "renamed_columns": {"type": "array", "items": {"$ref": "#/$defs/rename"}},
    "warnings": {"type": "array", "items": {"$ref": "#/$defs/warning"}},
    "variable_types": {"type": "null"},
    "extra": {"type": "object"}
  },
  "$defs": {
    "path": {
      "type": "string",
      "pattern": "^[^/\\\\\\u0000-\\u001f][^\\\\\\u0000-\\u001f]*$",
      "not": {"pattern": "(^|/)\\.{1,2}(/|$)|//|/$"}
    },
    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "software": {
      "type": "object",
      "additionalProperties": false,
      "required": ["name", "version"],
      "properties": {
        "name": {"type": "string", "minLength": 1},
        "version": {"type": "string"},
        "options": {"type": "object"}
      }
    },
    "source": {
      "type": "object",
      "additionalProperties": false,
      "required": ["path", "sha256", "bytes"],
      "properties": {
        "path": {"$ref": "#/$defs/path"},
        "sha256": {"oneOf": [{"$ref": "#/$defs/sha256"}, {"type": "null"}]},
        "bytes": {"type": ["integer", "null"], "minimum": 0},
        "kind": {"type": "string", "minLength": 1},
        "archive": {"$ref": "#/$defs/path"},
        "data_rows": {"type": "integer", "minimum": 0},
        "logger": {"$ref": "#/$defs/software"},
        "triggers": {"type": "array", "items": {"$ref": "#/$defs/trigger"}},
        "precision": {"$ref": "#/$defs/precision"}
      }
    },
    "archive": {
      "type": "object",
      "additionalProperties": false,
      "required": ["path", "sha256", "bytes"],
      "properties": {
        "path": {"$ref": "#/$defs/path"},
        "sha256": {"$ref": "#/$defs/sha256"},
        "bytes": {"type": "integer", "minimum": 0},
        "archive": {"$ref": "#/$defs/path"}
      }
    },
    "trigger": {
      "type": "object",
      "additionalProperties": false,
      "required": ["kind"],
      "properties": {
        "kind": {"enum": ["all", "improvement", "fixed_budget", "fixed_target", "other"]},
        "params": {"type": "object"}
      }
    },
    "precision": {
      "type": "object",
      "additionalProperties": false,
      "required": ["notation"],
      "properties": {
        "notation": {"enum": ["fixed", "scientific", "general", "roundtrip", "unknown"]},
        "digits": {"type": ["integer", "null"], "minimum": 0}
      }
    },
    "rename": {
      "type": "object",
      "additionalProperties": false,
      "required": ["from", "to"],
      "properties": {
        "from": {"type": "string", "minLength": 1},
        "to": {"type": "string", "minLength": 1},
        "source": {"$ref": "#/$defs/path"}
      }
    },
    "warning": {
      "type": "object",
      "additionalProperties": false,
      "required": ["code", "message"],
      "properties": {
        "code": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
        "message": {"type": "string"},
        "source": {"$ref": "#/$defs/path"}
      }
    }
  }
}
```

## runs.parquet

One row per run. "Not null" columns MUST NOT contain nulls. Writers MUST NOT add other columns in version 1.0; per-run information without a column goes into `attributes`.

| Column | Type | Null | Meaning |
|---|---|---|---|
| `run_id` | int64 | no | See [Run id](#run-id) |
| `algorithm_name` | string | no | Algorithm name as given by the source |
| `algorithm_info` | string | no | Free-text algorithm description; `""` when the source has none |
| `suite` | string | no | Problem suite, for example `BBOB`, `PBO` or `bbob`; `""` when the source does not name one |
| `function_id` | int32 | no | Function id within the suite. Readers of formats that name functions without ids assign ids and keep the name in `function_name` |
| `function_name` | string | yes | Function name, when known |
| `instance` | int32 | yes | Problem instance, when the source has instances |
| `dimension` | int32 | no | Search-space dimension, at least 1 |
| `maximization` | bool | no | True when larger values are better |
| `optimum` | float64 | yes | The optimal value, when known |
| `budget` | int64 | yes | The evaluation budget the run was given, when the source records it |
| `evals_used` | int64 | yes | Evaluations the run used, as the source reports them. It can be larger than the last logged evaluation, for example with improvement-only logging |
| `best_y` | float64 | yes | The best value as the source reports it, for example IOHexperimenter's `best.y`. Analysis computes best-so-far values from the trace and does not rely on this column |
| `n_objectives` | int32 | no | Number of objectives. MUST be 1 in version 1 (see [Reserved fields](#reserved-for-multiple-objectives-and-mixed-variables)) |
| `attributes` | string | no | A JSON object (RFC 8259 text, so no NaN or Infinity) with every other run attribute from the source, such as seeds and algorithm parameters. `{}` when there are none. Writers SHOULD sort keys and use compact separators |
| `source_file` | string | no | The file the run's rows were read from. See [Source paths](#source-paths). It MUST be listed in `manifest.sources` |
| `source_run_index` | int64 | no | Index of the run in `source_file`, counting from 0 |
| `source_rows_sha256` | string | yes | Lower-case hex SHA-256 of the run's data lines in the source, normalised as described below. Null when the source is not text (an `.rds` file, or a logger writing directly) |

`(source_file, source_run_index)` MUST be unique. When the rows of one run come from several files (COCO `.dat` and `.tdat`), `source_file` is the file the reader treats as primary, and the reader's page says which one that is.

`source_rows_sha256` hashes the run's data lines of `source_file` in source order. Each line is split into tokens at the source's column separator (runs of whitespace for whitespace-separated files), the tokens are joined by a single space (U+0020), and each line is followed by `\n` (U+000A). Header lines, comment lines and blank lines are left out. The tokens are hashed as text, before any parsing, so that `iohinspector verify` can recompute the hash from the source without sharing code with the readers.

**Sort order.** Rows are sorted by `suite`, `function_id`, `dimension`, `algorithm_name`, `algorithm_info`, `source_file` and `source_run_index`, all ascending. Strings compare byte by byte in UTF-8, as in the C locale. Because `(source_file, source_run_index)` is unique, this order is total. The position of a run in this order is its *run order*.

## trace.parquet

One row per logged evaluation.

| Column | Type | Null | Meaning |
|---|---|---|---|
| `run_id` | int64 | no | The run, which MUST exist in `runs.parquet` |
| `function_id` | int32 | no | Copy of the run's `function_id`, so that filters prune row groups |
| `dimension` | int32 | no | Copy of the run's `dimension`, for the same reason |
| `row` | int64 | no | Position within the run: 0, 1, 2, ... without gaps |
| `evaluations` | int64 | no | The evaluation count the source logged for the row |
| `raw_y` | float64 | no | The objective value as logged. NaN is allowed; null is not |
| extra columns | float64, int64, bool or string | yes | Every other logged column, such as algorithm parameters or constraint values, with its source name |

**Sort order.** The rows of a run are contiguous and in `row` order, and runs appear in run order (the order of `runs.parquet`). This sorts the trace by suite, function, dimension, algorithm, run and row. A run MAY have no trace rows.

**Extra columns.** Readers keep every extra logged column. Names MUST be non-empty and MUST NOT be one of the reserved names: the six columns above, every `runs.parquet` column name, and every name matching `y[0-9]+` or `x[0-9]+`. A reader that meets a reserved source name renames the column and records it in `manifest.renamed_columns`. A column that a source file does not have is null for that file's rows.

**Storage.** These rules make filters on function and dimension fast:

- Compression SHOULD be zstd at level 3.
- Row-group statistics (min and max) SHOULD be written for every column; without them for `run_id`, `function_id`, `dimension`, `row` and `evaluations`, filters cannot skip row groups.
- Row groups SHOULD hold between 2^16 and 2^20 rows, and SHOULD start where (`suite`, `function_id`, `dimension`) changes, unless the previous row group would then hold fewer than 2^16 rows.
- Each Parquet file SHOULD carry the key-value metadata `iohinspector.format_version` (for example `1.0`) and `iohinspector.table` (`runs`, `trace` or `positions`).

## positions.parquet

Optional. One row per logged evaluation whose position the source recorded.

| Column | Type | Null | Meaning |
|---|---|---|---|
| `run_id` | int64 | no | As in the trace |
| `function_id` | int32 | no | As in the trace |
| `dimension` | int32 | no | As in the trace |
| `row` | int64 | no | The trace row this position belongs to. `(run_id, row)` MUST exist in the trace and MUST be unique |
| `x0` ... `x(D-1)` | float64 | see below | The coordinates. D is the largest `dimension` in `runs.parquet`. For a run of dimension d, `x0` to `x(d-1)` are not null and the other columns are null |

Rows follow the trace order, and not every trace row needs a position. Integer and binary variables are stored as float64, which is exact up to 2^53. The storage rules of the trace apply.

## Reserved for multiple objectives and mixed variables

Version 1 only reserves room; the design follows after launch with IOHexperimenter (E01). Single-objective data does not depend on it.

- `runs.n_objectives` exists and MUST be 1.
- Trace columns named `y0`, `y1`, ... are reserved for objective values and MUST NOT be written.
- `manifest.variable_types` is reserved for per-variable types and MUST be null or absent.

## Validation

`validate_dataset(path)` checks a dataset without reading it all into memory. It reports errors for:

- the manifest: JSON Schema, the format name, an unknown major version, duplicate source paths, and an `archive` that is not listed;
- the schemas: missing or unknown columns, wrong types, nulls in not-null columns, reserved names, positions columns that are not `x0` to `x(D-1)`;
- the runs: duplicate or wrongly computed run ids, duplicate `(source_file, source_run_index)`, sort order, `n_objectives` other than 1, `dimension` below 1, invalid `attributes` JSON, `source_file` not in the manifest, and a malformed `source_rows_sha256`;
- the trace and positions: run ids that are not in `runs.parquet` (referential integrity), `function_id` or `dimension` that differ from the run's, sort order, row numbers with gaps, positions without a trace row, and coordinates whose nulls do not match the run's dimension;
- the counts: `manifest.counts` against the row counts of the files.

It warns about a codec other than zstd, missing row-group statistics, missing key-value metadata, and unexpected files in the directory.

## Python API

All of it lives in `iohinspector.format`.

| Function | Purpose |
|---|---|
| `write_dataset(path, manifest, runs, trace, positions=None)` | Writes a dataset. `trace` and `positions` may be DataFrames, LazyFrames or iterables of DataFrame chunks in any order; the writer sorts them with an external merge sort in bounded memory, writes to a temporary directory, validates, and then moves the result into place |
| `scan_dataset(path)` | Returns a `StoredDataset` with the manifest and `runs`, `trace` and `positions` as polars LazyFrames. Refuses an unknown major version |
| `validate_dataset(path)` | Returns a `ValidationReport` with errors and warnings |
| `read_manifest(path)`, `write_manifest(path, manifest)`, `validate_manifest(manifest)` | Manifest input and output |
| `new_manifest(dataset_id, reader_name, reader_version, ...)` | Builds a manifest for a reader to fill in |
| `run_id(dataset_id, source_file, source_run_index)`, `run_id_expr(...)` | Run ids, for one run or as a polars expression |
| `RUNS_SCHEMA`, `TRACE_SCHEMA`, `POSITIONS_KEY_SCHEMA` | Polars schemas of the fixed columns |

## Open questions for H2

Each question gives the recommendation that this draft implements.

1. **`attributes`: JSON text or a struct?** A struct is typed and can be queried in polars, but every run in a file must share one set of fields and one type per field, a value of the wrong type would have to be converted, and datasets with different fields are hard to combine on the website. **Recommendation: JSON text**, which keeps any value without conversion and is read the same way in Python, R, JavaScript and Postgres.
2. **Function and dimension copied into the trace and positions.** The plan lists only `run_id`, `row`, `evaluations` and `raw_y` as fixed trace columns. With only a hashed `run_id`, a filter on function or dimension cannot use row-group statistics. **Recommendation: copy `function_id` and `dimension`**. Sorted, they compress to almost nothing on disk, and polars then skips row groups for the single-function views.
3. **Run id recipe and type.** **Recommendation:** the first 63 bits of SHA-256 over `ioh-run-v1`, the dataset id, the source path and the 0-based run index, separated by newlines, stored as int64. A string id would be easier to read but slower to join; an unsigned id is lost in R and Java.
4. **Sort order.** The plan says "function, dimension, algorithm, run and row". **Recommendation:** put `suite` first, so that datasets holding several suites are well ordered, and then order runs within an algorithm by source file and run index, which keeps the source order.
5. **Signed integer types.** The current code uses unsigned 16-bit ids. **Recommendation: int32 and int64 only**, for R and Java.
6. **Unknown `suite` and `algorithm_info`.** **Recommendation: an empty string instead of null**, because both are sort keys and every language sorts nulls differently.
7. **`best_y` and `evals_used`.** **Recommendation: store them as the source reports them, and null when it reports nothing.** The writer does not compute them from the trace, because with improvement-only or fixed-budget logging the trace does not show every evaluation.
8. **Multiple objectives.** **Recommendation: version 1 rejects `n_objectives` other than 1 and any `y0...` column**, and leaves the design to E01 and a later version.
9. **Positions layout.** **Recommendation: wide columns `x0...x(D-1)` with D the largest dimension, null beyond each run's dimension**, rather than a list column or one row per coordinate.
10. **`source_rows_sha256`.** **Recommendation: hash the whitespace-normalised data lines as text**, as described under runs.parquet, so that `verify` (I09) can recompute it independently. It is null for sources that are not text.
11. **Versioning.** **Recommendation: minor versions only add; readers read newer minor versions with a warning and refuse unknown major versions.**
12. **New dependency `jsonschema`.** The manifest is validated against the JSON Schema above. **Recommendation: depend on `jsonschema>=4`** (already present wherever Jupyter is installed) rather than keep a second, hand-written validator in step with the schema.
