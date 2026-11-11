# iohinspector dataset format, version 1

Status: draft of version 1.0. Every question of checkpoint H2 was decided on 2026-11-10; the decisions are recorded under [Decisions at H2](#decisions-at-h2).

This page specifies how iohinspector stores benchmark data on disk. Every reader in `iohinspector.io` writes this format, the website stores it, and other tools write it without iohinspector: the R scripts that migrate the `.rds` repository (S05) and the IOHexperimenter Parquet logger (E04). The page is therefore written so that it can be implemented from the text alone.

The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.

## Overview

One dataset is one directory:

| File | Required | Contents |
|---|---|---|
| `manifest.json` | yes | Format version, dataset id, the source files with their sha256 and byte size, the reader and writer, row counts, logger triggers, precision and warnings |
| `runs.parquet` | yes | One row per run |
| `descriptors.parquet` | yes | One row per run and descriptor: the algorithm, its parameters, experiment and run attributes, and tags. See [Descriptors](#descriptorsparquet) |
| `trace.parquet` | yes | One row per logged evaluation, in source order |
| `positions.parquet` | no | The search-space position of logged evaluations. Kept apart from the trace because it is large |

Readers ignore other files in the directory, and `validate_dataset` reports them as warnings.

The format is lossless. Every data row of a source file is stored in `trace.parquet`, including rows that repeat a value, rows of a run that did not improve, and the final evaluation. Best-so-far values, deduplication and alignment are computed during analysis and are never applied when the dataset is written.

The algorithm is not the identifier of a run (decision D17). A run is identified by its `run_id` alone, and described by named, typed descriptors, of which the algorithm name is one. No combination of algorithm, function, dimension and instance has to be unique: repeated runs of one algorithm with different parameters, in one file or in many, are valid. What is compared is chosen per analysis by grouping on any combination of descriptors and `runs.parquet` columns.

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

1. Build the UTF-8 string `"ioh-run-v1\n" + dataset_id + "\n" + source_file + "\n" + decimal(run_index)`, where `\n` is U+000A and `decimal` writes the integer in base 10 without sign or leading zeros.
2. Take the SHA-256 digest of that string.
3. Read the first 8 bytes of the digest as an unsigned big-endian integer, and clear its highest bit.

The result is in [0, 2^63) and is stored as `int64`. Test vector: dataset id `example-dataset`, source file `algA/data_f1/IOHprofiler_f1_DIM5.dat` and run index `0` give run id `7884767605179371010`.

`run_index` counts from 0, in the order the runs appear in the source file. Run ids MUST be unique within a dataset. With 63 bits, the chance of any collision among ten million runs is about 5e-6, and `validate_dataset` reports one if it happens.

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
| `updated` | no | Time of the last change after conversion (an edit of the descriptors), in the syntax of `created` |
| `counts` | yes | `runs`, `descriptors_rows`, `trace_rows`, and `positions_rows` exactly when `positions.parquet` exists. They MUST equal the row counts of the Parquet files |
| `descriptors` | yes | A summary of each descriptor key, see [Manifest summary](#manifest-summary). `{}` for a dataset without runs |
| `renamed_columns` | no | Source names the reader renamed because they are reserved: `from`, `to`, optionally `source`, and `table` (`trace`, the default, or `descriptors`) |
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
  "required": [
    "format", "format_version", "dataset_id", "created", "reader", "writer",
    "sources", "counts", "descriptors"
  ],
  "properties": {
    "format": {"const": "iohinspector-dataset"},
    "format_version": {"type": "string", "pattern": "^1\\.(0|[1-9][0-9]*)$"},
    "dataset_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"},
    "created": {"$ref": "#/$defs/time"},
    "updated": {"$ref": "#/$defs/time"},
    "reader": {"$ref": "#/$defs/software"},
    "writer": {"$ref": "#/$defs/software"},
    "sources": {"type": "array", "items": {"$ref": "#/$defs/source"}},
    "archives": {"type": "array", "items": {"$ref": "#/$defs/archive"}},
    "counts": {
      "type": "object",
      "additionalProperties": false,
      "required": ["runs", "descriptors_rows", "trace_rows"],
      "properties": {
        "runs": {"type": "integer", "minimum": 0},
        "descriptors_rows": {"type": "integer", "minimum": 0},
        "trace_rows": {"type": "integer", "minimum": 0},
        "positions_rows": {"type": "integer", "minimum": 0}
      }
    },
    "descriptors": {
      "type": "object",
      "propertyNames": {"$ref": "#/$defs/descriptor_key"},
      "additionalProperties": {"$ref": "#/$defs/descriptor_summary"}
    },
    "renamed_columns": {"type": "array", "items": {"$ref": "#/$defs/rename"}},
    "warnings": {"type": "array", "items": {"$ref": "#/$defs/warning"}},
    "variable_types": {"type": "null"},
    "extra": {"type": "object"}
  },
  "$defs": {
    "time": {
      "type": "string",
      "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]{1,9})?Z$"
    },
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
        "source": {"$ref": "#/$defs/path"},
        "table": {"enum": ["trace", "descriptors"]}
      }
    },
    "descriptor_key": {
      "type": "string",
      "minLength": 1,
      "maxLength": 256,
      "not": {"pattern": "[\\u0000-\\u001f]"}
    },
    "descriptor_summary": {
      "type": "object",
      "additionalProperties": false,
      "required": ["types", "origins", "runs", "distinct", "examples"],
      "properties": {
        "types": {
          "type": "array",
          "minItems": 1,
          "uniqueItems": true,
          "items": {"enum": ["bool", "float", "int", "string"]}
        },
        "origins": {
          "type": "array",
          "minItems": 1,
          "uniqueItems": true,
          "items": {"enum": ["algorithm", "experiment", "path", "run", "user"]}
        },
        "runs": {"type": "integer", "minimum": 1},
        "distinct": {"type": "integer", "minimum": 1},
        "examples": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string"}}
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

One row per run. "Not null" columns MUST NOT contain nulls. Writers MUST NOT add other columns in version 1.0; every other per-run value is a descriptor. That includes the algorithm: `algorithm_name` and `algorithm_info` are descriptors only, and not columns of this table.

| Column | Type | Null | Meaning |
|---|---|---|---|
| `run_id` | int64 | no | See [Run id](#run-id) |
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
| `n_objectives` | int32 | no | Number of objectives, at least 1. Objective 0 is the trace's `raw_y`, the others are `y1` ... (see [Multiple objectives](#multiple-objectives)) |
| `source_file` | string | no | The file the run's rows were read from. See [Source paths](#source-paths). It MUST be listed in `manifest.sources` |
| `run_index` | int64 | no | Index of the run in `source_file`, counting from 0 |
| `source_rows_sha256` | string | yes | Lower-case hex SHA-256 of the run's data lines in the source, normalised as described below. Null when the source is not text (an `.rds` file, or a logger writing directly) |

`(source_file, run_index)` MUST be unique. When the rows of one run come from several files (COCO `.dat` and `.tdat`), `source_file` is the file the reader treats as primary, and the reader's page says which one that is.

`source_rows_sha256` hashes the run's data lines of `source_file` in source order. Each line is split into tokens at the source's column separator (runs of whitespace for whitespace-separated files), the tokens are joined by a single space (U+0020), and each line is followed by `\n` (U+000A). Header lines, comment lines and blank lines are left out. The tokens are hashed as text, before any parsing, so that `iohinspector verify` can recompute the hash from the source without sharing code with the readers.

**Sort order.** Rows are sorted by `suite`, `function_id`, `dimension`, `source_file` and `run_index`, all ascending. Strings compare byte by byte in UTF-8, as in the C locale. Because `(source_file, run_index)` is unique, this order is total. The position of a run in this order is its *run order*. It depends on no descriptor, so editing descriptors never reorders the trace. Runs of one algorithm usually stay together anyway, because sources keep one algorithm per file or folder.

Run identity does not depend on descriptors either: nothing in the format, and nothing `validate_dataset` checks, assumes that algorithm, function, dimension and instance identify a run.

## descriptors.parquet

One row per run and descriptor. A descriptor is a named, typed value that describes a run: the algorithm name and info, algorithm parameters, experiment and run attributes such as seeds, tags the reader derives from folder names, and tags a user adds. Keys are free-form, so a new kind of attribute needs no format change.

| Column | Type | Null | Meaning |
|---|---|---|---|
| `run_id` | int64 | no | The run, which MUST exist in `runs.parquet` |
| `key` | string | no | The descriptor's name |
| `origin` | string | no | Where the value came from, see below |
| `type` | string | no | The value's type: `string`, `int`, `float`, `bool` |
| `value` | string | no | The value as canonical text, see below. It is exact for every type |
| `value_int` | int64 | yes | The value when `type` is `int` and it lies in [-2^63, 2^63); null otherwise |
| `value_float` | float64 | yes | The value when `type` is `float`; null otherwise |
| `value_bool` | bool | yes | The value when `type` is `bool`; null otherwise |

Origins:

| Origin | Meaning |
|---|---|
| `algorithm` | Given with the algorithm: its name, info and parameters |
| `experiment` | Given for a whole experiment or file, such as IOHexperimenter's experiment attributes |
| `run` | Given for one run, such as its seed |
| `path` | Derived by the reader from folder or file names |
| `user` | Added or changed by a person after conversion |

Types, and what `value` holds for each:

| Type | `value` | Typed column |
|---|---|---|
| `string` | The string itself | none |
| `int` | Decimal digits with an optional `-`, no leading zeros and no `-0` (`0\|-?[1-9][0-9]*`), of any size. Seeds above 2^63 are kept exactly here | `value_int`, when the integer fits in int64 |
| `float` | Text that parses back to the same float64, including the sign of zero; `inf`, `-inf` or `nan`. Writers SHOULD use the shortest such text (Python's `repr`) | `value_float` |
| `bool` | `true` or `false` | `value_bool` |

Typed columns that a row's type does not use MUST be null.

**Values are scalars.** There is no type for lists, objects or "no value" (decided at H2, see [question 1](#decisions-at-h2)). Every known source gives scalar descriptors, so nothing is lost:

- A source key without a value gets no row. Grouping already gives null for runs that lack the key.
- A reader that meets a list, tuple or object converts it to text itself and stores a `string`. The Python helper `encode_descriptor` does not convert: it raises `FormatError` for a list, tuple, dict or None, so that types are kept exactly. `descriptor_frame` skips records whose value is None, and `update_descriptors` does the same for records it is given.

Rules:

- `(run_id, key)` MUST be unique: a run has one value per key.
- One key MAY have different types in different runs (for example `budget` as an `int` in one run and as the string `unlimited` in another). Nothing is converted.
- Every run MUST have the descriptors `algorithm_name` and `algorithm_info`, of type `string`, with origin `algorithm`, or `user` after an edit. They are stored only here. `algorithm_info` is `""` when the source has none.
- Keys have 1 to 256 characters and no control characters (U+0000 to U+001F). A key MUST NOT be a column name of `runs.parquet` or one of the fixed trace columns, because analysis resolves names in both places. A reader that meets such a source name renames it and records that in `manifest.renamed_columns` with `table` set to `descriptors`.
- Rows are sorted by `key` (byte by byte in UTF-8) and then by `run_id`, so that reading a few keys skips most row groups.
- The storage rules of the trace apply, and the key-value metadata `iohinspector.table` is `descriptors`.

**Editing after conversion.** Descriptors can be added, changed and removed after conversion, for example to tag runs. An edit rewrites `descriptors.parquet` and the manifest (its `descriptors` summary, `counts.descriptors_rows` and `updated`), and nothing else: not `runs.parquet`, also when `algorithm_name` or `algorithm_info` changes, and never `trace.parquet` or `positions.parquet`. Run ids do not change. Edited and added values get the origin `user`. `algorithm_name` and `algorithm_info` can be changed but not removed.

**Grouping.** An analysis names what it compares with `by`, a list of names (default `["algorithm_name"]`). Each name is a `runs.parquet` column (such as `function_id` or `dimension`) or a descriptor key; the default is therefore read from the descriptors. A descriptor key becomes one column with one value per run:

- if every value of the key has one type, the column has that type: string for `string`, int64 for `int` (string with the exact decimal text if any value is outside int64), float64 for `float`, boolean for `bool`;
- if the key has several types, the column is a string column of the `value` texts, so that values that differ in type stay apart (the caller may ask for an error instead);
- runs without the key are null.

### Manifest summary

`manifest.descriptors` maps each key to a summary, so that a user interface can offer the keys without reading the table:

| Field | Meaning |
|---|---|
| `types` | The types the key has, sorted |
| `origins` | The origins the key has, sorted |
| `runs` | The number of runs that have the key |
| `distinct` | The number of distinct `(type, value)` pairs |
| `examples` | Up to five distinct `value` texts of the key. Writers SHOULD take the smallest in byte order |

`validate_dataset` checks that the summary matches the table.

## trace.parquet

One row per logged evaluation. The columns come in the order of the table below: the fixed columns, the objective columns, then the extra columns.

| Column | Type | Null | Meaning |
|---|---|---|---|
| `run_id` | int64 | no | The run, which MUST exist in `runs.parquet` |
| `function_id` | int32 | no | Copy of the run's `function_id`, so that filters prune row groups |
| `dimension` | int32 | no | Copy of the run's `dimension`, for the same reason |
| `row` | int64 | no | Position within the run: 0, 1, 2, ... without gaps |
| `evaluations` | int64 | no | The evaluation count the source logged for the row |
| `raw_y` | float64 | no | The objective value as logged; objective 0 when the run has several. NaN is allowed; null is not |
| `y1` ... `y(M-1)` | float64 | see below | Objectives 1 to M-1, only when some run has more than one objective. M is the largest `n_objectives` in `runs.parquet`. See [Multiple objectives](#multiple-objectives) |
| extra columns | float64, int64, bool or string | yes | Every other logged column, such as algorithm parameters or constraint values, with its source name |

**Sort order.** The rows of a run are contiguous and in `row` order, and runs appear in run order (the order of `runs.parquet`). This sorts the trace by suite, function, dimension, source file, run and row. A run MAY have no trace rows.

**Extra columns.** Readers keep every extra logged column. Names MUST be non-empty and MUST NOT be one of the reserved names: the six fixed columns above, every `runs.parquet` column name, `algorithm_name`, `algorithm_info`, and every name matching `y[0-9]+` or `x[0-9]+`. Of those, only the objective columns `y1` ... are written, and only as described above; `y0` is never written. A reader that meets a reserved source name renames the column and records it in `manifest.renamed_columns`. A column that a source file does not have is null for that file's rows.

**Storage.** These rules make filters on function and dimension fast:

- Compression SHOULD be zstd at level 3.
- Row-group statistics (min and max) SHOULD be written for every column; without them for `run_id`, `function_id`, `dimension`, `row` and `evaluations`, filters cannot skip row groups.
- Row groups SHOULD hold between 2^16 and 2^20 rows, and SHOULD start where (`suite`, `function_id`, `dimension`) changes, unless the previous row group would then hold fewer than 2^16 rows.
- Each Parquet file SHOULD carry the key-value metadata `iohinspector.format_version` (for example `1.0`) and `iohinspector.table` (`runs`, `descriptors`, `trace` or `positions`).

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

## Multiple objectives

Version 1 stores multi-objective data in a simple form. Richer support follows with IOHexperimenter (E01), and MAY extend this in a later minor version, or change it in a major one.

- `runs.n_objectives` is the run's number of objectives, any integer from 1.
- `raw_y` holds objective 0 of every run. There is no `y0` column, so no value is stored twice.
- When the largest `n_objectives` M is above 1, the trace has the columns `y1` to `y(M-1)`, of type float64. For a run with k objectives, `y1` to `y(k-1)` are not null (NaN is allowed) and the other `y` columns are null. A single-objective dataset (M = 1) has no `y` columns.
- This follows the layout of the positions, where `x0` ... go up to the largest dimension.
- Single-objective analyses use `raw_y` only, and so treat a multi-objective run by its objective 0. Analyses of several objectives read the `y` columns.
- `best_y` in `runs.parquet` stays what the source reports.

## Reserved for mixed variables

- `manifest.variable_types` is reserved for per-variable types and MUST be null or absent in version 1.

## Validation

`validate_dataset(path)` checks a dataset without reading it all into memory. It reports errors for:

- the manifest: JSON Schema, the format name, an unknown major version, duplicate source paths, and an `archive` that is not listed;
- the schemas: missing or unknown columns, wrong types, nulls in not-null columns, reserved names, positions columns that are not `x0` to `x(D-1)`, objective columns that are not `y1` to `y(M-1)`;
- the runs: duplicate or wrongly computed run ids, duplicate `(source_file, run_index)`, sort order, `n_objectives` below 1, `dimension` below 1, `source_file` not in the manifest, and a malformed `source_rows_sha256`;
- the descriptors: run ids that are not in `runs.parquet`, duplicate `(run_id, key)`, unknown origins or types, a `value` that is not canonical for its type or disagrees with the typed column, typed columns that should be null, invalid or reserved keys, sort order, runs without `algorithm_name` or `algorithm_info`, those two with a type other than `string` or an origin other than `algorithm` or `user`, and a manifest summary that does not match;
- the trace and positions: run ids that are not in `runs.parquet` (referential integrity), `function_id` or `dimension` that differ from the run's, sort order, row numbers with gaps, positions without a trace row, coordinates whose nulls do not match the run's dimension, and objective values whose nulls do not match the run's `n_objectives`;
- the counts: `manifest.counts` against the row counts of the files.

It warns about a codec other than zstd, missing row-group statistics, missing key-value metadata, and unexpected files in the directory.

## Python API

All of it lives in `iohinspector.format`.

| Function | Purpose |
|---|---|
| `write_dataset(path, manifest, runs, trace, *, positions=None, descriptors=None)` | Writes a dataset. `runs` has the columns of `RUNS_SCHEMA` only; `algorithm_name` or `algorithm_info` there is refused. `descriptors` MUST give every run an `algorithm_name`, usually as a record `(run_id, "algorithm_name", name, "algorithm")`; a run without `algorithm_info` gets `""` with origin `algorithm`. So the algorithm has one place in the input as on disk, and a dataset opened with `scan_dataset` writes back unchanged. `trace` holds `y1` to `y(M-1)` when some run has several objectives; a chunk MAY leave out a `y` column that none of its runs needs, and it is null there. `trace` and `positions` may be DataFrames, LazyFrames or iterables of DataFrame chunks, in any order. The writer sorts them with an external merge sort, writes to a temporary directory next to `path`, runs `validate_dataset`, and then moves the result into place. Memory stays near twice `memory_limit` (default 256 MiB) plus one input chunk: writing ten million trace rows in scrambled order peaks at about 0.8 GB resident, including the Python process itself |
| `scan_dataset(path)` | Returns a `StoredDataset` with the manifest and `runs`, `descriptors`, `trace` and `positions` as polars LazyFrames. Refuses an unknown major version, and issues the manifest's warnings as `DatasetWarning` |
| `StoredDataset.run_columns(by)` | A LazyFrame of `run_id` and one column per name in `by`, taken from `runs.parquet` or turned from descriptors into columns, ready for `trace.join(..., on="run_id").group_by(by)` |
| `descriptor_columns(descriptors, keys)` | The same for a descriptors table alone |
| `descriptor_frame(records)`, `encode_descriptor(value)` | Build a descriptors table from `(run_id, key, value, origin)` records with Python values; types are kept exactly |
| `update_descriptors(path, upsert=None, remove_keys=())` | Add, change or remove descriptors of a stored dataset. Rewrites only `descriptors.parquet` and the manifest |
| `validate_dataset(path)` | Returns a `ValidationReport` with errors and warnings; `raise_for_errors()` raises `DatasetValidationError` |
| `read_manifest(path)`, `write_manifest(path, manifest)`, `validate_manifest(manifest)` | Manifest input and output, with the JSON Schema check |
| `new_manifest(dataset_id, reader_name, reader_version, ...)`, `file_entry(local_path, path, ...)` | Build a manifest, and a `sources` entry with the file's sha256 and size |
| `run_id(dataset_id, source_file, run_index)`, `run_ids(...)` | Run ids, for one run or as an Int64 series |
| `RUNS_SCHEMA`, `TRACE_SCHEMA`, `POSITIONS_KEY_SCHEMA`, `MANIFEST_SCHEMA` | The fixed polars schemas and the manifest's JSON Schema |
| `objective_columns(M)`, `position_columns(D)` | The names `y1` to `y(M-1)` and `x0` to `x(D-1)` |

Data that breaks the format raises `FormatError`, an invalid manifest `ManifestError`, and an unknown version `UnsupportedFormatVersion`.

## Decisions at H2

Each entry gives the question, the decision taken at checkpoint H2 on 2026-11-10, and the reasons. Questions 2 and 10 were decided differently from the draft's recommendation, and question 1 with a change; the others as recommended. Questions 1, 2 and 5 came with decision D17; 4, 6 and 7 changed with it. The text above implements every decision.

1. **Descriptors: how per-run values are stored (D17).** *Decided at H2 on 2026-11-10: the long typed table, scalar types only; `json` and `null` dropped.* Every known source gives scalar descriptors (IOHexperimenter experiment attributes are string pairs and run attributes are doubles; `.info`, COCO and Nevergrad give single values), and `null` only told "key named without a value" from "key absent", which no analysis uses. The text below is the design as proposed, and it is kept for the record. The design above is a long table, one row per run and key, with the type kept in a `type` column, the exact value as text in `value`, and the value again in a typed column (`value_int`, `value_float`, `value_bool`) so that polars groups and filters without parsing. Rejected alternatives:
   - *A JSON text column in `runs.parquet`* (the previous draft): grouping needs every row parsed; JSON readers in JavaScript and R turn large integers into doubles; ints and floats blur; there is no cheap summary of the keys.
   - *A struct column:* one set of fields and one type per field for the whole file, so mixed types under one key need conversion, and adding a user tag changes the schema of `runs.parquet`.
   - *One column per key in `runs.parquet`:* the same problems as a struct, plus very wide, mostly empty tables when algorithms log different parameters.
   - *The long table with only a JSON-encoded value:* exact, but every grouping decodes text.
   - *An Arrow union column:* Parquet cannot store unions.

   **Recommendation (accepted, without `json` and `null`): the long typed table**, sorted by key, with the per-key summary in the manifest and in-place editing of descriptors only.
2. **Algorithm columns in `runs.parquet`.** *Decided at H2 on 2026-11-10, changed from the recommendation: `algorithm_name` and `algorithm_info` are descriptors only, and not columns of `runs.parquet`.* They stay required for every run, of type `string`, with origin `algorithm` or `user`. Nothing is stored twice, so nothing has to be kept in agreement: an edit of either rewrites only `descriptors.parquet` and the manifest, and the default `by=["algorithm_name"]` reads the descriptors. `write_dataset` takes them in `descriptors` like every other descriptor, and refuses them as runs columns. The draft recommended keeping both columns in `runs.parquet` as copies, so that the website catalog and the default grouping could read them without a join.
3. **Function and dimension copied into the trace and positions.** *Decided at H2 on 2026-11-10, as recommended: copy `function_id` and `dimension`.* The plan lists only `run_id`, `row`, `evaluations` and `raw_y` as fixed trace columns. With only a hashed `run_id`, a filter on function or dimension cannot use row-group statistics. Sorted, the copies compress to almost nothing on disk, and polars then skips row groups for the single-function views.
4. **Sort order (changed by D17).** *Decided at H2 on 2026-11-10, as recommended: suite, function, dimension, source file and run index*, without the algorithm. The plan said "function, dimension, algorithm, run and row". Run order then depends on nothing that can be edited, so descriptor edits never reorder the trace; and since sources keep one algorithm per file or folder, runs of one algorithm usually stay together. `suite` comes first so that datasets holding several suites are well ordered.
5. **Reserved descriptor keys and one value per key.** *Decided at H2 on 2026-11-10, as recommended: a key has one value per run, and keys may not reuse `runs.parquet` or fixed trace column names*, because `by` resolves names in both places. Readers rename clashes and record them in `renamed_columns`. With question 2, the draft's exception for the algorithm pair is gone: `algorithm_name` and `algorithm_info` are no longer column names, only required keys.
6. **Run id recipe and type (reviewed for D17).** *Decided at H2 on 2026-11-10, as recommended: the first 63 bits of SHA-256 over `ioh-run-v1`, the dataset id, the source path and the 0-based run index, separated by newlines, stored as int64.* It depends on no descriptor: runs of one algorithm with different parameters get different ids because they are different runs in the source, and editing a descriptor never changes an id.
7. **Signed integer types.** *Decided at H2 on 2026-11-10, as recommended: int32 and int64 only*, for R and Java. The earlier code used unsigned 16-bit ids. Descriptor integers outside int64 are kept exactly as text, with `value_int` null.
8. **Unknown `suite` and `algorithm_info`.** *Decided at H2 on 2026-11-10, as recommended: an empty string instead of null.* `suite` is a sort key, and every language sorts nulls differently. With question 2 the rule applies to the `algorithm_info` descriptor, which therefore always exists and always has type `string`.
9. **`best_y` and `evals_used`.** *Decided at H2 on 2026-11-10, as recommended: store them as the source reports them, and null when it reports nothing.* The writer does not compute them from the trace, because with improvement-only or fixed-budget logging the trace does not show every evaluation.
10. **Multiple objectives.** *Decided at H2 on 2026-11-10, changed from the recommendation: keep multi-objective data in a simple form.* `n_objectives` is any integer from 1, `raw_y` is objective 0, and the trace has nullable float64 columns `y1` to `y(M-1)` for the largest `n_objectives` M, null beyond each run's own count, as described under [Multiple objectives](#multiple-objectives). There is no `y0`, so nothing is stored twice, and a single-objective dataset has no `y` columns. Richer support (E01) may extend this later. The draft recommended rejecting `n_objectives` other than 1 and every `y` column in version 1.
11. **Positions layout.** *Decided at H2 on 2026-11-10, as recommended: wide columns `x0...x(D-1)` with D the largest dimension, null beyond each run's dimension*, rather than a list column or one row per coordinate. The objective columns of question 10 follow the same layout.
12. **`source_rows_sha256`.** *Decided at H2 on 2026-11-10, as recommended: hash the whitespace-normalised data lines as text*, as described under runs.parquet, so that `verify` (I09) can recompute it independently. It is null for sources that are not text.
13. **Versioning.** *Decided at H2 on 2026-11-10, as recommended: minor versions only add; readers read newer minor versions with a warning and refuse unknown major versions.*
14. **Function ids for sources without them** (Nevergrad, custom CSV). *Decided at H2 on 2026-11-10, as recommended: such readers assign ids and keep the source's name in `function_name`*, and their reader page says how the ids are assigned. `function_id` is a not-null int32 because it is a sort key and the main filter.
15. **New dependency `jsonschema`.** *Decided at H2 on 2026-11-10, as recommended: depend on `jsonschema>=4`* (already present wherever Jupyter is installed) rather than keep a second, hand-written validator in step with the schema above.
