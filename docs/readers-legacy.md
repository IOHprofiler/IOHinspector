# Reader: legacy IOHprofiler `.info` + data files

`iohinspector.io.read_legacy(source, dataset_id)` reads the format the IOHprofiler loggers wrote before the JSON metadata of IOHexperimenter 0.3.3: the original IOHprofiler, the `Default` logger of IOHexperimenter 0.2, and the `.info` Analyzer of IOHexperimenter 0.3.0 to 0.3.2. Its name in `READERS` is `ioh-legacy`. `source` is an `.info` file or a folder, searched recursively. It returns the tables of [format v1](format.md), which `ReadResult.write` stores.

The reader follows IOHanalyzer's `read_index_file__IOH` and `read_dat` (`R/readFiles.R`, IOHanalyzer at `caecada^`), except where this page says otherwise.

## Files

An `.info` file is a sequence of three-line blocks:

```text
suite = "PBO", funcId = 1, funcName = "OneMax", DIM = 16, maximization = "T", algId = "ea", algInfo = "", dynamicAttribute = "seed"
%
data_f1_OneMax/IOHprofiler_f1_DIM16.dat, 1:1000|16;42, 2:1000|15;43
```

1. A header of `key = value` pairs. Values may be in single or double quotes.
2. A comment line, usually `%`.
3. The data file, relative to the `.info` file, then one entry per run: `instance:evaluations|best f`, followed by `;value` for each name in `dynamicAttribute` (names separated by `|`).

A data file holds the runs one after the other. Each run starts with a header line of column names, then one line per logged evaluation:

```text
"function evaluation" "current f(x)" "best-so-far f(x)" "current af(x)+b" "best af(x)+b" "sigma"
1 +3.80000e+01 +3.80000e+01 +3.80000e+01 +3.80000e+01 0.500
```

A header line is a line whose first token is not a number. Runs are split on header lines only.

The logger writes up to four data files with the same base name, each with its own trigger:

| File | Trigger (manifest `triggers`) |
|---|---|
| `.dat` | inferred from the data: `all` if every evaluation is logged, `improvement` if every row but the last improves, otherwise unknown |
| `.tdat` | `fixed_budget` (time points) |
| `.cdat` | `all` |
| `.idat` | `fixed_budget` (every n-th evaluation) |

## Mapping to format v1

| Source | Stored as |
|---|---|
| `suite` | `runs.suite`, `""` when absent |
| `funcId` | `runs.function_id`. A value that is not an integer (IOHanalyzer's extdata has `funcId = bla`) gets an id: the names are numbered in byte order from one above the largest integer `funcId` of the dataset, and a `function_ids_assigned` warning lists them |
| `funcName` | `runs.function_name`. Without it, a non-integer `funcId` is the name. With it, the non-integer `funcId` text is kept as the descriptor `funcId` |
| `DIM` | `runs.dimension` |
| `maximization` or `Maximization` | `runs.maximization`, from `T`, `F`, `TRUE`, `FALSE`, `true`, `false`, `True` or `False` (R's `as.logical`). When absent, minimisation if the suite name contains `bbob` and maximisation otherwise, as IOHanalyzer does (`R/DataSet.R` 56–79), with a `maximization_inferred` warning |
| `algId`, `algInfo` | descriptors `algorithm_name`, `algorithm_info` (origin `algorithm`); `""` when absent |
| other header keys | descriptors with origin `experiment`. Quoted values are strings; unquoted values are `int`, `float` or `string` by their text; `constrained` is a `bool` |
| comment line | descriptor `comment` (origin `experiment`) when it holds more than `%` |
| run entry | `runs.instance`, `runs.evals_used` (the evaluations as written), `runs.best_y` |
| `dynamicAttribute` values | descriptors with origin `run`, of type `float` (the logger writes doubles) |
| data column 1 | `trace.evaluations` (an integer) |
| data column 2 | `trace.raw_y` (`current f(x)`, the objective value as logged) |
| columns `x0` … `x(d-1)`, when all are present | `positions.parquet` |
| every other column | an extra trace column with its source name, float64 |

Descriptor keys and trace columns that clash with reserved names are renamed (`instance_experiment`, `instance_logged`) and listed in `manifest.renamed_columns`.

The runs of a data file are matched in order with the run entries of the blocks that name it; several blocks may name one file, and their entries follow each other. `source_file` is the file the `.info` names; when it does not exist, the first that exists of `.dat`, `.tdat`, `.cdat`, `.idat` (IOHanalyzer's order, `R/DataSet.R` 93–101). `source_rows_sha256` hashes that file's data lines of the run.

## Merging the data files

All data files of a base name are read. They must hold the same number of runs, otherwise the reader fails. A row that two files log for the same evaluation of a run is stored once when every column is identical (NaN equals NaN). Otherwise both rows are kept. Nothing is merged within one file, so rows a file repeats stay. Two trace columns, present only for run sets with more than one file, describe the merge:

- `logged_in`: the kinds that hold the row, joined by `+` in the order `dat`, `tdat`, `cdat`, `idat`, for example `dat+cdat`;
- `merge_conflict`: true for every row of an evaluation that the files log with different values. A `merge_conflict` warning counts them.

Rows are ordered by evaluation, then by kind in that order, then by their position in the file. If a file's evaluations decrease within a run, the files are not merged: their rows follow each other per run, and an `unsorted_evaluations` warning is added.

## Variants

The variants IOHanalyzer accepts, from `R/readFiles.R`, and how this reader treats each. Tests: `tests/io/test_legacy_info.py`, `tests/io/test_legacy.py` and `tests/io/test_legacy_variants.py`.

| Variant | IOHanalyzer | Here |
|---|---|---|
| Values in `'` or `"` | `read.csv(quote = …)`, then removes every `'` (151–167) | unquoted; the quotes decide the descriptor type |
| `maximization` / `Maximization` / absent | both spellings; absent: guess from the suite (`R/DataSet.R` 56–79) | the same, with a warning |
| `constrained` | `as.logical` (170) | `bool` descriptor |
| `dynamicAttribute` | one name; takes the value after the first `;` (186, 200–206) | names split on `\|`, one value per name; a count mismatch fails |
| Backslashes in the data file path | replaced by `/` (215) | the same |
| Record with only the data file | block skipped (182) | data runs kept, without instance (`run_count_mismatch`) |
| Empty run entries (cut-off file) | warns, drops the block's run information (190) | empty entries skipped, the others kept (`incomplete_info_record`) |
| Entries with 0 evaluations (`reset_problem` misuse) | dropped (199) | dropped before matching, listed in `info_run_without_evaluations` |
| Several blocks naming one data file | `merge_indexinfo` concatenates (242, 256) | the same |
| Empty `.info` files | skipped (36) | skipped in a folder (`skipped_file`) |
| COCO `.info` | first data line starting with `%` (517) | not this reader (`skipped_file` in a folder); COCO is `I07` |
| Two-column data files (`TWO_COL`) | detected by the header width (529) | nothing special: column 2 is `raw_y` |
| Header detection | first column without a digit (588) | first token not a number |
| Rows wider than the shortest header | cut to the shortest header (591) | each run uses its own header; a row of another width fails with file:line |
| Only one data file read: `.dat`, else `.tdat`, else `.cdat` | `R/DataSet.R` 93–101 | every file read and merged |
| Non-integer `funcId` | kept as a string | assigned id, name in `function_name` |
| Unquoted header names (`evaluations raw_y`) | accepted | accepted |
| CRLF line endings | accepted by `fread` | accepted |

## Differences from IOHanalyzer that affect analysis

- IOHanalyzer analyses the logged `best-so-far f(x)` column (column 3, `idxTarget` in `R/readFiles.R` 702). Analyses on format v1 compute best-so-far values from `raw_y`. The two agree whenever the logger logged every improvement, which holds for `.dat` and `.cdat` files. When they differ, a `best_so_far_differs` warning says in how many rows. The logged column stays in the trace.
- IOHanalyzer takes a run's budget as the larger of its last logged evaluation and the `.info` count (`R/DataSet.R` 165). The reader stores the `.info` count as `evals_used`; the rule belongs to analysis (`docs/definitions/runtime.md`).

## Errors and warnings

Malformed input raises `ReaderError`, whose message starts with `file:line`: a header that is not `key = value` pairs, a missing `funcId` or `DIM`, an unknown maximisation value, a malformed run entry, a data row whose width differs from its header, a token that is not a number (or not an integer in column 1), a data line before the first header, duplicate column names, files of one base name with different run counts, and text that is not UTF-8.

Warnings in the manifest: `function_ids_assigned`, `maximization_inferred`, `missing_algorithm`, `incomplete_info_record`, `info_run_without_evaluations`, `run_count_mismatch`, `missing_data_file` (the runs are kept without trace rows, the source is listed with a null sha256), `unreferenced_data_file`, `skipped_file`, `merge_conflict`, `unsorted_evaluations`, `limited_precision` (fixed notation: values below 5e-(k+1) are stored as 0) and `best_so_far_differs`.
