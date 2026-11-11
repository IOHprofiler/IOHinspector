# CSV readers

Two readers turn CSV files into datasets of the [storage format](format.md). Both live in `iohinspector.io`, are listed in `READERS`, and read every data line (no deduplication, no repair of values).

| Reader | Function | Source |
|---|---|---|
| `nevergrad-csv` | `read_nevergrad(source, dataset_id)` | Nevergrad `allxps` result files, `.csv` or `.csv.gz` |
| `custom-csv` | `read_custom_csv(source, dataset_id, mapping)` | Any CSV, with a column mapping given as JSON |

Values follow the format's value rules: float64 tokens are plain decimals, `inf`, `-inf` or `nan`; integers are integers; anything else is a `ReaderError` that names the file and line. The file is read once; the trace goes to temporary files in chunks, so memory grows with the number of runs, not rows. A mapping that is not valid, or names a column that the file lacks, raises `MappingError` with the reason and, for a misspelt name, a suggestion.

## Nevergrad

One row is one run with one trace row. `loss` is `raw_y`, `budget` is the evaluations and the run's budget, `optimizer_name` is `algorithm_name`, and the objective is minimised. The function is `name` (or `function_class`), with `_rescaled` added when `rescale` (or `transform`) is true. Function ids are assigned from the sorted names and listed in `manifest.extra.function_ids`. Every other column is a descriptor with origin `experiment`, typed by the whole column; an empty cell gives no descriptor. Only the fixed-budget result is in these files, so no convergence data exists.

`scripts/download_nevergrad.py` downloads sample files for the tests.

## Custom CSV

The mapping keys, with an example, are documented in the module docstring of `iohinspector.io.custom_csv`. `validate_mapping(mapping, columns)` checks a mapping, and optionally the column names of a file, without reading data; the website uses it while a user builds the mapping. The mapping is stored in `manifest.reader.options.mapping`.
