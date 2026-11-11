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

Any CSV with one row per logged evaluation can be read, given a *mapping*: a small JSON object that says which column holds what. The files of this example are in [`examples/custom_csv`](../examples/custom_csv).

`results.csv`, two algorithms on one function, three runs:

```text
algorithm,function,dim,seed,evals,best_f,sigma
CMA-ES,sphere,5,1,1,12.5,0.5
CMA-ES,sphere,5,1,10,3.2,0.31
CMA-ES,sphere,5,1,100,0.004,0.02
CMA-ES,sphere,5,2,1,15.1,0.5
CMA-ES,sphere,5,2,10,4.0,0.28
CMA-ES,sphere,5,2,100,0.009,0.03
RandomSearch,sphere,5,1,1,14.2,
RandomSearch,sphere,5,1,10,9.8,
RandomSearch,sphere,5,1,100,5.1,
```

`mapping.json`:

```json
{
  "version": 1,
  "evaluations": "evals",
  "value": "best_f",
  "run_by": ["algorithm", "function", "dim", "seed"],
  "fields": {
    "algorithm_name": "algorithm",
    "function_name": "function",
    "dimension": "dim",
    "maximization": {"constant": false}
  },
  "descriptors": ["seed"],
  "trace_columns": ["sigma"]
}
```

- `evaluations` and `value` name the columns with the evaluation count and the objective value (`raw_y`).
- `run_by` lists the columns that together identify a run: rows with the same values are one run. Here that gives three runs.
- `fields` fills the run's fields, from a column or as `{"constant": ...}`. `algorithm_name`, `dimension` and `maximization` are required. Without `function_id`, the function names get ids 1, 2, ... in sorted order.
- `descriptors` keeps columns as run descriptors (here `seed`), so you can select and group by them. `"rest"` keeps every column not used otherwise.
- `trace_columns` keeps columns per evaluation (here `sigma`; an empty cell is null).

In Python, give the mapping as a path or as a dict:

```python
from iohinspector import DataManager, aggregate_convergence

manager = DataManager()
manager.add_folder("results.csv", mapping="mapping.json")
df = manager.load(monotonic=True, include_meta_data=True)
aggregate_convergence(df, by=["algorithm_name"])
```

Or convert once and open the stored dataset, which is faster for large files:

```sh
iohinspector convert results.csv results-dataset --mapping mapping.json
iohinspector verify results.csv results-dataset
```

```python
manager = DataManager()
manager.add_dataset("results-dataset")
```

Every mapping key, with its rules, is documented in the module docstring of `iohinspector.io.custom_csv`. A mapping that is not valid, or names a column the file lacks, fails with the reason and, for a misspelt name, a suggestion. `validate_mapping(mapping, columns)` checks a mapping, and optionally the column names of a file, without reading data; the website uses it while a user builds the mapping. The mapping is stored in `manifest.reader.options.mapping`.
