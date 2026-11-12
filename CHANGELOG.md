# Changelog

## 0.1.0

One data model and one way to load and use data. The outward API stays close to
0.0.x; what changed fails at once with a message that says how to upgrade.

### New

- A stored dataset format (`docs/format.md`): a folder of Parquet tables with a
  manifest that records every source file and its sha256. Converting is lossless:
  every data row of the source is kept.
- `iohinspector convert` and `iohinspector verify` (`docs/convert-verify.md`):
  convert a file, folder or archive in any supported format, and check a converted
  dataset against its sources with an independent scanner. On 192M logged
  evaluations (3.9 GB of IOHexperimenter files), convert takes about 3 minutes, and
  verify about 3 minutes in bounded memory (peak about 1.4 GB).
- Readers for Nevergrad CSV, and for any CSV with a column mapping
  (`docs/readers-csv.md`), next to IOHexperimenter and COCO (`bbob`,
  `bbob-largescale`, `bbob-noisy`, `bbob-boxed`; `docs/readers-coco.md`).
- Descriptors: every attribute of a run (algorithm info and parameters, experiment
  and run attributes, tags from paths or added later) is a typed value that `select`
  and `by` can use.
- polars 1.33+ and 2.x are both supported and tested.

### Fixed (readers of 0.0.x)

- IOHexperimenter: runs were split where `evaluations == 1` instead of at header
  lines, a filter removed rows across run boundaries, the final row of a run could
  be dropped, f = 0 failed an assertion, and run attributes and maximisation were not
  fully read.
- COCO: instance ids were lost, and `.dat` and `.tdat` rows were merged wrongly.

### Data model

- Data is held as the four tables of dataset format v1 (`docs/format.md`): `runs`,
  `descriptors`, `trace` and `positions`. `DataManager` shows them as polars
  LazyFrames (`manager.runs`, `manager.descriptors`, `manager.trace`,
  `manager.positions`).
- Runs are keyed by `run_id`, a stable hash of the dataset id, the source file and the
  run's index in it. The process-global `data_id` counter is gone, so ids no longer
  depend on what was loaded before. `run_id` is no longer 1, 2, ... per file; that
  number is the overview column `source_run_index` (from 0).
- Columns have the format's types: `run_id` int64; `function_id`, `instance` and
  `dimension` int32. `evaluations` stays UInt64 in `load()`.
- A descriptor that is an int in some runs and a float in others (a parameter such as
  `0` and `0.5`) is a float64 column; other mixes of types stay text.

### DataManager

| 0.0.x | 0.1.0 |
|---|---|
| `add_folder(folder)` (IOHexperimenter JSON and COCO) | `add_folder(path, **options)`: a file, folder or archive in any format (IOHexperimenter, COCO, Nevergrad, custom CSV with `mapping=`); options as for `iohinspector convert` |
| — | `add_dataset(path)`: a dataset folder written by `iohinspector convert` |
| `add_json(file)`, `add_coco_info(file)` | `add_folder(file)` |
| `add_data_set`, `extend_overview` | `add_folder` or `add_dataset` |
| adding the same folder twice duplicated it | fails; pass `dataset_id=` for a second copy |
| source warnings (such as limited precision) shown on every load | not shown; they are in each dataset's `manifest["warnings"]` |
| `data_sets` (old `Dataset` objects) | `datasets` (format v1 `StoredDataset`s) |
| `select(data_ids, ...)`, positional | `select(run_ids=...)`, keyword-only |
| `select(experiment_attributes=[(key, value)])` | `select(key=[value])`: any runs column or descriptor |
| `select(data_attributes=...)` | removed |
| `select_indexes(idxs)` | `select(run_ids=...)` |
| `functions`, `algorithms`: tuples of `Function` / `Algorithm` objects, with repeats | sorted tuples of function ids and algorithm names |
| `dimensions`, `instances` | sorted tuples |
| `experiment_attributes` | `descriptor_keys`; the values are `overview` columns |
| `data_attributes` | `manager.trace.collect_schema().names()` |
| `overview`: `data_id`, ..., `evals`, attributes renamed with `_exp_attr` / `_run_attr` | `run_id`, `algorithm_name`, `algorithm_info`, every runs column (`evals` is `evals_used`) and every descriptor; clashing names are renamed by the reader (for example `instance_experiment`) |
| `load(...)`: `data_id`, `run_id` (per file) | `load(...)`: same arguments; `run_id` first, positions `x0` ... joined; a logged property named like a position is renamed (`x0_logged`) |

### Removed

- `iohinspector.data` and its classes `Dataset`, `Scenario`, `Run`, `Function`,
  `Algorithm`, `Solution` and `METADATA_SCHEMA`; use `DataManager`, or
  `iohinspector.format.scan_dataset` for a converted dataset.
- The reader internals that `from iohinspector import *` leaked (`IohJson`,
  `load_ioh_json`, `parse_dat`, `parse_coco_dat`, `merge_files`, ...); they stay in
  `iohinspector.io`. The star import now gives only the names in `__all__`.
- Reading `.parquet` files placed next to IOHexperimenter `.dat` files; convert the
  data with `iohinspector convert` and use `add_dataset`.
- `indicators.anytime.NonDominated`; use `indicators.NonDominated`.
- Placeholders that only raised `NotImplementedError`:
  `plot_multi_function_fixed_target`, `plot_multi_function_fixed_budget`,
  `plots.ranking.robustranking`, `stats_comparison`, `winnning_fraction_heatmap`.

### Metrics and plots

| 0.0.x | 0.1.0 |
|---|---|
| `free_vars=` (`aggregate_running_time`, `aggregate_convergence`, `get_data_ecdf`, `get_aocc`, `plot_ecdf`, `plot_single_function_fixed_target`, `plot_single_function_fixed_budget`) | `by=` |
| `free_variables=` (`get_trajectory`) | `by=` |
| `alg_vars=` (`get_tournament_ratings`, `plot_tournament_ranking`) | `by=` (`fid_vars` stays) |
| `free_var="..."` (`plot_paretofronts_2d`, `plot_indicator_over_time`) | `by=["..."]` |
| `return_as_pandas=` (every metric) | removed; results are polars, call `.to_pandas()` |
| plots return pandas frames | plots return the polars frame they drew |
| frames grouped on `data_id` (`align_data`, `add_indicator`, EAF, attractor network, single-run heatmap) | grouped on `run_id` |
| `get_discritized_eaf_single_objective`: pandas, `eaf_target` as index | polars, `eaf_target` as first column, one column per evaluation value |
| `get_attractor_network`: pandas nodes and edges | polars; a node's number is its row number |

A removed keyword argument raises `TypeError`, and a frame with a `data_id` column
but no `run_id` raises `ValueError`, each with the upgrade hint. The numbers the
metrics compute are unchanged, except for the alignment fixes below.

### Alignment

`align_data` is the one alignment function. It is the former `turbo_align` (one join
for all runs), with what only the old `align_data` did: best-so-far values, `inf`
where a run has no value yet (before its first evaluation, or a target it never
reached), any `group_cols`, and a wide output per group.

| 0.0.x | 0.1.0 |
|---|---|
| `turbo_align(df, x_values, ...)` | `align_data(df, x_values, ...)` |
| `align_data(..., silence_warning=True)` | `align_data(...)`; it no longer warns |
| `get_data_ecdf(..., turbo=True)` | `get_data_ecdf(...)`; the alignment is always the fast one |
| `turbo_align`, so `plot_ecdf` and the single-objective EAF, used `raw_y` as given | best-so-far, so a frame loaded with `monotonic=False` gives the same result as with `monotonic=True` |
| `turbo_align` left a missing value null | `inf` |
| the old `align_data` set every numeric column of a missing row to `inf` (so means of `optimum`, `budget` or a run attribute could become `inf`) | only the aligned column is `inf`; the others stay null |
| fixed target: on a tie (the best-so-far value repeats, for example the final row of a monotonic frame) the old `align_data` could take a later row | the first evaluation that reached the target. This changes `aggregate_running_time` where the old result was a later evaluation (one target of the test data: minimum 100 becomes 82, ERT 1500 becomes 1482) |
| `load(x_values=...)` on maximisation runs | aligned per run with its own direction |
