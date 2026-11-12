# Changelog

## Unreleased

### New

- Parameters, positions and constraint penalties (`docs/definitions/parameters.md`):
  `metrics.get_parameter_sample` and `get_parameter_summary` give parameter
  values at budgets (the last row at or before b) or targets (the first row that
  reaches t). `get_logged_rows` gives the raw rows for correlations and positions.
  `apply_penalty` penalises infeasible rows before best-so-far: v != 0 by default,
  negative violations included, and a penalty always makes a value worse.
- `iohinspector.targets` (`docs/definitions/targets.md`): `seq_fv` and `seq_rt`
  (target and budget grids), `get_ecdf_targets` (per function and dimension:
  `log-linear`, `linear` or the 51 COCO `bbob` targets) and `get_target_dt` (one
  default target or budget per pair, with an explicit `perspective`).
- Stopped runs (`docs/definitions/alignment.md`, AL-9; `fixed-budget.md`, FB-6):
  with an `evals_used` column (`load(include_columns=["evals_used"])`),
  fixed-budget `align_data` flags each row `stopped` (the budget exceeds the
  evaluations the run used) and `aggregate_convergence` reports `n_active`;
  `load(x_values=...)` always adds `stopped`. `load(monotonic=True)` marks each
  run's last logged row `is_final`. No rows are added.
- ECDF, AUC, AOCC and rank tables (`docs/definitions/ecdf.md`):
  `metrics.get_ecdf` is the target ECDF, inclusive (f <= t and T <= x), exact at its
  steps or sampled at `x_values`, averaged over the (function, dimension) pairs
  with equal weight. By default it uses only the pairs every group has and warns
  about the others (`pairs="all"` to average each group's own pairs).
  `continuous=True` gives the continuum ECDF with bounds fixed per pair.
  `metrics.get_ecdf_auc` is the exact normalised area under either; with
  `continuous=True` it is the AOCC. `metrics.get_rank_table` ranks a per-pair
  summary (average ties, missing values last). `get_data_ecdf` and `get_aocc` stay
  as they were (legacy).
- Runtime and fixed-budget summaries as defined in `docs/definitions/runtime.md`
  and `fixed-budget.md` (WP I11):
  - `aggregate_running_time`: per run set and target `n_runs`, `n_success`,
    `success_ratio`, `ERT`, `par_{k}` (`k=10` by default), `mean`, `median`
    and `sd` of the successful runs, and the Hyndman-Fan type 3 quantiles `q2`
    ... `q98`, which count an unfinished run as `inf` unless
    `include_unfinished=False`. A failed run is charged its own `evals_used`
    (`load(include_columns=["evals_used"])`), or else its last logged
    evaluation; `eval_max` is a budget cap. New keyword `targets`.
  - `aggregate_convergence`: per run set and budget `mean`, `median`, `sd`, the
    type 7 quantiles `q2` ... `q98`, `best`, `worst`, `geometric_mean` (only
    when every value is > 0), `n_runs`, `n_defined`, and with `evals_used`
    `n_active`. New keywords `budgets`, and `stopped="restart"` with `seed`
    for simulated restarts of runs that stopped before a budget.
  - `get_overview`: per run set `n_runs`, `budget`, the best, worst, mean and
    median final values, `n_best`, `worst_recorded`, `min_runtime` and
    `max_runtime`.
  - `scripts/bench_ert.py`: the full-bbob ERT table benchmark.
- `metrics.group_label(by, template)`: a label per group of `by` from a template
  such as `"{algorithm_name} (sigma {sigma0})"`. Plots grouped by several columns
  label them `"a, b"` instead of a tuple.

- Distributions (`docs/definitions/distributions.md`): `get_runtime_sample` and
  `get_value_sample` (raw per-run tables), `get_histogram` (R's `nclass.FD` and
  `pretty` breaks, right-closed bins; equal pooled bins by default for several
  groups, DI-4, or `bins="per_algorithm"`) and `get_violin` (Gaussian kernel,
  Silverman bandwidth, box quartiles). Failed runs are counted in `n_failed`.

### Fixed

- Budget grids (`get_sequence(..., cast_to_int=True)` and the default grids of
  `aggregate_convergence`, `get_data_ecdf` and the EAF) are rounded, not truncated:
  a grid up to 100 ended at 99 (TG-5).
- Log-spaced default targets (`aggregate_running_time`) start at the smallest
  positive value when the data reaches 0 or less, instead of failing an assertion
  (TG-3). `get_sequence` raises `ValueError` for a log range with a bound <= 0.

- `aggregate_running_time` and `aggregate_convergence` follow the definitions,
  and some columns are renamed:

  | Before | Now |
  |---|---|
  | `PAR-10` (10 × the largest evaluation of the whole frame) | `par_10` (10 × the largest `evals_used` of the run set) |
  | `success_count` | `n_success` |
  | `std` | `sd` |
  | runtime `mean`, `median`, `std` over all runs (`inf` with one failure) | over the successful runs |
  | runtime `min`, `max` | the quantiles `q2` ... `q98` |
  | fixed-budget `min`, `max` | `best`, `worst` |

  ERT charged every failed run the largest evaluation of the whole frame (all
  algorithms and functions); it now charges the run its own budget (RT-1). The
  geometric mean is null unless every value is > 0 (FB-3). The plots
  (`measures`) fail on a name that is not a column.

- Alignment: before a run's first logged evaluation its value is null, not `inf`
  (`docs/definitions/alignment.md`, AL-5). Under maximisation `inf` counted as the best
  value; 0.1.0 listed this as a known issue. Every metric that aligns leaves such a
  run out of its statistics but counts it: `aggregate_convergence` (new columns
  `n_runs` and `n_defined`), `get_data_ecdf` and the single-objective EAF (it reached
  nothing), `get_trajectory`, `get_eaf_data` and `load(x_values=...)` (null).
  `get_discritized_eaf_single_objective` and `get_eaf_data` take `maximization`.
  This only matters when a budget comes before a run's first logged evaluation
  (IOHexperimenter and COCO always log evaluation 1).

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
| `get_data_ecdf`: a run with no value yet at a budget formed its own group with a null `by` value | it stays in its own group |
