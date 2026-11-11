# WP I26: One data model and user API
Branch: wp/I26-one-data-model
Status: in progress

## Done
- step 1: inventory (this file)
- steps 2-3: `_upgrade.py`; `DataManager` on the four tables; `read_dataset`;
  `data.py` is a stub; metrics, plots, indicators and `align` on `run_id`, `by`,
  polars; tests updated; notebooks updated and running; `examples/parquet` removed.

- step 4: `CHANGELOG.md` (every rename and removal), README, `FUNCTION_REFERENCE.md`;
  notebooks re-executed (stored outputs match; stderr outputs dropped, they held local
  paths); `pyproject.toml` version 0.1.0 for the tag.

## Next step
Step 5: lean pass over the changed modules; line counts; final notes.

## Baseline
- `src/iohinspector`: 11,873 lines of Python; `tests`: 9,259 lines.
- `pytest -q`: 664 passed, 3 skipped, 2 xfailed (26 s), notebooks included
  (`tests/test_tutorials.py`).

## Inventory: public names (`from iohinspector import *`, 0.0.8 + I25)

There is no `__all__`, so the star import gives 120 names, about 40 of them leaked
imports. After I26 the package defines `__all__` at every level.

| Name(s) | Today | After I26 |
|---|---|---|
| `DataManager` | over `data.Dataset` objects | kept; over the four tables (see below) |
| `Dataset`, `Scenario`, `Run`, `Function`, `Algorithm`, `Solution`, `METADATA_SCHEMA` | old model (`data.py`) | removed; `AttributeError`/`ImportError` with an upgrade hint |
| `check_keys`, `try_eval`, `get_polars_type`, `count_dat_runs`, `count_coco_runs`, `monotone_per_run`, `process_header` | `data.py` helpers | removed (internal to the old model) |
| `COCO_KINDS`, `IohJson`, `IohScenario`, `load_ioh_json`, `parse_dat`, `parse_coco_dat`, `header_style`, `merge_files`, `parse_info` | leaked by `from .data import *` | not top-level; still in `iohinspector.io.*` (reader internals) |
| `np`, `pl`, `os`, `re`, `ast`, `glob`, `warnings`, `deepcopy`, `dataclass`, `field`, `partial`, `Callable`, `Iterable`, `version`, `PackageNotFoundError`, `hypervolume`, `igd_plus`, `epsilon_additive`, `epsilon_mult`, `filter_dominated`, `is_nondominated`, `get_reference_directions` | leaked imports | gone from the star import (`__all__`) |
| submodules `align`, `data`, `manager`, `io`, `format`, `metrics`, `plots`, `indicators`, `anytime`, `final`, `aocc`, `ecdf`, `eaf`, ... | leaked module names | packages stay importable; `data` becomes a stub that fails with a hint |
| `align_data`, `turbo_align` | group on `data_id` | group on `run_id` (`group_cols` default `("run_id",)`) |
| `aggregate_convergence`, `aggregate_running_time`, `get_data_ecdf`, `get_aocc` | `free_vars`, `return_as_pandas` | `by`; return polars |
| `get_trajectory` | `free_variables`, `return_as_pandas` | `by`; polars |
| `get_tournament_ratings` | `alg_vars`, `fid_vars`, `return_as_pandas` | `by` (was `alg_vars`), `fid_vars` kept (the problem key, not a grouping of results); polars |
| `get_discritized_eaf_single_objective`, `get_eaf_data`, `get_eaf_pareto_data`, `get_eaf_diff_data` | `return_as_pandas` | polars |
| `get_heatmap_single_run_data` | `return_as_pandas`, asserts one `data_id` | polars, one `run_id` |
| `get_attractor_network` | returns pandas (nodes, edges) | returns polars |
| `get_robustrank_over_time`, `get_robustrank_changes` | robustranking objects | unchanged |
| `get_sequence`, `normalize_objectives`, `add_normalized_objectives`, `transform_fval` | | unchanged |
| `add_indicator` | groups on `data_id` | groups on `run_id` |
| `HyperVolume`, `Epsilon`, `IGDPlus`, `R2`, `NonDominated`, `get_reference_set` | | unchanged; the dead duplicate `indicators.anytime.NonDominated` (shadowed by `final.NonDominated`) goes |
| `plot_single_function_fixed_target`, `plot_single_function_fixed_budget`, `plot_ecdf` | `free_vars`; return pandas | `by`; return polars |
| `plot_paretofronts_2d`, `plot_indicator_over_time` | `free_var: str` | `by: list` |
| `plot_tournament_ranking` | `alg_vars` | `by` |
| other `plot_*`, `BasePlotArgs`, `LinePlotArgs` | | return polars where they returned pandas |
| `metrics.multi_objective.get_pareto_front_2d`, `get_indicator_over_time_data` | `return_as_pandas` | polars |

### DataManager attributes

| Attribute | After I26 |
|---|---|
| `add_folder(path)` | kept; any source (file, folder, archive) in any reader format; options go to `convert` (`format`, `mapping` for CSV, `dataset_id`, `tags`, `tags_from_path`) |
| `add_folders(paths)` | kept |
| `add_dataset(path)` | new: a converted format v1 folder (`scan_dataset`) |
| `add_json`, `add_coco_info` | removed -> `add_folder(path)` |
| `add_data_set`, `extend_overview` | removed -> `add_folder` / `add_dataset` |
| `data_sets` | removed -> `datasets` (format v1 `StoredDataset`s) |
| `runs`, `descriptors`, `trace`, `positions` | new: the four tables of the selection, as LazyFrames |
| `overview` | kept; one row per run: runs columns + every descriptor (`evals` -> `evals_used`, `data_id` gone, `run_id` the stable hash) |
| `select(...)` | keyword-only: `function_ids`, `algorithms`, `dimensions`, `instances`, `run_ids`, and any runs column or descriptor as `key=[values]`; `data_ids` -> `run_ids`; `experiment_attributes` -> `key=[value]`; `data_attributes` removed |
| `select_indexes` | removed -> `select(run_ids=...)` |
| `functions`, `algorithms`, `dimensions`, `instances` | sorted tuples of ids / names / ints (were tuples of `Function` / `Algorithm` objects with repeats) |
| `experiment_attributes`, `data_attributes` | removed -> `descriptor_keys`; trace columns via `trace.collect_schema()` |
| `n_runs`, `any`, `__add__`, `load(...)` | kept; `load` builds the same wide frame from the tables |

## Inventory: call sites of the removed names (before I26)

`data_id` / `free_vars` (incl. `free_variables`, `free_var`, `alg_vars`) /
`return_as_pandas`, lines per file:

- src: `align.py` 10, `manager.py` 5, `data.py` 11, `indicators/__init__.py` 3,
  `metrics/` aocc 8, attractor_network 2, eaf 16, ecdf 7, fixed_budget 7,
  fixed_target 7, multi_objective 6, ranking 8, single_run 4, trajectory 6;
  `plots/` eaf 1, ecdf 9, fixed_budget 6, fixed_target 6, multi_objective 8,
  ranking 5, single_run 3.
- tests: `test_align.py` 24, `test_data.py` 12, `test_manager.py` 24 (incl.
  `data_sets`, `add_json`, ...), `io/conftest.py` 3, `io/test_ioh.py` 1,
  `io/test_ioh_defects.py` 1, `test_metrics/` aocc 7, attractor_network 1, eaf 24,
  ecdf 1, fixed_budget 11, fixed_target 12, multi_objective 4, ranking 2,
  single_run 5, trajectory 11; `test_plots/` attractor_network 1, eaf 4,
  multi_objective 3, ranking 2, single_run 1.
- docs: `examples/SO_Examples.ipynb` 10, `examples/MO_Examples.ipynb` 12,
  `FUNCTION_REFERENCE.md` 28.

## Decisions
- `DataManager.add_folder` reads any source through `convert.read_dataset`: it
  converts to a temporary folder and opens it with `scan_dataset`, so a source and a
  converted dataset are the same `StoredDataset` (one code path, the writer's). The
  temporary folder lives as long as the dataset object (`StoredDataset.files`).
- A manager is a list of datasets; `select` returns a manager over the same datasets
  with their four tables filtered on the selected `run_id`s.
- Adding a dataset whose `dataset_id` is already loaded fails; pass `dataset_id=`.
  (0.0.8's `add_json` warned and skipped; `add_folder` silently duplicated.)
- `load()` keeps `evaluations` as UInt64 as before (the indicators join on UInt64).
- `run_id` is the format's stable hash, so it is not 1, 2, ... per file any more.
  Two places used the old per-file number: the robustranking pairing (now a run number
  per algorithm in data order, `_indicator_table`, the same pairing for data from
  `load`) and moocore's set ids (now a dense rank of `run_id`, `eaf._points`).
- `get_discritized_eaf_single_objective` returns `eaf_target` as a column (it was the
  pandas index), and one column per evaluation value named by its value.
- `get_attractor_network` returns polars; a node's number is its row number.
- Plots take `by` (a list) and return the polars frame they drew; `_hue` gives seaborn
  the one column, or tuples for several (one `by` column now gives plain legend labels
  instead of 1-tuples).
- Removed stubs that only raised `NotImplementedError`
  (`plot_multi_function_fixed_target`, `plot_multi_function_fixed_budget`,
  `robustranking`, `stats_comparison`, `winnning_fraction_heatmap`), and the dead
  `indicators.anytime.NonDominated` (shadowed by `final.NonDominated`).
- Removed names fail through one table (`iohinspector/_upgrade.py`): a decorator for
  removed keyword arguments, and a `__getattr__` for removed module and class names.
- `fid_vars` of `get_tournament_ratings` stays: it names the problems, not a grouping
  of the output.

## Open questions / Blocked
- none yet
