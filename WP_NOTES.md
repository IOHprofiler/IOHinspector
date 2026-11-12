# WP I29: Pre-release review before 0.2.0
Branch: wp/I29-prerelease-review
Status: ready for review

Review of `align.py`, `targets.py` and `metrics/` (I10 to I14) against
`docs/definitions/` (alignment, runtime, fixed-budget, ecdf, targets,
distributions, parameters). Every golden test passes unchanged. The COCO reader
was not touched.

## Fixed (each with a test that failed before; CHANGELOG "Fixed")
- `align_data` chose the fixed-budget view by the column name `"evaluations"`:
  every metric with another `eval_var` (aggregate_convergence, value and
  parameter samples, trajectory, EAF, get_data_ecdf) returned wrong values. New
  keyword-only `eval_var`.
- Runs whose `by` value is null (a descriptor some runs lack) were never matched
  by `join_asof(by=group_cols)`: get_ecdf, get_ecdf_auc, the samples,
  get_data_ecdf and get_trajectory treated them as reaching nothing. align_data
  now joins on one key per run (row index of the runs, `nulls_equal`).
- A target or budget given twice counted every run twice (`n_runs`, statistics).
- `by=[]` failed in get_overview, get_ecdf and get_ecdf_auc.
- align_data at a target took the other columns (a parameter) from the last of
  the rows tied on evaluation count and best-so-far value, not the first (AL-3).
- get_runtime_sample / get_value_sample carried every other data column, null
  for a failed run; they now return `run_id`, `by`, target/budget and value.

## Merged (DRY)
- Fixed-target alignment keeps each run's first row of every best-so-far value
  (`is_first_distinct`) instead of a two-key sort; parameters' own copy of that
  filter is gone. Runtime and value samples are lookups of get_parameter_sample.
- `describe` (n_runs, n_defined, mean, median, sd, type 7 quantiles) is the one
  summary of aggregate_convergence and get_parameter_summary; `custom` is the one
  custom_op column (runtime and fixed-budget summaries).
- The runtime summary computes mean, median and the null quantiles in its
  aggregation; no post-processing pass.
- ECDF hits align by run only; the hit row carries the group columns.

## Removed / simplified
- The `pair` option (get_ecdf, get_ecdf_auc, get_ecdf_targets, get_target_dt,
  get_rank_table): pairs are always (function_id, dimension) by the definitions.
- Smaller get_ecdf_targets, get_target_dt, get_parameter_summary, histogram
  breaks, violin rows, legacy normalize_objectives (duplicated branches).

## Public API (all new since 0.1.0, so no break of a release)
- Keyword-only options: seq_fv/seq_rt (`log`, `lower`, `upper`),
  get_ecdf_targets (`n`, `fval_var`), get_overview(data, by, *, eval_var,
  fval_var, maximization), get_histogram/get_violin (`view`, `bins`),
  get_logged_rows (`eval_var`).
- Histogram and violin column `n` is `n_defined`, as in the summaries.

## Tests added
eval_var under any name (7 metrics), repeated targets, null group columns
(align_data and 6 metrics), by=[] (7 metrics), AL-3 for carried columns, sample
columns, maximisation mirrors minimisation (fixed budget with restarts,
overview), null quantiles without success when include_unfinished=False.

## Line counts (lines / code lines)
Code lines: lines with a token other than a string or comment (Python tokenize);
this reproduces the plan's baseline exactly.

| | before (16e84e0) | after |
|---|---|---|
| src/iohinspector | 12,546 / 8,433 | 12,530 / 8,417 |
| metrics/ | 2,165 / 1,397 | 2,150 / 1,382 |
| tests | 10,176 / 7,929 | 10,315 / 8,041 |
| align.py | 87 / 48 | 94 / 53 |
| targets.py | 145 / 99 | 137 / 93 |
| metrics/fixed_target.py | 133 / 86 | 120 / 75 |
| metrics/fixed_budget.py | 226 / 160 | 220 / 154 |
| metrics/ecdf.py | 362 / 224 | 366 / 224 |
| metrics/distributions.py | 241 / 183 | 239 / 183 |
| metrics/parameters.py | 201 / 118 | 186 / 101 |
| metrics/utils.py | 239 / 137 | 242 / 142 |
| metrics/eaf.py | 162 / 90 | 175 / 103 |
| metrics/trajectory.py | 47 / 28 | 48 / 29 |

## Checks
- `pytest -q`: 927 passed, 3 skipped (polars 2.1.0). Polars 1.44.2: 926 passed;
  `test_verify_memory_does_not_grow_with_the_data` fails locally there at
  16e84e0 too (verify, not in scope).
- ruff (123 findings) and pyright (131) totals equal to 16e84e0; pylint on the
  touched files no new finding; lizard: no new warning.
- `scripts/bench_ert.py` full size: same time before and after (5-7 s on this
  machine, run-to-run noise).

## Next step
Controller: push, open the draft PR, request dvermetten (D6).

## Open questions
- alignment.md ("Runtime and value samples") says a runtime sample has null
  where the run never reached t; get_runtime_sample gives `inf`, as align_data
  and the runtime summary (RT-5 counts a failure as infinity). Keep `inf` and fix
  the page, or return null?
- get_data_ecdf and get_aocc stay as the 0.0.8 legacy functions (EC-6, I12);
  plot_ecdf still uses get_data_ecdf. Deprecate them in 0.2.0 in favour of
  get_ecdf(continuous=True) and get_ecdf_auc(continuous=True)?
