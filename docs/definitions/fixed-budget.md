# Fixed-budget summaries and overview tables

Status: **proposed**, for sign-off at checkpoint H1; the open questions were
answered by the D6 review (dvermetten, 2026-11-06).
Parity-matrix row: "Fixed-budget summary, overviews".
Implemented in: I11 (`iohinspector.metrics.aggregate_convergence` and
`get_overview`).
Sources: [IOHanalyzer], [cocopp], [iohinspector]; see [references](references.md).

The fixed-budget view asks which function value a run set has reached after b
evaluations [Doerr2018, Wang2022]. It uses the value at a budget V_i(b) from
[alignment](alignment.md).

## Notation

- Run set S with n runs, budgets B_i, first logged evaluation e_i1.
- At budget b: the **defined** runs D(b) = { i : b ≥ e_i1 }, with
  n_def(b) = |D(b)|; the **active** runs A(b) = { i : B_i ≥ b }, with
  n_active(b) = |A(b)|. Runs in D(b) but not in A(b) contribute their final
  value carried forward (AL-4).
- The values summarised at b are { V_i(b) : i ∈ D(b) }.

## Definitions

### Fixed-budget summary at budget b

Over the defined runs:

- `mean`, `median` (conventional sample median), `sd` (denominator
  n_def − 1; null when n_def < 2);
- quantiles, Hyndman–Fan **type 7** [Hyndman1996] (linear interpolation
  between order statistics, the R and numpy default), at 2, 5, 10, 25, 50, 75,
  90, 95 and 98 %;
- `best` and `worst` (min and max when minimising, the other way round when
  maximising);
- optional `geometric_mean` = exp(mean(log V)), defined only when every value
  is > 0; otherwise null;
- the counts `n_runs` = n, `n_defined` = n_def(b) and `n_active` = n_active(b).

Function values are continuous, so an interpolating quantile is appropriate;
runtimes are integers and use type 3 instead (see [runtime](runtime.md)).
Together these answer the plan's question on quantile types.

### Stopped runs: carry forward or simulated restarts

A run **stopped** at b when B_i < b (B_i is `evals_used` from the run
metadata; alignment flags such rows `stopped`, AL-9). A run with B_i ≥ b that
has not improved since an earlier evaluation did not stop: its value at b is
its best-so-far, and it counts in n_active. For stopped runs the summary has two options,
`stopped = "carry_forward"` (default) or `"restart"`:

- **carry forward**: V_i(b) is the run's final best value (as AL-4);
- **simulated restarts**, after the fixed-target restarts of [Hansen2010]: run
  i is continued by restarts. Each restart draws a run j uniformly with
  replacement from the same run set (seeded) and appends its trace after B_i
  evaluations; V_i(b) is the best value over the original run and every
  restart segment up to evaluation b, the last segment being cut at b. Runs
  with B_i ≥ b are left as they are. The number of restarted runs is reported
  next to `n_active`.

### Overview table per run set

Each run's **final value** is its best-so-far value at B_i (`best_y` in
`runs.parquet`).

| Column | Definition |
|---|---|
| `n_runs` | n |
| `budget` | max_i B_i |
| `best_reached` | best final value |
| `worst_reached` | worst final value |
| `mean_reached`, `median_reached` | mean and median of the final values |
| `n_best` | number of runs whose final value equals `best_reached` exactly (IOHanalyzer `succ`; not target-based) |
| `worst_recorded` | worst first logged value over all runs (the worst value of the best-so-far traces) |
| `min_runtime`, `max_runtime` | smallest first logged evaluation and largest evaluation at which any run improved (the min and max finite entries of IOHanalyzer's RT matrix) |

## How IOHanalyzer implements it

- `get_FV_summary.DataSet` (`R/DataSet.R` 1068–1127). Budgets are sorted
  (1076). If the largest budget exceeds the last FV-matrix row, the last row is
  duplicated at a larger evaluation count so stopped runs are carried forward
  (1080–1085). Then every NA in a run's column is replaced by that run's
  **minimum** (1087–1093) "to preserve monotonicity of mean". NA only occurs
  before a run's first row, and for minimisation the minimum is the run's
  best value, so a run that has not started yet is credited with its final
  result. This is the R-core finding "the fixed-budget summary fills missing
  values with the run minimum". Statistics are taken over all runs: `.mean`,
  `.median`, `.sd` and `C_quantile` (1099–1108), which is R's default type 7
  (`R/IOHanalyzer.R` 136–147). The `runs` column is the number of columns
  (1071, 1102).
- Geometric mean: `exp(mean(log(x)))` (1120–1125), so NaN for negative values
  and 0 when one value is 0. The column rename `setnames(dt, 'V2', ...)`
  (1124) assumes the appended column is called `V2`; this is the Shiny finding
  "the geometric mean aggregation never matches its column".
- `get_FV_sample.DataSet` (1134–1163): the value at the last row with
  evaluation ≤ b, NA before the first row, no carry-forward row added.
- Overviews: `get_FV_overview.DataSet` (767–797), `get_RT_overview.DataSet`
  (802–829) and `get_overview.DataSet` (834–874). They use the last FV-matrix
  row as the final values (770, 846) and `budget = max(attr(ds, 'maxRT'))`
  (771, 815, 838); for IOHexperimenter JSON data `maxRT` is the largest logged
  evaluation (see [runtime](runtime.md)). `succ` counts runs whose last value
  equals the best last value (782, 857). `worst recorded` is the worst value in
  the whole FV matrix (778, 853), which is the worst first value because the
  matrix holds best-so-far values. The RT overview's minimal and maximal
  runtime are the min and max of the RT matrix (816–817); `get_overview` calls
  the latter `max evals used` (867) although it is the last improvement, not
  the evaluations used.

## How cocopp implements it

cocopp has no fixed-budget summary table. Its fixed-budget plot
(`pprldistr.plotFVDistr`, `cocopp/pprldistr.py` 546–582) shows the ECDF of
Δf at budget B·dimension: the last row with evaluations ≤ the budget, so
stopped runs carry their last value; Δf ≤ 0 is replaced by the smallest
positive value for the log axis. Runs are not instance-balanced in this plot
(docstring, line 556); the lookup is at lines 564–573.

## How iohinspector 0.0.8 implements it

`aggregate_convergence` (`metrics/fixed_budget.py` 8–75):

- budgets: 50 log-spaced integers between the data's smallest and largest
  evaluation count (39–43);
- values from `align_data` (45–53), so carried forward, and `inf` before a
  run's first row (`inf` is the best value when maximising; see
  [alignment](alignment.md));
- `mean`, `min`, `max`, `median`, `std` and `geometric_mean` over all runs
  (54–61); any `inf` makes the mean `inf`; the geometric mean is computed even
  for values ≤ 0 (NaN or −inf); polars `median` is the conventional median and
  `std` uses ddof = 1;
- no quantiles, no run counts, no overview table.

## Differences

| Item | IOHanalyzer | iohinspector 0.0.8 | Proposed |
|---|---|---|---|
| Value before a run's first row | the run's minimum (wrong for minimisation) | `inf` | null, excluded, counted |
| Stopped runs | carried forward | carried forward | carried forward, `n_active` reported |
| Quantiles | type 7 | none | type 7 |
| Geometric mean | `exp(mean(log x))`, column name bug | always computed | only when all values > 0 |
| `budget` in overviews | max logged evaluation (JSON data) | – | max_i B_i from run metadata |
| `max evals used` | largest hitting time | – | renamed `max_runtime`; `budget` holds the evaluations used |

## Decisions

- **FB-1.** At budget b, statistics are over runs that have started
  (b ≥ e_i1). Runs that stopped before b are included with their carried-
  forward final value, and the table shows `n_active`. This answers the plan's
  question on runs that stopped early: the last value, not missing.
- **FB-2.** Function-value quantiles are Hyndman–Fan type 7 at the nine
  IOHanalyzer probabilities. Runtime quantiles stay type 3 (RT-5).
- **FB-3.** The geometric mean is reported only when every value is strictly
  positive.
- **FB-4.** Overview columns are as in the table above, with `budget` from run
  metadata and the IOHanalyzer `succ` column renamed `n_best`.
- **FB-5.** IOHanalyzer's run-minimum fill is not ported; golden tests name
  FB-1 where it changes a value.
- **FB-6.** Runs that stopped before b (B_i < b) are handled by one of two
  options: carry the final value forward (default, FB-1) or perform simulated
  restarts as defined above. A run that stopped is distinguished from a run
  that merely did not improve, using the `stopped` flag that alignment sets
  from `evals_used` (AL-9). Excluding stopped runs is not offered. Decided by
  the D6 review (dvermetten, 2026-11-06); mechanics: flag column `stopped`
  from evals_used plus `is_final` on the last logged row in monotonic load
  (user, 2026-11-12).

## Questions answered by the D6 review

- Q-FB-1 (exclude stopped runs, or carry forward): two options, carry forward
  or simulated restarts, with the run stopping told apart from the run not
  improving; alignment keeps the last evaluation at all times (AL-9). See FB-6.

Remaining points for H1: the restart procedure is the proposal above (draw
uniformly with replacement from the run set, one chain per run, seeded); the
review named the option but not the procedure. The default stays carry forward.

## Test tolerance

Counts and `best`/`worst` must match exactly. Means, sd, quantiles and the
geometric mean must agree to a relative error of 1e-12.
