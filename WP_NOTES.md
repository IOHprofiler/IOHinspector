# WP I02: Definitions audit
Branch: wp/I02-definitions-audit
Status: in progress

## Done
- step 0: research notes for step 1 recorded here (this commit). No pages yet.

## Next step
Write `docs/definitions/runtime.md` (step 1: ERT/aRT, PAR-k, success rate, mean,
median, sd, quantiles) from the findings below plus the cocopp report saved at
`/tmp/claude-1000/-home-jacob-code-ioh-web/f01f1b98-84c4-4877-a947-05ad604b602a/scratchpad/I02-cocopp-report.md`
(cocopp 2.9.1 source unpacked next to it under `cocopp-src/`; re-check every
line number before citing). Then steps 2 to 7 as listed in plan/iohinspector.md.

## Planned pages (one per parity-matrix row group)
- step 1 `runtime.md`: ERT/aRT, PAR-k, success rate/ratio, mean, median, sd, quantiles.
- step 2 `alignment.md` (hit rule, best-so-far, ties, maximisation, fixed-target
  alignment, runtime and value samples) and `fixed-budget.md` (fixed-budget
  summary, overview tables, carrying values forward, stopped runs).
- step 3 `targets.md` (seq_FV, seq_RT, get_ECDF_targets, get_target_dt, COCO grid)
  and `ecdf.md` (ECDF, aggregation, AUC, AOCC, multi-function ranks, the 0.67
  vs 0.90 worked example).
- step 4 `distributions.md` (histogram breaks, PMF, KDE/violin) and
  `parameters.md` (parameters, correlations, positions, constraint penalties;
  this row is in the parity matrix but not named in the I02 step list).
- step 5 `comparisons.md` (KS + Holm, win fractions, dominance graph) and
  `glicko2.md`.
- step 6 `portfolio.md` (Shapley), `cdp.md`, `eaf.md`, `dsc.md`.
- step 7 `README.md`: decision table, differences from IOHanalyzer, open questions.

## Findings so far (step 1, to be written up)
IOHanalyzer (source/IOHanalyzer, HEAD 20e14e4):
- RT matrix: `src/align.cc` `c_impute_running_time` (l.70-100): first row j with
  best-so-far `index[j] <= FV[i]` (min) / `>=` (max). Rows of the RT matrix are
  all distinct observed f values (`R/readFiles.R` 1594-1600 for v1plus JSON data,
  `align_running_time` 754-809 for old .dat). Target lookup in
  `get_RT_summary.DataSet` (`R/DataSet.R` 910-966) and `get_ERT.DataSet`
  (879-905): first RT-matrix row whose f value `<=` target (min) / `>=` (max).
- `SP()` (`R/stats.R` 12-25): success = RT not NA and RT <= max_runtime;
  failed runs charged `max(max_runtime)` (a vector is collapsed to its max), so
  every failed run is charged the largest budget in the set, not its own;
  ERT = rowSums / succ, so 0 successes gives Inf (x/0) or NaN (0/0 cannot occur
  since budgets > 0).
- `get_RT_summary.DataSet`: maxRT = max(attr maxRT) or user budget; mean,
  median, sd with na.rm (successful runs only, but runs with RT > budget are
  NOT removed from mean/median/sd/quantiles, only from ERT/ps); quantiles via
  `IOHanalyzer_env$D_quantile` = `quantile(type = 3, na.rm = T)`
  (`R/IOHanalyzer.R` 117-133) at probs 2,5,10,25,50,75,90,95,98 %
  (`R/IOHanalyzer.R` 71); median column uses `median()` (type-7 middle average),
  so `median` and `50%` can differ. PAR-k: NA and RT > maxRT replaced by
  `maxRT * IOHanalyzer.PAR_penalty` (default 1, set from Shiny input
  `ERTPlot.ParX`, `inst/shiny-server/server/ERTPlot.R:16`), mean over all runs.
- v1plus (IOHexperimenter JSON) reader: `maxRT = max(runtimes)` scalar over all
  runs (`R/readFiles.R:1638`), overriding per-run `evals` read from JSON
  (`R/readFiles.R` 112-113); `finalFV = min(FV)` even when maximising (1639).
- C_quantile (function values) = R default type 7 (`R/IOHanalyzer.R` 136-147).
- numpy: method='inverted_cdf'/'closest_observation'/'linear'/'median_unbiased'
  reproduce R types 1/3/7/8 exactly (checked with Rscript on random integer
  samples, n = 1..26, the 9 IOHanalyzer probs).
- `get_target_dt(which)` naming is inverted relative to `generate_data.*`:
  `generate_data.Aggr` (`R/DataSetList.R:1679`) passes its fixed-target
  `which='by_RT'` to `get_target_dt`, which then returns a budget
  (`get_target_RT`, `R/stats.R` 572-574) used as an f target. Same pattern in
  `R/stats.R:615` (Glicko-2) and `:912` (DSC). Check in steps 3 and 5.
- ECDF (`generate_data.ECDF`, `R/DataSetList.R` 1334-1440): strict `<` (min) /
  `>` (max) against targets, mean over runs, then mean over (function, dim)
  DataSets per algorithm (equal weight per pair). AUC: trapezoid over the 50
  grid points, divided by the upper x (not x_max - x_min) (1454-1515).
  Default ECDF targets: `get_ECDF_targets` (1083-1112): 10 log-spaced targets
  between min and max observed f per (function, dim), or 'bbob' 51 targets
  1e2..1e-8.

iohinspector 0.0.8 (worktree src/iohinspector):
- `metrics/fixed_target.py` `aggregate_running_time`: targets = 50 log-spaced
  between data min/max; ERT = sum(finite RT else eval_max) / successes with
  `eval_max = data["evaluations"].max()` over the whole frame passed in (all
  algorithms/functions); mean/median/std over all runs including inf (so inf/NaN
  when any run fails); PAR-10 = mean(finite RT else 10 * eval_max).
- `align.py` `align_data` fixed-target: cum_min/cum_max then `join_asof`
  backward (min) / forward (max) on f; minimisation picks the LAST row among
  rows with equal best-so-far (overestimates RT when rows repeat a value); this
  is masked by the loader's global `raw_y.diff() != 0` filter (`data.py:219`),
  a listed defect. Unreached targets filled with inf.
- `metrics/ecdf.py` `get_data_ecdf`: not a target ECDF; mean over runs of the
  log-scaled min-max normalised best-so-far (`transform_fval`,
  `metrics/utils.py` 167-197), bounds default to the aligned data min/max.
  Equivalent to an ECDF over a continuum of log-uniform targets. Likely the
  source of 0.67 vs 0.90 (to be shown in step 3).
- fixed-budget `align_data`: backward asof on evaluations, fill_null(inf):
  before the first logged evaluation a maximisation run gets +inf (best value).

## Decisions
- Parity matrix = the "Gap between IOHanalyzer and iohinspector" table in
  plan/site/plan.html section 8 (metric rows only; reader rows belong to I04-I08)
  together with the metric table in plan/iohinspector.md "Definitions audit (D6)".
- Scope kept to docs/definitions/ and WP_NOTES.md; no code, tests or pyproject.

## Open questions / Blocked
- Citations still to verify (the first verification sub-agent was lost to a
  usage limit). Verify with Crossref before citing.
