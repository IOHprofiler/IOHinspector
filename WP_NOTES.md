# WP I02: Definitions audit
Branch: wp/I02-definitions-audit
Status: in progress

## Done
- step 0: notes with runtime findings (commit ee4a9e6).
- step 1: `docs/definitions/runtime.md` (ERT/aRT, PAR-k, success rate, mean,
  median, sd, quantiles; decisions RT-1..RT-6) and
  `docs/definitions/references.md` (verified bibliography) (commit 1573fa4).
- step 2: `alignment.md` (hit rule, ties, best-so-far, samples; AL-1..AL-7)
  and `fixed-budget.md` (FV summary, overviews; FB-1..FB-5). Tie and
  edge-case behaviour of iohinspector `align_data`/`turbo_align` checked with
  a script on polars 1.44.2 and 2.1.0 (scratchpad `ties.py`) (commit 4d29aee).
- step 3: `targets.md` (TG-1..TG-6) and `ecdf.md` (EC-1..EC-8, worked example
  of 0.67 vs 0.90). Worked example computed with IOHanalyzer's own R functions
  sourced from source/IOHanalyzer (scratchpad `ecdf_example/ioha.R`: magrittr
  and rjson in scratchpad `rlib`, `align.cc` via Rcpp::sourceCpp, a dplyr
  `mutate` shim) and with iohinspector from the worktree
  (`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<wt>/src venv1/bin/python`).

## Next step
Step 4: write `docs/definitions/distributions.md` (histogram breaks, PMF,
KDE/violin: IOHanalyzer `generate_data.hist` and `generate_data.PMF`,
`R/DataSetList.R` 1232-1333, plot code in `R/plotDataSetList.R`, Shiny
`RTPMF.R`, `FCEPDF.R`) and `docs/definitions/parameters.md` (parameters by
target/budget, `get_PAR_summary`/`get_PAR_sample` `R/DataSet.R` 1178-1318,
`generate_data.Parameters` and `Parameter_correlation` `R/DataSetList.R`
1577-1667, positions `get_position_dsl` 1802-1854, constraint penalties via
option `IOHanalyzer.Violation_Function` and `FV_raw_mat`; Shiny `RT_Par.R`,
`FV_PAR.R`, `parrallel_coord.R`). The R harness in the scratchpad can run
these functions if needed.

## Planned pages (one per parity-matrix row group)
- step 4 `distributions.md` (histogram breaks, PMF, KDE/violin) and
  `parameters.md` (parameters, correlations, positions, constraint penalties;
  this row is in the parity matrix but not named in the I02 step list).
- step 5 `comparisons.md` (KS + Holm, win fractions, dominance graph) and
  `glicko2.md`.
- step 6 `portfolio.md` (Shapley), `cdp.md`, `eaf.md`, `dsc.md`.
- step 7 `README.md`: decision table, differences from IOHanalyzer, open
  questions. Collect every decision ID (RT-*, AL-*, FB-*, ...) and open
  question (Q-*) from the pages.

## Working material
- cocopp report (model output, line numbers spot-checked for the items cited in
  runtime.md):
  `/tmp/claude-1000/-home-jacob-code-ioh-web/f01f1b98-84c4-4877-a947-05ad604b602a/scratchpad/I02-cocopp-report.md`;
  cocopp 2.9.1 source under `.../scratchpad/cocopp-src/cocopp-2.9.1/src/cocopp`.
  If the scratchpad is gone: `pip download cocopp==2.9.1 --no-deps --no-binary :all:`
  into an empty folder and read it without importing it.
- Scratch venv with polars 2.1, numpy, scipy: `.../scratchpad/venv`.
- Citation lookups: Crossref `api.crossref.org/works?query.bibliographic=`,
  OpenAlex, arXiv API. arxiv.org PDF download is blocked in this environment.

## Findings for later steps
- `get_target_dt(which)` naming is inverted relative to `generate_data.*`:
  `generate_data.Aggr` (`R/DataSetList.R:1679`) passes its fixed-target
  `which='by_RT'` to `get_target_dt`, which then returns a budget
  (`get_target_RT`, `R/stats.R` 572-574) used as an f target. Same pattern in
  `R/stats.R:615` (Glicko-2) and `:912` (DSC). Shiny pages pass explicit
  values: `RT_DSCinterface.R:200` passes "by_RT" (bug B1).
- IOHanalyzer ECDF (`generate_data.ECDF`, `R/DataSetList.R` 1334-1440): strict
  `<` (min) / `>` (max) against targets, mean over runs, then mean over
  (function, dim) DataSets per algorithm (equal weight per pair). x grid: 50
  points of `seq_RT` over all runtimes. AUC (1454-1515): trapezoid over the
  grid points, divided by the upper x (not x_max - x_min). Default targets:
  `get_ECDF_targets` (1083-1112): 10 log-spaced targets between min and max
  observed f per (function, dim), or 'bbob' 51 targets 1e2..1e-8.
  `seq_FV`/`seq_RT`: `R/stats.R` 188-363.
- iohinspector `align_data` fixed-target (`align.py` 36-61): cum_min/cum_max,
  then `join_asof` backward (min) / forward (max) on f. Minimisation picks the
  LAST row among rows with equal best-so-far (overestimates T when rows repeat
  a value, and polars sort is not stable by default); masked by the loader's
  global `raw_y.diff() != 0` filter (`data.py:219`). Unreached -> inf.
  Fixed-budget: backward asof on evaluations, `fill_null(np.inf)`: before the
  first logged evaluation a maximisation run gets +inf (its best value).
- iohinspector `get_data_ecdf` (`metrics/ecdf.py`): not a target ECDF; mean
  over runs of the log-scaled, min-max normalised best-so-far
  (`transform_fval`, `metrics/utils.py` 167-197), bounds default to the
  aligned data min/max (lb <= 0 reset to 1e-8). This is an ECDF over a
  continuum of log-uniform targets; likely the 0.67 vs 0.90 cause (show it).
- IOHanalyzer FV summary (`R/DataSet.R` 1068-1127): NA filled with the run's
  min (1087-1093; wrong for minimisation), stopped runs carried forward by a
  duplicated last row (1080-1085), quantiles type 7 (C_quantile).
- cocopp: hit rule `<=` with rel. tolerance 1e-9 (`readalign.py` 249, 252,
  545-546); targets rounded down to its 5-per-decade grid; ECDF weights each
  (function, target) pair equally with simulated restarts (report section 4);
  fixed budget carries the last value forward (report section 7).

## Decisions
- Parity matrix = the "Gap between IOHanalyzer and iohinspector" table in
  plan/site/plan.html section 8 (metric rows only; reader rows belong to
  I04-I08) together with the metric table in plan/iohinspector.md
  "Definitions audit (D6)".
- One shared bibliography, `references.md`, cited by key from every page.
- Each page has: notation, definitions with citations, IOHanalyzer, cocopp
  (where it applies), iohinspector 0.0.8, differences table, proposed
  decisions with IDs, open questions, test tolerance.
- Scope kept to docs/definitions/ and WP_NOTES.md; no code, tests or pyproject.

## Open questions / Blocked
- Paywalled papers were not re-read; their content claims are cross-checked
  against the authors' implementations instead (stated in references.md).
