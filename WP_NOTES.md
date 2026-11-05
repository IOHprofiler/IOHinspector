# WP I02: Definitions audit
Branch: wp/I02-definitions-audit
Status: ready for review

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
  (`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<wt>/src venv1/bin/python`)
  (commit 6ec13be).
- step 4: `distributions.md` (DI-1..DI-3) and `parameters.md` (PA-1..PA-5;
  found IOHanalyzer's JSON parameter-by-target lookup broken: dcast without
  fun.aggregate gives counts, and row order mismatch; reproduced with
  scratchpad `ecdf_example/run4.R`) (commit cad37e1).
- step 5: `comparisons.md` (CM-1..CM-6; KS p-value methods compared in R
  4.3.3 and scipy 1.18.1) and `glicko2.md` (GL-1..GL-4; PlayerRatings 1.1-0
  installed in scratchpad `rlib`, checked on Glickman's example). Corrected
  the inverted-`which` list in targets.md (Glicko-2 and Heatmaps are fine)
  (commit bb66db9).
- step 6: `portfolio.md` (SH-1..SH-3), `cdp.md` (CD-1..CD-4; RVCompare 0.1.8
  source in scratchpad `rvcompare/`), `eaf.md` (EA-1..EA-4) and `dsc.md`
  (DS-1..DS-4, partial: papers' full text and the DSCtool service were not
  reachable; open questions listed for I18) (commit 90df7d2).
- step 7: `docs/definitions/README.md`: page index mapped to the parity
  matrix, answers to the plan's eight questions, a table of all 65 decisions
  (with whether each differs from IOHanalyzer), the IOHanalyzer defects found,
  19 open questions, tolerances and how the worked examples were computed.

## Next step
None for the agent. HUMAN: review the decisions in
`docs/definitions/README.md` and sign checkpoint H1. Q-DS-1 (DSC paper PDFs or
DSCtool API docs) blocks I18 step 1, not this WP.

## Pages
`docs/definitions/`: README, references, runtime, alignment, fixed-budget,
targets, ecdf, distributions, parameters, comparisons, glicko2, portfolio,
cdp, eaf, dsc. No code, tests, CI or pyproject changes.

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
- `which` conventions in IOHanalyzer: 'by_FV' = fixed target in
  get_target_dt, generate_data.Heatmaps, glicko2_ranking; 'by_RT' = fixed
  target in generate_data.ECDF, generate_data.Aggr, convert_to_dsc_compliant
  and get_dsc_rank. Inverted defaults only at `R/DataSetList.R:1679` and
  `R/stats.R:912` (fixed in targets.md in step 5).
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
- Not blocked. The 19 questions for the user are listed in
  `docs/definitions/README.md` ("Open questions for the user").
- Paywalled papers were not re-read; content claims are cross-checked against
  the authors' implementations instead (stated in references.md).
- `dsc.md` is a partial specification: the DSC papers' full text and the
  DSCtool service were not reachable, so the ranking algorithm is left as
  explicit questions for I18 (Q-DS-1).
- The inputs behind the plan's "0.67 vs 0.90" and "0.146 vs 0.226" numbers are
  not recorded; both gaps are re-derived from the definitions with worked
  examples (Q-EC-1).
