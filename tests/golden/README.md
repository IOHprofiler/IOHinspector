# Golden references

Reference outputs from IOHanalyzer and cocopp on a few small fixtures (WP I03). Tests in later
work packages compare iohinspector with them. They are references, not truths: where
`docs/definitions/` decides that IOHanalyzer is wrong, the test asserts the agreed value and names
the difference. The goldens record what IOHanalyzer actually returns, defects included.

## Layout

| Path | Content |
|---|---|
| `r/Dockerfile` | `rocker/r-ver:4.3.3` (pinned by digest, CRAN snapshot 2024-04-23), IOHanalyzer at `caecada^` (81b8dba, reports 0.1.8.11), PlayerRatings 1.1-0, eaf 2.5, RVCompare 0.1.8, arrow 15.0.1 |
| `r/generate.R` | runs IOHanalyzer's R functions on every fixture |
| `fixtures/` | the input logs, 0.9 MB in total (below) |
| `iohanalyzer/<fixture>/<metric>.parquet` | one file per metric and fixture, zstd, written by `generate.R` |
| `iohanalyzer/versions.json` | package versions, seed, the R call behind every file, and the calls that failed |
| `cocopp/generate.py`, `cocopp/coco/` | cocopp 2.9.1 references for the COCO fixture and their `versions.json` |
| `make_fixtures.py` | regenerates the ioh 0.4.0 fixtures (fixed seeds) |
| `load.py` | `golden(fixture, metric)`, `golden_cocopp(name)`, `versions()`, `fixture_path(name)` |
| `test_golden_files.py` | self-consistency checks of the committed files |

## Fixtures

| Name | Source | Content |
|---|---|---|
| `ioh040_min` | `ioh==0.4.0`, BBOB | f1-f3, dimensions 5 and 10, algorithms `rs`, `es_small`, `es_large`, instances 1-5, parameter `sigma` |
| `ioh040_max` | `ioh==0.4.0`, PBO | OneMax and LeadingOnes, dimensions 16 and 32, `rs`, `rls`, `ea`, maximisation |
| `ioh040_cons` | `ioh==0.4.0` | sphere with a soft constraint, every evaluation logged with `violation` and `penalty` |
| `ioh040_pos` | `ioh==0.4.0` | positions logged as parameters `x0..x2` |
| `extdata` | IOHanalyzer `inst/extdata` | legacy IOHprofiler `.info` + `.dat`, three data sets with 10 runs each |
| `coco` | `tests/test_coco_data` | COCO BFGS on f1, 2-D, 15 instances |

## Regenerate

```sh
make golden          # builds ioh-golden:1 if missing, then IOHanalyzer and cocopp references
make golden-check    # regenerates and fails when any committed file changes (byte-identical)
make golden-fixtures # only when the fixtures themselves change
```

Docker needs the docker group; in a session without it use `sg docker -c 'make golden'`. The
IOHanalyzer source tree for the image is exported with `git archive caecada^` from
a clone of <https://github.com/IOHprofiler/IOHanalyzer> (cached in `r/cache/`; set `IOHANALYZER=<path>` to use a local
checkout instead) into `r/ctx/`. Both folders are git ignored. cocopp runs in a uv venv (`.venv`, from `requirements-golden.txt`).

Everything is deterministic: random steps use `set.seed(20240423)`, Parquet is written with zstd,
nothing records a time.

## Parity-matrix rows and the files that cover them

Row names follow `docs/definitions/README.md` (WP I02). The `<fixture>` prefix is one of the
fixtures above. A file is only present when the fixture allows the call (for example the
multi-algorithm metrics need at least two algorithms); `versions.json` names every call that failed.
`per pair` means one call per (function, dimension) pair, identified by `pair_f`, `pair_d`.

| Row | Files | IOHanalyzer function |
|---|---|---|
| Runtime and value samples | `RT_sample`, `FV_sample`, `maxRT`, `runtimes`, `funvals` | `get_RT_sample`, `get_FV_sample`, `get_maxRT`, `get_runtimes`, `get_funvals` |
| ERT, PAR-k, runtime summary | `ERT`, `RT_summary_par1`, `RT_summary_par10`, `RT_overview_par*`, `overview_par*` | `get_ERT`, `get_RT_summary`, `get_RT_overview`, `get_overview`, option `IOHanalyzer.PAR_penalty` 1 and 10 |
| Fixed-budget summary, overviews | `FV_summary`, `FV_summary_geom`, `FV_overview` | `get_FV_summary`, `get_FV_overview` |
| Target and budget grids | `targets_log_linear`, `targets_linear`, `targets_bbob`, `budgets_log`, `budgets_linear`, `target_dt_by_FV`, `target_dt_by_RT` | `get_ECDF_targets`, `seq_RT`, `get_target_dt` |
| ECDF and AUC | `ECDF_by_RT`, `ECDF_by_FV`, `AUC_by_RT`, `AUC_by_RT_raw`, `AUC_by_FV`, `ECDF_raw` | `generate_data.ECDF`, `generate_data.AUC`, `generate_data.ECDF_raw` |
| Multi-function aggregation and ranks | `ECDF_default_targets`, `Aggr_funcId`, `Aggr_DIM` | `generate_data.ECDF` on all pairs, `generate_data.Aggr` |
| Histograms, violin data | `hist_by_RT`, `hist_by_FV`, `PMF`, `single_function_by_*`; violins use the raw `RT_sample`, `FV_sample` | `generate_data.hist`, `generate_data.PMF`, `generate_data.Single_Function` |
| Parameters, correlations, positions | `PAR_summary_by_*`, `PAR_sample_by_*`, `Parameters_by_RT`, `Parameter_correlation`, `positions` (`ioh040_pos`) | `get_PAR_summary`, `get_PAR_sample`, `generate_data.Parameters`, `generate_data.Parameter_correlation`, `get_position_dsl` |
| Constraint penalties | `ioh040_cons/penalty_{none,add,absolute}_{RT_summary,FV_summary,FV_matrix}` | option `IOHanalyzer.Violation_Function` with `get_RT_summary`, `get_FV_summary`, `get_FV` |
| KS tests with Holm correction | `KS_by_FV`, `KS_by_RT` | `pairwise.test` |
| Win-fraction heatmap | `Heatmaps_by_FV`, `Heatmaps_by_RT`, `*_5targets` | `generate_data.Heatmaps` |
| Glicko-2 ranking | `glicko2_by_FV`, `glicko2_by_RT`, and the game lists `glicko2_games_by_*` | `glicko2_ranking` (PlayerRatings 1.1-0), seed 20240423, 50 rounds; the game list is the data frame handed to `PlayerRatings::glicko2` (I02 GL-3) |
| Shapley portfolio values | `shapley_log`, `shapley_lin`, `marg_contrib_ecdf` | `get_shapley_values`, `get_marg_contrib_ecdf` |
| Cumulative difference plot | `CDP_by_FV`, `CDP_by_RT` | `generate_data.CDP` (RVCompare 0.1.8), 200 bootstrap samples, seeded |
| EAF and EAF differences | `EAF`, `EAF_logx`, `ECDF_From_EAF`, `EAF_Difference`, `EAF_diff_Approximate` | `generate_data.EAF`, `generate_data.ECDF_From_EAF`, `generate_data.EAF_Difference`, `generate_data.EAF_diff_Approximate` (eaf 2.5) |
| DSC (partial) | none | see below |
| AOCC | none from IOHanalyzer | see below |
| cocopp (COCO fixture) | `cocopp/coco/{runtimes_51targets,art_51targets,art_table_targets,ecdf_51targets}` | `DataSet.detEvals`, `detERT`, `detSuccesses` |

Not produced, and why:

- **DSC.** `get_dsc_rank`, `get_dsc_omnibus` and `get_dsc_posthoc` post to the DSCtool web service
  (`ws.ijs.si`) with credentials; the container has neither, and I02 (DS-1) specifies a local DSC
  without the network. I18 builds its references from the papers.
- **AOCC.** IOHanalyzer has no AOCC function. I02 EC-5 defines it as the AUC of the continuum
  ECDF; the nearest IOHanalyzer references are `AUC_by_RT` and `AUC_by_RT_raw`.
- **cocopp ECDF** is the fraction of (target, run) pairs reached within a budget, computed from
  `detEvals`; cocopp's own plotting code is not run.

## Where IOHanalyzer's output is a known defect (I02)

The files hold IOHanalyzer's actual output. Tests that compare iohinspector with them must use the
agreed value from `docs/definitions/` for these:

| Files | Defect (page in `docs/definitions/`) |
|---|---|
| `RT_summary_*`, `ERT`, `*_overview_*` on JSON fixtures (`ioh040_*`) | Failed runs are charged the largest logged evaluation, not their budget (runtime, RT-1) |
| `ECDF_by_RT`, `ECDF_default_targets`, `Aggr_*`, `shapley_*` | Strict comparison: the best observed value is never reached, the ECDF can top out below 1 (ecdf, EC-1); Shapley ECDFs normalised by successful runs only, more than 10 algorithms truncate (portfolio, SH-1/SH-2) |
| `AUC_by_*` | AUC divides by the upper budget, not the integration range (ecdf, EC-4) |
| `target_dt_*` on `ioh040_max` | Default per-pair target under maximisation is an easy target (targets, TG-4) |
| `PAR_*`, `Parameters_*`, `Parameter_correlation` on JSON fixtures | Parameter-by-target reads wrong rows; by-budget uses the next logged row (parameters, PA-1, PA-5) |
| `penalty_absolute_*`, `penalty_add_*` | "Penalize Absolute" penalises feasible rows (parameters, PA-4) |
| `KS_*`, `Heatmaps_*` | KS tests drop failed runs and reverse direction under maximisation; win fractions use the mean of successes (comparisons, CM-2, CM-3, CM-5) |
| `CDP_by_FV_with_failures` (not produced) | `generate_data.CDP` fails when a run misses the target (cdp, CD-2); `CDP_by_FV` uses a target every run reaches |

Beyond the pages, calls that IOHanalyzer itself rejects on these fixtures are listed under
`skipped` in `iohanalyzer/versions.json` with the error. The current list:

- `PAR_summary_by_FV` and `PAR_sample_by_FV` fail for targets reached in the first logged row or never
  reached on the JSON fixtures (`subscript out of bounds`); those targets are left out per pair.
- `Parameters_by_FV` fails on `ioh040_min`, `ioh040_max` and `ioh040_pos` (same family).
- `EAF`, `EAF_logx`, `ECDF_From_EAF` fail on `extdata` because its function ids are strings.
- `positions` needs the attribute `contains_position`, which no reader sets at this commit;
  `ioh040_pos` therefore sets it by hand; without it `get_position_dsl` returns nothing.
