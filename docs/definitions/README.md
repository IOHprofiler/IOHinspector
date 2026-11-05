# Metric definitions

This folder holds the written definition of every metric iohinspector
computes for the IOHanalyzer website (work package I02, decision D6 of the
rebuild plan). iohinspector implements these definitions. Where IOHanalyzer
differs, iohinspector follows the definition and the test names the
difference. Nothing here assumes IOHanalyzer is correct.

Status: **proposed**. Every decision below needs sign-off at checkpoint H1.

Each page gives the definition with its literature source, how IOHanalyzer
implements it (file:line in `source/IOHanalyzer` at 20e14e4), how cocopp
implements it where that applies (cocopp 2.9.1), how iohinspector 0.0.8
implements it, the differences, the proposed decisions, open questions and
the test tolerance. Citations are in [references](references.md); each was
checked against Crossref, arXiv, OpenAlex or the publisher.

## Pages and parity-matrix rows

| Parity-matrix row | Page | WP |
|---|---|---|
| Runtime and value samples | [alignment](alignment.md) | I10 |
| ERT, PAR-k, runtime summary with quantiles | [runtime](runtime.md) | I11 |
| Fixed-budget summary, overviews | [fixed-budget](fixed-budget.md) | I11 |
| Target and budget grid generation | [targets](targets.md) | I10 |
| ECDF and AUC as in IOHanalyzer | [ecdf](ecdf.md) | I12 |
| Multi-function aggregation and ranks | [ecdf](ecdf.md) | I12 |
| Histograms, violin data | [distributions](distributions.md) | I13 |
| Parameters, correlations, positions | [parameters](parameters.md) | I14 |
| Constraint penalties | [parameters](parameters.md) | I14 |
| KS tests with Holm correction | [comparisons](comparisons.md) | I15 |
| Win-fraction heatmap | [comparisons](comparisons.md) | I15 |
| Glicko-2 ranking | [glicko2](glicko2.md) | I15 |
| Shapley portfolio values | [portfolio](portfolio.md) | I16 |
| Cumulative difference plot | [cdp](cdp.md) | I16 |
| EAF and EAF differences | [eaf](eaf.md) | I17 |
| DSC | [dsc](dsc.md) (partial) | I18 |
| AOCC (from the "New" row) | [ecdf](ecdf.md) | I12 |

The parity matrix is the metric part of the "Gap between IOHanalyzer and
iohinspector" table in the rebuild plan (section 8) together with the metric
table of `plan/iohinspector.md` ("Definitions audit (D6)"). Its reader rows
belong to I04–I08. Multi-objective indicators, robust ranking and attractor
networks are after launch and not covered.

## Answers to the plan's questions

| Question | Answer | Decision |
|---|---|---|
| Is a target hit by f < t or f ≤ t, and under maximisation? | f ≤ t when minimising, f ≥ t when maximising, everywhere including the ECDF. | AL-1, EC-1 |
| Which budget does ERT charge a failed run? | Its own evaluations from run metadata, capped by a user budget. | RT-1 |
| What is ERT when no run succeeds? | +∞ (null when there are no runs). | RT-2 |
| Which quantile type for runtimes and for function values? | Type 3 for runtimes (over successful runs), type 7 for function values. | RT-5, FB-2 |
| Does an aggregated ECDF weight each (function, dimension) pair or each target equally? | Each pair equally, then each target and run within it; the same as per (pair, target) when every pair has the same number of targets. | EC-2 |
| How are targets generated? | Log-spaced by default, 10 per (function, dimension) from its observed range; linear and the COCO 51-target grid 10^(2 − k/5), k = 0..50, are available. | TG-1, TG-2 |
| Fixed-budget value of a run that stopped before the budget? | Its final best value, carried forward; the table reports how many runs were still active. | AL-4, FB-1 |
| Do 0.67 and 0.90 differ by definition or by bug? | Both. Definitions: 10 discrete targets with a strict comparison against a continuum of normalised targets; the strict comparison caps IOHanalyzer's default ECDF at 0.90 when every run ends at the best observed value. Bugs in iohinspector: data-derived, shared normalisation bounds, dropped final rows and a truncated budget grid. Worked example in [ecdf](ecdf.md). | EC-1, EC-5, EC-6 |

## Decisions for sign-off

"Differs" means the agreed value differs from IOHanalyzer on some inputs;
golden tests name the decision there.

### Alignment ([alignment](alignment.md))

| ID | Decision | Differs |
|---|---|---|
| AL-1 | Inclusive hit rule (≤ / ≥) in every metric | ECDF only |
| AL-2 | Exact float64 comparison; targets not snapped to a grid | no (cocopp: yes) |
| AL-3 | Ties resolve to the first row that reaches the target | no |
| AL-4 | Stopped runs carry their final best value forward | no |
| AL-5 | Value before a run's first row is null | yes |
| AL-6 | Best-so-far always computed from raw rows | no |
| AL-7 | COCO targets apply to Δf = f − f_opt | no |

### Runtime ([runtime](runtime.md))

| ID | Decision | Differs |
|---|---|---|
| RT-1 | Failed run charged its own evaluations, min(B_i, C) | yes |
| RT-2 | ERT +∞ with no success, null with no runs | no |
| RT-3 | PAR-k: one cutoff per run set (C or max B_i); k a parameter, Python default 10, website default 1 | JSON data only |
| RT-4 | mean, median, sd over successful runs, with n and n_success | with a budget cap only |
| RT-5 | Runtime quantiles type 3 over successful runs | no |
| RT-6 | Runs pooled over instances, no balancing | no (cocopp: yes) |

### Fixed budget ([fixed-budget](fixed-budget.md))

| ID | Decision | Differs |
|---|---|---|
| FB-1 | Statistics over started runs; stopped runs carried forward; `n_active` reported | yes (no run-minimum fill) |
| FB-2 | Function-value quantiles type 7 | no |
| FB-3 | Geometric mean only when all values > 0 | yes |
| FB-4 | Overview columns with budget from run metadata; `succ` renamed `n_best` | JSON data only |
| FB-5 | Run-minimum fill not ported | yes |

### Targets ([targets](targets.md))

| ID | Decision | Differs |
|---|---|---|
| TG-1 | Generators log-linear (default 10 per pair), linear, bbob 51, custom | no |
| TG-2 | Ranges from observed values per (function, dimension) | no |
| TG-3 | Log spacing with min ≤ 0 starts at the smallest positive value, not 1e-12 | yes |
| TG-4 | Default single target mirrored under maximisation (98 % quantile) | yes, maximisation |
| TG-5 | Budget grids rounded, never truncated, include the maximum | no |
| TG-6 | Explicit `perspective` argument; inverted `which` defaults and bug B1 not ported | yes (defaults) |

### ECDF, AUC, AOCC, ranks ([ecdf](ecdf.md))

| ID | Decision | Differs |
|---|---|---|
| EC-1 | Target-based ECDF, inclusive in f and budget | yes |
| EC-2 | Equal weight per (function, dimension) pair | no |
| EC-3 | Exact step function, sampled only for display | yes (grid) |
| EC-4 | AUC as exact normalised integral | yes |
| EC-5 | AOCC on best-so-far with fixed bounds per pair; equals the continuum-ECDF AUC | – (new) |
| EC-6 | Current `get_data_ecdf` kept under a clear name, with EC-5 bounds | – |
| EC-7 | No simulated restarts | no (cocopp: yes) |
| EC-8 | Rank tables per pair, average ties, missing last | no |

### Distributions ([distributions](distributions.md))

| ID | Decision | Differs |
|---|---|---|
| DI-1 | Histogram breaks as R: `nclass.FD`, `pretty`, right-closed | no |
| DI-2 | Violin: Gaussian KDE, Silverman bandwidth, hard span (as plotly.js) | no (ggplot backend: yes) |
| DI-3 | Runtime distributions over successful runs, failures counted | no |

### Parameters and constraints ([parameters](parameters.md))

| ID | Decision | Differs |
|---|---|---|
| PA-1 | Parameter on the hitting row (target) or the last row ≤ b (budget) | yes |
| PA-2 | Parameter summaries with type 7 quantiles | no |
| PA-3 | Correlation views return raw matched rows | no |
| PA-4 | Penalty before best-so-far; "absolute" only on infeasible rows; penalties always worsen | yes |
| PA-5 | JSON parameter-by-target lookup not ported | yes |

### Comparisons ([comparisons](comparisons.md), [glicko2](glicko2.md))

| ID | Decision | Differs |
|---|---|---|
| CM-1 | One-sided KS with R's asymptotic p-value; exact as option | no |
| CM-2 | Failed runs count as +∞ | yes |
| CM-3 | Fixed-budget orientation follows maximisation | yes, maximisation |
| CM-4 | Holm over all ordered pairs; α = 0.01; heat map and graph | no |
| CM-5 | Win fractions on ERT (fixed target) or mean (fixed budget), equal weight per pair | yes |
| CM-6 | Simulated-restart bootstrap not ported unless asked | yes |
| GL-1 | Tournament: one game per round, pair and algorithm pair; failures lose | close (sampling scheme) |
| GL-2 | PlayerRatings 1.1-0 Glicko-2 update, init (1500, 350, 0.06), τ = 1.2 | no |
| GL-3 | Seeded; golden tests use recorded game lists | yes (seed) |
| GL-4 | Elo function kept under its own name | – |

### Portfolio, CDP, EAF, DSC

| ID | Decision | Differs |
|---|---|---|
| SH-1 | Portfolio value: summed AUC of the virtual-best ECDF over all runs | yes |
| SH-2 | Exact Shapley up to 15 algorithms, seeded stratified sampling above | yes |
| SH-3 | Normalised by v(N), sum 1 | no |
| CD-1 | RVCompare 0.1.8 bootstrap estimator, seeded | no |
| CD-2 | Failed runs +∞ | yes |
| CD-3 | Unequal sample sizes rejected | yes |
| CD-4 | ε = 1e-80 | no |
| EA-1 | EAF on best-so-far, levels 0–100 % in steps of 10 | no |
| EA-2 | EAF differences exact and on the 50 × 50 log grid | no |
| EA-3 | Multi-function EAF on normalised quality | yes |
| EA-4 | `xmax = Min` bug not ported | yes |
| DS-1 | Local DSC, no network | – |
| DS-2 | Inputs as IOHanalyzer's client, budgets from run metadata, B1 not ported | yes |
| DS-3 | NA options PAR-1, PAR-10 (default), ERT, Remove-na | no |
| DS-4 | Ranking, omnibus, post-hoc specified in I18 from papers and recordings | – |

## IOHanalyzer defects found in this audit

Beyond those already listed in the rebuild plan. Each is described, with
file:line, on the page named.

- JSON data: every failed run is charged the largest *logged* evaluation of
  the set instead of its budget ([runtime](runtime.md)).
- The ECDF's best observed value is itself a target and is never reached under
  the strict comparison, so the default ECDF tops out at 0.90 ([ecdf](ecdf.md)).
- AUC divides by the upper budget, not the integration range ([ecdf](ecdf.md)).
- Default per-pair target under maximisation is an easy target
  ([targets](targets.md)).
- Parameter-by-target on JSON data reads the wrong rows and, with repeated f
  values, counts instead of values ([parameters](parameters.md)).
- Parameter-by-budget uses the next logged row, values use the previous one
  ([parameters](parameters.md)).
- "Penalize Absolute" penalises feasible rows; penalties improve values under
  maximisation ([parameters](parameters.md)).
- KS tests drop failed runs by default and reverse direction under
  maximisation; win fractions use the mean of successes and drop matrices with
  several dimensions ([comparisons](comparisons.md)).
- Shapley ECDFs are normalised by successful runs only; more than 10
  algorithms truncate the estimate ([portfolio](portfolio.md)).
- The Shiny CDP calls pass an argument `generate_data.CDP` does not accept
  ([cdp](cdp.md)).
- DSC PAR-1/PAR-10 leave NA in place for JSON data ([dsc](dsc.md)).

## Open questions for the user

| ID | Question | Page |
|---|---|---|
| Q-AL-1 | Use cocopp's 1e-9 relative tolerance for COCO targets? | [alignment](alignment.md) |
| Q-RT-1 | Should Python and website defaults for PAR-k's k agree? | [runtime](runtime.md) |
| Q-RT-2 | Also offer unconditional runtime quantiles (failures as ∞)? | [runtime](runtime.md) |
| Q-FB-1 | Exclude stopped runs at b instead of carrying them forward? | [fixed-budget](fixed-budget.md) |
| Q-TG-1 | Smallest positive value instead of the 1e-12 floor (TG-3)? | [targets](targets.md) |
| Q-TG-2 | Offer cocopp's runlength-based targets after launch? | [targets](targets.md) |
| Q-EC-1 | Which data gave 0.67 and 0.90? | [ecdf](ecdf.md) |
| Q-EC-2 | Aggregate over each algorithm's own pairs, or only common pairs? | [ecdf](ecdf.md) |
| Q-DI-1 | Per-algorithm or equal histogram bins by default? | [distributions](distributions.md) |
| Q-PA-1 | Is `violation` always ≥ 0 with 0 = feasible? | [parameters](parameters.md) |
| Q-CM-1 | Asymptotic or exact KS p-value by default? | [comparisons](comparisons.md) |
| Q-CM-2 | Keep the simulated-restart bootstrap option? | [comparisons](comparisons.md) |
| Q-CM-3 | Accept ERT-based win fractions (CM-5)? | [comparisons](comparisons.md) |
| Q-GL-1 | Keep τ = 1.2 or use e.g. 0.5? | [glicko2](glicko2.md) |
| Q-SH-1 | Is 15 algorithms the right limit for exact Shapley values? | [portfolio](portfolio.md) |
| Q-CD-1 | Subsample unequal run counts for the CDP instead of refusing? | [cdp](cdp.md) |
| Q-EA-1 | Default bounds for the multi-function EAF (observed or 1e-8..1e2)? | [eaf](eaf.md) |
| Q-DS-1 | Provide the DSC papers or DSCtool API docs for I18 (blocking for I18) | [dsc](dsc.md) |
| Q-DS-2 | Unify the two PAR-k conventions (DSC and summary)? | [dsc](dsc.md) |

## Tolerances

Counts, hitting times, ranks and test decisions match exactly. Means, rates,
ERT, AUC, AOCC and closed-form p-values agree to a relative error of 1e-12.
Exceptions are stated per page: Glicko-2 volatility (1e-5, minimiser
tolerance), log-spaced targets (1e-15), and bootstrap-based outputs, which are
compared as distributions with a fixed seed.

## How the worked examples were computed

IOHanalyzer's own R functions were sourced from the read-only reference copy
(R 4.3.3, data.table, magrittr, rjson, `src/align.cc` compiled with
`Rcpp::sourceCpp`, a three-line stand-in for `dplyr::mutate`), without
installing the package. iohinspector 0.0.8 ran from the worktree on polars
1.44.2. PlayerRatings 1.1-0 and RVCompare 0.1.8 came from CRAN source
tarballs. I03's container replaces this setup with pinned versions.
