# Metric definitions

This folder holds the written definition of every metric iohinspector
computes for the IOHanalyzer website (work package I02, decision D6 of the
rebuild plan). iohinspector implements these definitions. Where IOHanalyzer
differs, iohinspector follows the definition and the test names the
difference. Nothing here assumes IOHanalyzer is correct.

Status: **proposed**. Every decision below needs sign-off at checkpoint H1.
The open questions were answered by the D6 review (Diederick Vermetten,
dvermetten, 2026-11-06, on IOHinspector#16); the decisions below contain those
answers, and the defaults chosen where an answer left a detail open are listed
under "Points to confirm at H1".

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
| DSC | [dsc](dsc.md) (DQ-1 to DQ-21 settled from dsc-core or decided) | I18 |
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
| Which quantile type for runtimes and for function values? | Type 3 for runtimes (over all runs, unfinished as ∞, by default), type 7 for function values. | RT-5, FB-2 |
| Does an aggregated ECDF weight each (function, dimension) pair or each target equally? | Each pair equally, then each target and run within it; the same as per (pair, target) when every pair has the same number of targets. Only pairs every compared algorithm has, with a warning. | EC-2 |
| How are targets generated? | Log-spaced by default, 10 per (function, dimension) from its observed range; linear and the COCO 51-target grid 10^(2 − k/5), k = 0..50, are available. | TG-1, TG-2, TG-3 |
| Fixed-budget value of a run that stopped before the budget? | Its final best value, carried forward (default), or simulated restarts as an option; the table reports how many runs were still active. | AL-4, FB-1, FB-6 |
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
| AL-8 | No tolerance on the hit rule for COCO-derived data | no (cocopp: yes) |
| AL-9 | Fixed-budget rows flag `stopped` (b > `evals_used`); monotonic load marks the last logged row `is_final` | – |

### Runtime ([runtime](runtime.md))

| ID | Decision | Differs |
|---|---|---|
| RT-1 | Failed run charged its own evaluations, min(B_i, C) | yes |
| RT-2 | ERT +∞ with no success, null with no runs | no |
| RT-3 | PAR-k: one cutoff per run set (C or max B_i); k a parameter, default 10 in Python and on the website | website default (1 → 10), JSON data |
| RT-4 | mean, median, sd over successful runs, with n and n_success | with a budget cap only |
| RT-5 | Runtime quantiles type 3 over all runs, unfinished as +∞; successful-only option | yes |
| RT-6 | Runs pooled over instances, no balancing | no (cocopp: yes) |

### Fixed budget ([fixed-budget](fixed-budget.md))

| ID | Decision | Differs |
|---|---|---|
| FB-1 | Statistics over started runs; stopped runs carried forward; `n_active` reported | yes (no run-minimum fill) |
| FB-2 | Function-value quantiles type 7 | no |
| FB-3 | Geometric mean only when all values > 0 | yes |
| FB-4 | Overview columns with budget from run metadata; `succ` renamed `n_best` | JSON data only |
| FB-5 | Run-minimum fill not ported | yes |
| FB-6 | Stopped runs: carried forward (default) or simulated restarts; stopping told apart from not improving | option only |

### Targets ([targets](targets.md))

| ID | Decision | Differs |
|---|---|---|
| TG-1 | Generators log-linear (default 10 per pair), linear, bbob 51, custom | no |
| TG-2 | Ranges from observed values per (function, dimension) | no |
| TG-3 | Log range from the data: min ≤ 0 starts at the smallest positive value; 1e-12 only when there is no other information | yes |
| TG-4 | Default single target mirrored under maximisation (98 % quantile) | yes, maximisation |
| TG-5 | Budget grids rounded, never truncated, include the maximum | no |
| TG-6 | Explicit `perspective` argument; inverted `which` defaults and bug B1 not ported | yes (defaults) |
| TG-7 | cocopp's runlength-based targets: enhancement after launch | – |

### ECDF, AUC, AOCC, ranks ([ecdf](ecdf.md))

| ID | Decision | Differs |
|---|---|---|
| EC-1 | Target-based ECDF, inclusive in f and budget | yes |
| EC-2 | Equal weight per (function, dimension) pair, over the pairs every algorithm has, with a warning; `pairs = "all"` option | yes |
| EC-3 | Exact step function, sampled only for display | yes (grid) |
| EC-4 | AUC as exact normalised integral | yes |
| EC-5 | AOCC on best-so-far with fixed bounds per pair (BBOB: 1e2, 1e-8); equals the continuum-ECDF AUC | – (new) |
| EC-6 | Current `get_data_ecdf` kept under a clear name, with EC-5 bounds | – |
| EC-7 | No simulated restarts | no (cocopp: yes) |
| EC-8 | Rank tables per pair, average ties, missing last | no |

### Distributions ([distributions](distributions.md))

| ID | Decision | Differs |
|---|---|---|
| DI-1 | Histogram breaks as R: `nclass.FD`, `pretty`, right-closed | no |
| DI-2 | Violin: Gaussian KDE, Silverman bandwidth, hard span (as plotly.js) | no (ggplot backend: yes) |
| DI-3 | Runtime distributions over successful runs, failures counted | no |
| DI-4 | Equal (pooled) bins by default with two or more algorithms | yes |

### Parameters and constraints ([parameters](parameters.md))

| ID | Decision | Differs |
|---|---|---|
| PA-1 | Parameter on the hitting row (target) or the last row ≤ b (budget) | yes |
| PA-2 | Parameter summaries with type 7 quantiles | no |
| PA-3 | Correlation views return raw matched rows | no |
| PA-4 | Penalty before best-so-far; "absolute" only on infeasible rows; penalties always worsen; infeasible when v ≠ 0 (negative values supported), size \|v\|^p | yes |
| PA-5 | JSON parameter-by-target lookup not ported | yes |
| PA-6 | Signed violation stored as logged; meaning per IOHexperimenter constraint type | – |

### Comparisons ([comparisons](comparisons.md), [glicko2](glicko2.md))

| ID | Decision | Differs |
|---|---|---|
| CM-1 | One-sided KS: exact p-value when n · m < 10000, else asymptotic; `exact` overrides | small samples |
| CM-2 | Failed runs count as +∞ | yes |
| CM-3 | Fixed-budget orientation follows maximisation | yes, maximisation |
| CM-4 | Holm over all ordered pairs; α = 0.01; heat map and graph | no |
| CM-5 | Win fractions with a selectable measure (ERT or mean hitting time, PAR-k, median; fixed budget mean, median, best); ERT and mean by default; equal weight per pair | yes |
| CM-6 | Simulated-restart bootstrap not ported now; possible later addition | yes |
| GL-1 | Tournament: one game per round, pair and algorithm pair; failures lose | close (sampling scheme) |
| GL-2 | PlayerRatings 1.1-0 Glicko-2 update, init (1500, 350, 0.06), τ a parameter, default 0.5 | yes (τ) |
| GL-3 | Seeded; golden tests use recorded game lists | yes (seed) |
| GL-4 | `method = "glicko2" \| "elo"` (Elo: K = 20, 1500) | – |

### Portfolio, CDP, EAF, DSC

| ID | Decision | Differs |
|---|---|---|
| SH-1 | Portfolio value: summed AUC of the virtual-best ECDF over all runs | yes |
| SH-2 | Exact Shapley up to 15 algorithms, seeded stratified sampling above | yes |
| SH-3 | Normalised by v(N), sum 1 | no |
| SH-4 | Generic in the portfolio value and the marginal contribution | – |
| CD-1 | RVCompare 0.1.8 bootstrap estimator, seeded | no |
| CD-2 | Failed runs +∞ | yes |
| CD-3 | Unequal sample sizes: larger sample subsampled (seeded) and shown to the user; `subsample=False` rejects | yes |
| CD-4 | ε = 1e-80 | no |
| EA-1 | EAF on best-so-far, levels 0–100 % in steps of 10 | no |
| EA-2 | EAF differences in both modes (EA-5) | no |
| EA-3 | Multi-function EAF on normalised quality; BBOB bounds 1e2 and 1e-8 | yes |
| EA-4 | `xmax = Min` bug not ported | yes |
| EA-5 | Precise (moocore) and discretised (matrix) modes, for the EAF and the difference | yes |
| DS-1 | Local DSC, no network | – |
| DS-2 | Inputs as IOHanalyzer's client, budgets from run metadata, B1 not ported; no NA sent; only problems every algorithm has, matched by name | yes |
| DS-3 | NA options PAR-k (the RT-3 definition, k user-set, default 10), ERT, Remove-na; ERT fallback is the sum of budgets | yes, JSON data |
| DS-4 | Ranking as dsc-core: all pairs, KS or AD (default AD, α = 0.05), equal when p ≥ α/m, equivalence check, groups by mean, average-position ranks; else ranks by mean | no |
| DS-5 | pDSC: one ε for all problems, iteration count, seeded Monte Carlo (DS-11) | no (seed added) |
| DS-6 | Omnibus: the service's names, statistics, `message` rule and `algorithm_means` (mean DSC ranks), limited by DS-12 to DS-14 | no |
| DS-7 | Post-hoc: z = (m₀ − mₐ)/SE, one-sided Φ(z), Holm and Hochberg over k − 1; method follows the omnibus test; all controls in one call; n, k from the ranked matrix | yes |
| DS-8 | Result fields keep the service's names | – |
| DS-9 | KS: scipy two-sided, exact when nₐ·n_b < 10 000, no tie jitter. AD: Scholz–Stephens midrank statistic, p from its limit distribution, held in [1e-5, 0.99999] | slightly (AD p) |
| DS-10 | At least two values per sample; identical samples p = 1; group order by the mean of member means; k ≥ 2 | yes, edge cases |
| DS-11 | Sequential pDSC as dsc-core (run pairs closer than ε → midpoint), runs paired up to the shorter sample; Monte Carlo: median p over seeded shuffles | yes (Monte Carlo) |
| DS-12 | `parametric_tests` always false; no paired t-test or ANOVA | only n ≤ 2 or tiny α |
| DS-13 | k = 2: one-sided Wilcoxon (`wilcoxon-signed-rank-less`), scipy defaults, t = 0 | not recorded |
| DS-14 | n < 2: no omnibus methods, "not enough problems" | not recorded |

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
- DSC PAR-1/PAR-10 leave NA in place for JSON data, and the ERT fallback for
  runs that never succeed is one budget instead of n for JSON data. The
  post-hoc test is chosen independently of the omnibus test, which the
  DSCTool documentation rules out. An algorithm missing a problem shifts the
  service's position-based pairing of problems ([dsc](dsc.md)).

## Open questions for the user

| ID | Question | Page |
|---|---|---|
| Q-EC-1 | Which data gave 0.67 and 0.90? | [ecdf](ecdf.md) |

DSC has no open questions. Q-DS-1 (answered): the DSCTool paper (2020) and
the v1.5 documentation were supplied; the 2017 and 2019 papers were not, and
dsc-core settles what they would. Q-DS-3 (answered): dsc-core may be read,
not copied; it was read for [dsc](dsc.md). Q-DS-4 (decided): the example
files carry no licence, so they are not committed; their numbers are cited.
Q-DS-5 (decided): no, one ε per call, as the service and IOHanalyzer do.

## Answered by the D6 review

Decided by the D6 review (dvermetten, 2026-11-06, IOHinspector#16).

| ID | Answer | Decision | Page |
|---|---|---|---|
| Q-AL-1 | No tolerance on the hit rule for COCO data | AL-8 | [alignment](alignment.md) |
| Q-RT-1 | k = 10 is the default | RT-3 | [runtime](runtime.md) |
| Q-RT-2 | Quantiles include unfinished runs by default | RT-5 | [runtime](runtime.md) |
| Q-FB-1 | Carry forward or simulated restarts; last evaluation kept in alignment | FB-6, AL-9 | [fixed-budget](fixed-budget.md) |
| Q-TG-1 | Range with a sensible separation from the data; 1e-12 only without other information | TG-3 | [targets](targets.md) |
| Q-TG-2 | Runlength-based targets: enhancement after launch | TG-7 | [targets](targets.md) |
| Q-EC-2 | Only the common pairs, with a warning; a parameter to average all, still warning | EC-2 | [ecdf](ecdf.md) |
| Q-DI-1 | Equal bins by default with two or more algorithms | DI-4 | [distributions](distributions.md) |
| Q-PA-1 | IOHexperimenter's constraint types checked; negative violations supported | PA-4, PA-6 | [parameters](parameters.md) |
| Q-CM-1 | Exact KS when the samples are small | CM-1 | [comparisons](comparisons.md) |
| Q-CM-2 | No bootstrap option now; possible later | CM-6 | [comparisons](comparisons.md) |
| Q-CM-3 | Win fractions take a selectable performance measure | CM-5 | [comparisons](comparisons.md) |
| Q-GL-1 | τ in the recommended range, modifiable; Elo/Glicko-2 switch | GL-2, GL-4 | [glicko2](glicko2.md) |
| Q-SH-1 | 15 is the limit; Shapley generic in the marginal contribution | SH-2, SH-4 | [portfolio](portfolio.md) |
| Q-CD-1 | Subsample by default, shown to the user | CD-3 | [cdp](cdp.md) |
| Q-EA-1 | BBOB bounds 1e2 and 1e-8; precise and discretised modes | EA-3, EA-5 | [eaf](eaf.md) |
| Q-DS-2 | PAR-k conventions unified, user-customisable; default k = 10 | DS-3 | [dsc](dsc.md) |

## Points to confirm at H1

Defaults chosen where an answer left a detail open. Each follows the
literature or the sources named on its page.

- Glicko-2 τ = 0.5 (Glickman's worked example, inside his range 0.3–1.2);
  Elo option K = 20, 1500, scale 400 (skelo's defaults). [glicko2](glicko2.md)
- "Sufficiently small" for the exact KS p-value: n · m < 10000, R's `ks.test`
  rule. [comparisons](comparisons.md)
- Win-fraction measures offered: ERT, mean hitting time, PAR-k, median (fixed
  target); mean, median, best (fixed budget). [comparisons](comparisons.md)
- Discretised EAF grid: 50 log-spaced budgets × 50 log-spaced values; BBOB data
  recognised from the suite name; the same 1e2 and 1e-8 bounds apply to AOCC.
  [eaf](eaf.md), [ecdf](ecdf.md)
- Simulated restarts for fixed budget: restart runs drawn uniformly with
  replacement from the run set, seeded, one chain per run. [fixed-budget](fixed-budget.md)
- Constraint violations: infeasible when v ≠ 0 (IOHexperimenter's rule for
  box and functional constraints), penalty size |v|^p; load-type violations
  (graph, pack-while-travel) need a user threshold. [parameters](parameters.md)
- DSC PAR-k uses the summary's definition (one cutoff per run set, RT-3),
  default k = 10. [dsc](dsc.md)
- DSC minimal decisions where the service is unknown or its 2019 code is
  defective: AD p-value from the limit distribution, Monte Carlo pDSC by the
  median p-value over seeded shuffles, no parametric omnibus tests, at least
  two problems for an omnibus test (DS-9 to DS-14). [dsc](dsc.md)
- CDP subsampling draws from the larger sample only. [cdp](cdp.md)
- Log-spaced targets: no automatic clipping of the span; 1e-12 only when no
  positive value was observed. [targets](targets.md)

## Tolerances

Counts, hitting times, ranks and test decisions match exactly. Means, rates,
ERT, AUC, AOCC and closed-form p-values agree to a relative error of 1e-12.
Exceptions are stated per page: Glicko-2 volatility (1e-5, minimiser
tolerance), log-spaced targets (1e-15), bootstrap-based outputs, which are
compared as distributions with a fixed seed, and DSC: omnibus and post-hoc
statistics to 1e-9 against the numbers of the published DSCTool example, and
Monte Carlo pDSC only against itself under a seed ([dsc](dsc.md)).

## How the worked examples were computed

IOHanalyzer's own R functions were sourced from the read-only reference copy
(R 4.3.3, data.table, magrittr, rjson, `src/align.cc` compiled with
`Rcpp::sourceCpp`, a three-line stand-in for `dplyr::mutate`), without
installing the package. iohinspector 0.0.8 ran from the worktree on polars
1.44.2. PlayerRatings 1.1-0 and RVCompare 0.1.8 came from CRAN source
tarballs. I03's container replaces this setup with pinned versions.
