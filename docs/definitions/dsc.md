# Deep Statistical Comparison (DSC)

Status: **specification for I18**, for sign-off at checkpoint H1. Statements
in the "Settled" lists follow from the sources named next to them. Points the
sources leave open are numbered DQ-1 to DQ-21. I18 step 2 decides them from
recorded service responses. I18 must not implement them from a guess.
Parity-matrix row: "DSC" (local implementation, decision in the plan's I18).
Implemented in: I18.
Sources: [Eftimov2020], [DSCTool2022], [IOHanalyzer]; see
[references](references.md).

## Sources

- **[Eftimov2020]** is the supplied `paper.pdf` (*Applied Soft Computing* 87,
  2020, article 105977). Read in full. Page numbers are the journal's.
- **[DSCTool2022]** is the supplied `documentation.pdf`, the DSCTool v1.5
  service documentation of 20 January 2022. Read in full. Page numbers are PDF
  pages. From page 2 on, the printed page number is one lower.
- Both are copyrighted and are not in the repository. This page paraphrases
  them and quotes only short phrases.
- **Not supplied, not read:** [Eftimov2017], which defines the ranking scheme
  in full; [Eftimov2019], which defines practical DSC in full; [Eftimov2017b],
  which [Eftimov2020] cites for the choice between KS and AD (p. 3); and the
  DSCTool R client manual ([DSCTool2022] p. 14).
- **Available, not read:** [dsc-core], the source of DSCTool's core library,
  which [Eftimov2020] names (pp. 3, 5). It is public and licensed
  GPL-3.0-or-later (Q-DS-3). The example requests and responses of
  [Eftimov2020] section 5 are public at `https://cs.ijs.si/dl/dsctool/`
  (Q-DS-4).

The two PDFs settle the structure of the method, the choice of tests, the
Bonferroni correction, what `epsilon` and `monte_carlo_iterations` switch on,
the omnibus and post-hoc interfaces, and every request and response field.
They do not settle the exact transitivity check and rank assignment, the
practical-DSC preprocessing, how the service picks the tests it offers, or
the formulas of the statistics. [Eftimov2020] refers to [Eftimov2017] and
[Eftimov2019] for these.

## Notation

k algorithms, n problems. A problem is one (function, dimension) pair, which
IOHanalyzer names `F{funcId}_{DIM}D` (`R/stats.R:964`). x(a, p) is the sample
of algorithm a on problem p: one value per run, smaller is better (step 1).
r(a, p) is the DSC rank of a on p.

## Pipeline

1. Client-side data preparation (not part of the service).
2. Ranking per problem: DSC, or practical DSC (pDSC). Endpoint `rank`.
3. Omnibus test on the k × n matrix of ranks. Endpoint `omnibus`.
4. Post-hoc test against a control algorithm. Endpoint `posthoc`.

[Eftimov2020] describes DSC as two statistical steps, a ranking and then a
test on the rankings (p. 3), with the post-hoc test applied when the omnibus
test finds a difference (p. 4; [DSCTool2022] p. 7). Its Fig. 2 (p. 6) shows
the same chain of services.

## Step 1: client-side data preparation

The service takes, for each algorithm and problem, a list of double values
([DSCTool2022] p. 3). The sources say nothing about missing values, about
the direction of the ranking, or about sample sizes. All of this is the
client's job. IOHanalyzer's `convert_to_dsc_compliant` (`R/stats.R` 901–973)
builds one vector of per-run values per algorithm and (function, dimension)
pair:

- **fixed target**: the runtime sample at the pair's target. Failed runs are
  NA and are handled by `na.correction` (Shiny default `PAR-10`, the first
  choice in `inst/shiny-server/ui/DSC_interface.R` 155–156):
  - `PAR-1`: the run's own budget (929–932);
  - `PAR-10`: 10 × the run's own budget (933–936);
  - `ERT`: the run set's ERT ([runtime](runtime.md)), or `sum(maxRT)` when
    the ERT is infinite (937–943);
  - `Remove-na`: the failed runs are dropped (944–945).
- **fixed budget**: the value sample at the pair's budget, negated under
  maximisation so that smaller is always better (955). The Shiny value page
  passes no `na.correction` (`inst/shiny-server/server/FV_DSCinterface.R`
  12–16). `get_FV_sample` returns NA when the budget is below a run's first
  logged evaluation ([fixed-budget](fixed-budget.md)), and the client then
  only warns that NA "will likely result in an error" (957–962).

Reconciled with the sources:

- **Orientation.** The sources rank "obtained solutions" and never state the
  direction. Their post-hoc example takes the algorithm with the lowest mean
  rank as control and finds that it significantly outperforms the others
  ([Eftimov2020] p. 10; [DSCTool2022] p. 12), so rank 1 is best. That the
  service puts the smallest values first is what IOHanalyzer's sign flip
  assumes. DQ-1 confirms it.
- **Missing values.** The documented format has no missing value, so the
  PAR-k, ERT and Remove-na options are entirely the client's, as the plan
  says. Nothing in the sources conflicts with them.
- **Ties.** PAR-1, PAR-10 and ERT give every failed run of a set the same
  value (with equal budgets), so samples with failures contain ties. How the
  service's tests treat ties is DQ-2.
- **Sample sizes.** `Remove-na` makes sample sizes unequal and leaves an
  empty sample when every run failed. The sources do not say what the service
  accepts (DQ-6, DQ-11).

Defects in IOHanalyzer's preparation:

- For IOHexperimenter JSON data `maxRT` is a single number
  ([runtime](runtime.md)), and `budgets[is.na(data)]` then indexes a length-1
  vector with a logical vector, which yields NA. PAR-1 and PAR-10 therefore
  leave the NA values in place for all JSON data, and the service call fails
  (checked in R 4.3.3: `budgets <- 100; data <- c(5, NA, 7, NA);
  data[is.na(data)] <- budgets[is.na(data)]` leaves `5 NA 7 NA`).
- When no run succeeds, the function's documentation promises "nr_runs *
  budget" (`R/stats.R:993`), but the code uses `sum(attr(ds, 'maxRT'))`
  (940). For JSON data that is one budget, the largest logged evaluation
  (`R/readFiles.R:1638`), not n budgets.
- Defect B1: when no targets are given, `convert_to_dsc_compliant` passes its
  fixed-target `which = 'by_RT'` to `get_target_dt` (912), which returns
  budgets; the Shiny runtime page does the same explicitly
  (`inst/shiny-server/server/RT_DSCinterface.R:200`). See
  [targets](targets.md).

## Step 2: ranking per problem (`rank`)

Settled:

- Each problem is ranked on its own, from its k samples only. This step is
  non-parametric ([Eftimov2020] p. 3).
- Every pair of algorithms is compared with a two-sample test of whether the
  two samples come from the same distribution: the two-sample
  Kolmogorov–Smirnov test (KS) or the two-sample Anderson–Darling test (AD)
  ([Eftimov2020] p. 3; [DSCTool2022] p. 3). All pairwise comparisons are
  needed ([Eftimov2020] p. 3).
- The pairwise p-values are corrected with the Bonferroni correction, which
  controls the family-wise error rate, and are arranged in a matrix. The
  ranks then follow from a "transitivity condition" on that matrix
  ([Eftimov2020] pp. 3–4). The significance level is the request's
  `method.alpha` ([DSCTool2022] p. 3).
- The response gives one rank per (problem, algorithm) as a double
  ([DSCTool2022] p. 4). Lower is better (step 1). The response contains no
  p-values, so the pairwise tests are observable only through the ranks.
- With more than four algorithms, the authors say the ranking suffers from
  the family-wise error control, and they recommend comparisons with a
  control algorithm, referring to their earlier studies ([Eftimov2020] p. 6).
  The documentation sets no limit on k.

Not settled by the supplied sources:

- **DQ-2. The tests.** Two-sided or one-sided; for KS, the exact or the
  asymptotic p-value and the handling of ties; for AD, which two-sample
  statistic (for example the k-sample test of [Scholz1987] with two samples)
  and how its p-value is computed (asymptotic, tabulated, interpolated,
  capped).
- **DQ-3. The correction.** The number of comparisons m (presumably the
  k(k−1)/2 unordered pairs), whether p is multiplied by m or α divided by m,
  and whether two samples count as equal when the corrected p is > α or ≥ α.
- **DQ-4. The transitivity check.** How the relation "not significantly
  different" is checked for transitivity, and what the scheme does when the
  check fails (for example A equal to B, B equal to C, A different from C).
- **DQ-5. Rank values.** Which statistic orders the groups of equal
  algorithms (mean, median or another), which rank a group of tied
  algorithms receives (for example the average of the positions it spans,
  which the double-valued ranks suggest), and the ranks when the transitivity
  check fails.
- **DQ-6. Edge cases.** Unequal sample sizes, a sample of one run, an empty
  sample, a constant sample, identical samples, a problem that only some
  algorithms have, and k = 1 (IOHanalyzer requires two algorithms,
  `RT_DSCinterface.R` 189–193).

## Practical DSC (`epsilon`, `monte_carlo_iterations`)

Settled:

- pDSC separates practical from statistical significance ([Eftimov2020]
  p. 2). Its threshold ε (the paper's ε_p) is chosen by the user and is the
  difference within which two solutions are not considered practically
  different. With ε = 0 the scheme reduces to standard DSC ([Eftimov2020]
  p. 4).
- There are two variants ([Eftimov2020] p. 2). **Sequential pDSC**
  preprocesses the independent runs in their given order. **Monte Carlo
  pDSC** removes the dependence on that order: for each pairwise comparison,
  it repeats the analysis over permutations of the runs within each
  algorithm ([Eftimov2020] p. 4).
- The service takes one `epsilon` per request, for all problems
  ([DSCTool2022] p. 3), although [Eftimov2020] calls the threshold problem
  specific (p. 4). A value > 0 switches pDSC on. `monte_carlo_iterations` > 0
  selects the Monte Carlo variant, with that number of permutations
  ([DSCTool2022] p. 3). So ε = 0 gives DSC, ε > 0 with 0 iterations gives
  sequential pDSC, and ε > 0 with iterations > 0 gives Monte Carlo pDSC.
- IOHanalyzer passes ε unscaled, and the same ε applies to every (function,
  dimension) pair. If ε is absolute (DQ-7), it is in the units of the
  samples: evaluations (after the PAR-k substitution) on the runtime page,
  negated function values on the value page.
- Monte Carlo iterations dominate the service's run time ([Eftimov2020]
  p. 6). The API has no seed parameter ([DSCTool2022] p. 3).

Not settled by the supplied sources:

- **DQ-7. The preprocessing.** Which values are compared with ε (the order
  dependence suggests run j of one algorithm against run j of the other, but
  the sources do not say), what replaces two values that are within ε, < or
  ≤ ε, and whether ε is absolute (as the wording suggests) or relative.
- **DQ-8. Combining the iterations.** What the Monte Carlo variant aggregates
  over the permutations (p-values, test decisions or ranks) and how.
- **DQ-9. ε = 0 with iterations > 0.** [Eftimov2020] p. 4 says ε = 0 is
  standard DSC; [DSCTool2022] p. 3 says iterations > 0 start the Monte Carlo
  variant. Which wins.
- **DQ-10. Randomness.** Whether repeated identical Monte Carlo requests give
  identical responses.
- **DQ-11. Unequal sample sizes under pDSC**, which a run-by-run pairing
  could not handle.

## Step 3: omnibus test (`omnibus`)

Settled:

- The request carries the rank response's `ranked_matrix`,
  `number_algorithms` and `parametric_tests`, plus `method = {name, alpha}`
  ([DSCTool2022] pp. 6–7). The name is one of the rank response's
  `valid_methods` ([DSCTool2022] pp. 4, 12).
- The ranking step checks the conditions for the safe use of parametric
  tests on its own output (independent samples, normality, equal variances)
  and points only to the tests that are appropriate, so that the user does
  not have to ([Eftimov2020] p. 4). `parametric_tests` says whether a
  parametric test is eligible ([DSCTool2022] p. 4).
- With too few problems, `valid_methods` holds the message "Not enough
  problems for multiple-problem analysis!" instead of test names
  ([DSCTool2022] p. 4).
- The response gives `message` (whether the null hypothesis was rejected),
  `p_value`, `t` (the test statistic, or 0 where there is none), `method`,
  and `algorithm_means`, one mean ranking per algorithm ([DSCTool2022] p. 7).
- Method names in the sources: `"wilcoxon-signed-rank-less"`, a two-algorithm
  test whose p-value is that of the algorithm with the lower mean against the
  one with the higher mean; the name and the description indicate a
  one-sided test ([DSCTool2022] p. 7); the
  Friedman, Friedman aligned-ranks and Iman–Davenport tests, named by the
  post-hoc description ([DSCTool2022] p. 8) and the worked example
  ([Eftimov2020] p. 9). The documentation gives no complete list and no
  name strings for the last three.
- Fig. 1 of [Eftimov2020] (p. 5) shows the general selection pipeline that
  the tool automates. For paired samples: with two algorithms, the paired
  t-test when the conditions hold, otherwise the Wilcoxon test; with more
  than two, repeated-measures ANOVA when they hold, otherwise Friedman,
  Friedman aligned ranks or Iman–Davenport. Normality can be checked with KS
  or Shapiro–Wilk and equal variances with Levene or Bartlett. The figure
  gives examples ("etc."), not the service's exact choice.
- Worked example: 3 algorithms on the 22 functions of BBOB 2015 in dimension
  10, KS at α = 0.05, then Friedman at α = 0.05 rejects with p = 0.007
  ([Eftimov2020] pp. 8–9; [DSCTool2022] pp. 11–12).

Not settled by the supplied sources:

- **DQ-12. The valid methods.** The full list of name strings, and the rule
  that picks them from k and `parametric_tests`.
- **DQ-13. The parametric check.** Which normality and equal-variance tests,
  at which α (the request's or a fixed one), applied to which data (for
  example each algorithm's column of ranks).
- **DQ-14. Too few problems.** The smallest n that gives test names.
- **DQ-15. The statistics.** For each method, the statistic in `t` and the
  p-value: for example Friedman's χ² with or without a tie correction,
  Iman–Davenport's F, the aligned-ranks statistic, the parametric tests, and
  Wilcoxon exact or normal.
- **DQ-16. `algorithm_means`.** Whether it is the mean DSC rank of each
  algorithm, or the mean of the ranks the omnibus method computes (aligned
  ranks differ from DSC ranks). The post-hoc step uses only these means.
- **DQ-17. Input checks.** Whether α changes anything but `message`, and
  whether the service re-checks `parametric_tests` or rejects a name that is
  not in `valid_methods`.

## Step 4: post-hoc test (`posthoc`)

Settled:

- Every algorithm is compared with one control algorithm, `base_algorithm`,
  which must be one of the algorithms in `algorithm_means` ([DSCTool2022]
  p. 8).
- The request carries `algorithm_means` from the omnibus response, `k` (the
  number of algorithms), `n` (the number of problems), `base_algorithm` and
  `method = {name: "friedman" | "friedman-aligned-rank", alpha}`
  ([DSCTool2022] p. 8).
- The post-hoc method must match the omnibus test: `friedman` after the
  Friedman or Iman–Davenport test, `friedman-aligned-rank` after the
  Friedman aligned-ranks test ([DSCTool2022] p. 8).
- It is meant for the case where the omnibus test rejects ([DSCTool2022]
  p. 7; [Eftimov2020] p. 5).
- The response's `adjusted_p_values` holds four named lists, in this order:
  `ZValue` (the statistic), `UnadjustedPValue` (computed from the z-value),
  `Holm` and `Hochberg`. Each lists `{algorithm, value}` for the algorithms
  compared with the control ([DSCTool2022] pp. 8–9).
- The service receives only the means, k and n. Each z-value is therefore a
  function of the control's mean, the other algorithm's mean, k and n alone.
- Worked example: control CMA-MSR (the lowest mean), k = 3, n = 22,
  `friedman` at α = 0.05; CMA-MSR is significantly better than both others
  ([Eftimov2020] p. 10; [DSCTool2022] p. 12).

Hypothesis for DQ-18 to DQ-20, to be tested, not specified. It comes from the
usual comparisons with a control in [Derrac2011] and [Garcia2010], which
were not re-read for this page: z = (m_a − m_0) / SE with SE = √(k(k+1)/(6n))
for `friedman` and SE = √(k(n+1)/6) for `friedman-aligned-rank`, where m_a
and m_0 are the means of algorithm a and of the control; the unadjusted
p-value from the standard normal distribution; Holm's and Hochberg's
adjusted p-values over the k − 1 comparisons with the control.

Not settled by the supplied sources:

- **DQ-18. The z-value.** Its formula and standard error for each method, and
  its sign.
- **DQ-19. One- or two-sided p-values.** IOHanalyzer's heat map marks a
  pair (i, j) as significant from the list with control i and subtracts the
  transpose (`inst/shiny-server/server/RT_DSCinterface.R` 143–150). With
  two-sided unadjusted p-values this matrix is zero everywhere, so the
  display assumes one-sided values or is wrong.
- **DQ-20. The adjusted values.** The form of the Holm and Hochberg values
  (adjusted p-values capped at 1, or another form), and whether the control
  itself appears in the lists.
- **DQ-21. Unsupported combinations.** What the service does with a post-hoc
  method that does not match the omnibus test, or after an omnibus test that
  has no documented post-hoc (the parametric and two-algorithm tests).

## Service interface as IOHanalyzer uses it

All endpoints take and return JSON by POST, with HTTP Basic
authentication after registration at `manage/user` ([DSCTool2022] pp. 2–3;
IOHanalyzer's `register_DSC`, `R/stats.R` 787–833). Every response wraps its
fields in `result`, next to a top-level `success`. IOHanalyzer reads only
`result` (`R/stats.R` 1036, 1079, 1134) and treats NULL as a failure
(`RT_DSCinterface.R` 15–19). In the tables below, rows marked → are response
fields inside `result`. Line numbers without a file refer to the last file
named above them in the same column.

Base URLs. IOHanalyzer uses `https://ws.ijs.si:8443/dsc-1.5/service/`
(`R/stats.R` 795, 1014, 1061, 1115). [DSCTool2022] p. 2 gives
`https://ws.ijs.si/dsc/service/` for the latest version and
`https://ws.ijs.si/dsc/1.5/service/` for v1.5. [Eftimov2020] p. 5 gives the
v1.4 URL on port 8443. The 8443 host did not answer during I02, and
`https://ws.ijs.si/dsc/service/` did not answer on 6 November 2026, both
from this environment.

### `rank` ([DSCTool2022] pp. 3–4)

| Field | Meaning | IOHanalyzer |
|---|---|---|
| `method.name` | `"KS"` or `"AD"` | default AD (`ui/DSC_interface.R` 180–182; `R/stats.R` 1019) |
| `method.alpha` | level of the pairwise tests | default 0.05 (157–159) |
| `epsilon` | pDSC threshold; > 0 switches pDSC on | default 0 (160–162) |
| `monte_carlo_iterations` | permutations; > 0 selects Monte Carlo pDSC | default 0 (170–172) |
| `data[].algorithm` | algorithm name | data set ID |
| `data[].problems[].name` | problem name | `F{funcId}_{DIM}D` |
| `data[].problems[].data` | the sample, doubles | step 1 |
| → `valid_methods` | omnibus test names, or the "not enough problems" message | omnibus choices (`RT_DSCinterface.R` 20–21) |
| → `ranked_matrix[]` | `{problem, result: [{algorithm, rank}]}`, rank a double | passed to `omnibus`; PerformViz plot (`R/plotDataSetList.R:2207`); rank table (`RT_DSCinterface.R` 39–55) |
| → `number_algorithms` | number of algorithms ranked | passed to `omnibus` |
| → `parametric_tests` | whether a parametric omnibus test is eligible | passed to `omnibus` |

### `omnibus` ([DSCTool2022] pp. 6–7)

| Field | Meaning | IOHanalyzer |
|---|---|---|
| `method.name` | one of `valid_methods` | user choice, default the first (`R/stats.R` 1057–1060) |
| `method.alpha` | level of the omnibus test | default 0.05 (`ui/DSC_interface.R` 230–232) |
| `ranked_matrix`, `number_algorithms`, `parametric_tests` | copied from the rank response | copied (`R/stats.R` 1062–1067) |
| → `message` | whether the null hypothesis was rejected | shown with the p-value (`RT_DSCinterface.R:92`) |
| → `p_value` | p-value | shown |
| → `t` | statistic, 0 if none | not read |
| → `method` | the test used | not read |
| → `algorithm_means[]` | `{algorithm, mean}` | passed to `posthoc` |

### `posthoc` ([DSCTool2022] pp. 7–9)

| Field | Meaning | IOHanalyzer |
|---|---|---|
| `method.name` | `"friedman"` or `"friedman-aligned-rank"`, matching the omnibus test | user choice, default `friedman`, independent of the omnibus choice (`ui/DSC_interface.R` 253–255) |
| `method.alpha` | level | default 0.05 (259–261) |
| `algorithm_means` | from the omnibus response | copied |
| `n` | number of problems | rows of the target table (`RT_DSCinterface.R:107`) |
| `k` | number of algorithms | number of selected IDs (106) |
| `base_algorithm` | the control | each algorithm in turn (105–110); R default the first in `algorithm_means` (`R/stats.R:1112`) |
| → `adjusted_p_values[]` | `ZValue`, `UnadjustedPValue`, `Holm`, `Hochberg`, each `{name, algorithms: [{algorithm, value}]}` | read by position, not by name, into columns z, unadjusted P, Holm, Hochberg (`RT_DSCinterface.R` 112–124) |

The Shiny post-hoc view builds a k × k matrix from the k calls, using the
correction the user picks (Holm by default, `ui/DSC_interface.R` 256–258),
and marks p ≤ α (`RT_DSCinterface.R` 138–157; see DQ-19).

## How iohinspector 0.0.8 implements it

No DSC.

## Differences

IOHanalyzer against this specification:

- PAR-1 and PAR-10 leave NA for JSON data, and the ERT fallback is one
  budget instead of n budgets for JSON data (step 1).
- Default targets swapped (B1).
- The post-hoc method is chosen independently of the omnibus test, which
  [DSCTool2022] p. 8 rules out, and the post-hoc runs whether or not the
  omnibus test rejects.
- `n` is the number of rows of the target table, not the number of problems
  in the ranked matrix. They can differ when an algorithm lacks a problem
  (DQ-6).
- The "not enough problems" message is offered as an omnibus choice
  (`RT_DSCinterface.R` 20–21) and then sent as a method name.
- The `epsilon` parameter is documented as being for "eDSC"
  (`R/stats.R:983`). In the sources ε belongs to pDSC, and eDSC is the
  multivariate service ([Eftimov2020] p. 4).

## Proposed decisions

- **DS-1.** iohinspector implements DSC locally (no network), as the plan's
  I18 decides; DSCtool serves only to record reference responses.
- **DS-2. Inputs.** As step 1, with the run's budget taken from run metadata
  (fixing the PAR-1/PAR-10 NA defect), fixed-budget values oriented so that
  smaller is better, and default targets from TG-4 (B1 not ported). A
  fixed-budget run without a value at the budget (the budget is below its
  first evaluation) is an error that names the problem, instead of an NA
  sent to the ranking. Only problems that every algorithm has enter the
  analysis, and the dropped problems are reported, because the omnibus tests
  are tests for paired samples ([Eftimov2020] Fig. 1, p. 5).
- **DS-3. Missing values.** Options PAR-k, ERT and Remove-na. PAR-k is the one
  definition of [runtime](runtime.md) (RT-3): a failed run counts as k · κ, with
  one cutoff κ per run set (the budget cap C, otherwise max_i B_i from run
  metadata). k is user-customisable and its default is 10, so the default is
  PAR-10 as in Shiny and as in the runtime summary. IOHanalyzer's two fixed
  options PAR-1 and PAR-10 are the cases k = 1 and k = 10; they differ from
  IOHanalyzer's client only where budgets differ between runs of a set, because
  the client charges each run its own budget. The two PAR-k conventions are
  unified. Decided by the D6 review (dvermetten, 2026-11-06). When no run
  succeeds, the ERT option uses the sum of the runs' budgets, as
  IOHanalyzer's documentation states, also for JSON data.
- **DS-4. Ranking.** Per problem, all pairs of algorithms, the two-sample KS
  or AD test (default AD, α = 0.05, as in Shiny), Bonferroni correction over
  the pairs, and ranks from the transitivity rule, lower is better. The test
  variants, the form of the correction, the transitivity check and the tie
  ranks are those the recordings establish (DQ-1 to DQ-6). iohinspector warns
  when k > 4 ([Eftimov2020] p. 6) but does not refuse.
- **DS-5. Practical DSC.** One ε for all problems and an iteration count, with
  the service's switches (ε = 0: DSC; ε > 0: sequential pDSC; ε > 0 and
  iterations > 0: Monte Carlo pDSC). The preprocessing and the combination of
  iterations are those the recordings establish (DQ-7 to DQ-11). Monte Carlo
  pDSC takes a seed, so iohinspector's results are reproducible whatever the
  service does.
- **DS-6. Omnibus.** The same method names, the same rule for the valid
  methods, the same parametric check and the same statistics as the service
  (DQ-12 to DQ-17).
- **DS-7. Post-hoc.** The post-hoc statistic follows from the omnibus method
  (`friedman` for Friedman and Iman–Davenport, `friedman-aligned-rank` for
  Friedman aligned ranks), instead of a free choice. Where the service offers
  no post-hoc for a method, iohinspector offers none (DQ-21). One call
  returns the four value lists for every algorithm as control, which is the
  k × k matrix the Shiny page builds from k calls. n and k come from the
  ranked matrix. The omnibus result is returned with it; the post-hoc is not
  refused when the omnibus test does not reject, but the result says so.
- **DS-8. Names.** Result objects keep the service's field names
  (`ranked_matrix`, `valid_methods`, `parametric_tests`, `algorithm_means`,
  `ZValue`, `UnadjustedPValue`, `Holm`, `Hochberg`), so that tests compare
  recorded responses field by field.

## Open points for I18 step 2

These are not questions for the user. The recorded service responses decide
them, and I18 records each answer on this page before implementing it.

Notes for the recording design:

- The rank response contains no p-values. With k = 2, the ranks show only
  whether the corrected p-value is above α. Repeating one request over a
  sequence of α values (bisection) brackets the corrected p-value of that
  pair to any precision. About 20 throttled requests per pair are enough.
- `omnibus` and `posthoc` take their inputs as given, so they can also be
  recorded on synthetic ranked matrices and means, not only on rank
  responses.
- If the user agrees (Q-DS-4), the public example of [Eftimov2020] section 5
  is a first reference case that needs no credentials.
- Each recording names the base URL and service version it used.

| ID | Point | Cases that decide it |
|---|---|---|
| DQ-1 | Smaller values rank first | k = 2, one sample entirely below the other |
| DQ-2 | Test variants and p-values | k = 2, α bisection; samples where exact and asymptotic KS, or AD variants, give different p-values; with and without ties; n = 5, 15, 30, 100 |
| DQ-3 | Bonferroni form, m, > or ≥ | the same pair with k = 2, 3 and 4 (extra algorithms far away), α bisection; the ratio of the thresholds gives m |
| DQ-4 | Transitivity check | three algorithms with A ~ B, B ~ C and A ≁ C (shifted samples) |
| DQ-5 | Group order and tie ranks | two groups of equal samples; groups whose mean and median orders disagree; all algorithms equal |
| DQ-6 | Edge cases | unequal sizes, one run, empty sample, constant samples, identical samples, a problem missing for one algorithm |
| DQ-7 | pDSC preprocessing | k = 2: the same values in different run orders; differences of exactly ε; the data scaled by 1000 with the same ε |
| DQ-8 | Combining iterations | an order-sensitive case with 1, 10, 100 and 1000 iterations |
| DQ-9 | ε = 0 with iterations > 0 | a DSC case repeated with 100 iterations |
| DQ-10 | Determinism | every Monte Carlo case sent three times |
| DQ-11 | Unequal sizes under pDSC | the DQ-6 unequal-size case with ε > 0 |
| DQ-12 | Valid method names and rule | k = 2, 3 and 5, each with n = 1, 2, 3, 5, 10 and 30 |
| DQ-13 | Parametric check | rankings that are close to normal and far from it, with equal and unequal spread; check `parametric_tests` |
| DQ-14 | Smallest n | the n sweep of DQ-12 |
| DQ-15 | Omnibus statistics | every valid method on the same rankings, compared with scipy and R implementations of each candidate |
| DQ-16 | `algorithm_means` | a ranked matrix with tied ranks, under Friedman and aligned ranks |
| DQ-17 | Omnibus input checks | an α sweep; a name not in `valid_methods`; `parametric_tests` flipped |
| DQ-18 | z-value | both methods, every control, several (k, n), compared with the hypothesis above |
| DQ-19 | One- or two-sided | the DQ-18 cases with the control above and below the others |
| DQ-20 | Holm and Hochberg values | k = 5 and 8, where the two procedures differ |
| DQ-21 | Unsupported combinations | `friedman-aligned-rank` after Friedman; post-hoc after a parametric and a Wilcoxon omnibus test |

## Open questions for the user

- Q-DS-1 (answered): the user supplied [Eftimov2020] and the DSCTool v1.5
  documentation [DSCTool2022]. [Eftimov2017] and [Eftimov2019] were not
  supplied. The points only they would settle are DQ-2 to DQ-11, left to I18
  step 2. Supplying them would let I18 check its reading against the source
  instead of inferring it from responses alone.
- Q-DS-2 (answered): the two PAR-k conventions are unified, user-customisable
  with a sensible default. Decided by the D6 review (dvermetten, 2026-11-06).
  The default is k = 10, matching the answer to Q-RT-1, and the definition is
  the summary's (RT-3). See DS-3. Remaining point for H1: the shared
  definition uses one cutoff per run set rather than each run's own budget; the
  review did not say which of the two to keep.
- Q-DS-3: May I18 read [dsc-core], the public source of DSCTool's core
  library, to settle the DQ points? It is GPL-3.0-or-later and iohinspector
  is MIT, so its code must not be copied or translated. Reading it to check a
  specification is a licensing judgement for the user. Its last commit (2019)
  is older than service version 1.5.
- Q-DS-4: May I18 download the public example files at
  `https://cs.ijs.si/dl/dsctool/` (3 algorithms, 22 problems, with rank,
  omnibus and post-hoc responses) as a reference case, and commit them under
  `tests/golden/dsc/`? Their licence is not stated.
- Q-DS-5: The service takes one ε for all problems, while [Eftimov2020] calls
  the threshold problem specific (p. 4). Should iohinspector also accept one
  ε per problem, beyond what the service offers?

## Test tolerance

- Ranks and every decision match the recorded responses exactly: the rank
  groups, `valid_methods`, `parametric_tests`, the omnibus `message`, and
  post-hoc significance at α.
- Pairwise test p-values are not in the responses. They are compared only
  through decisions, or through the α brackets of DQ-2.
- Omnibus `p_value` and `t`, and post-hoc `ZValue`, `UnadjustedPValue`,
  `Holm` and `Hochberg`: relative error 1e-9, or the precision the service
  prints if that is coarser. This is looser than the 1e-12 of the
  [README](README.md) because the service's distribution functions come from
  an unknown Java library. I18 may relax it for one statistic only with
  recorded evidence and the reason written here.
- Monte Carlo pDSC: iohinspector cannot reproduce the service's random
  permutations. If DQ-10 shows that the service is deterministic, its ranks
  are compared with iohinspector's over several seeds; otherwise both are
  compared as distributions. I18 states the criterion from the recorded
  spread.
