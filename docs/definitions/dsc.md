# Deep Statistical Comparison (DSC)

Status: **specification for I18**, for sign-off at checkpoint H1. Statements
in the "Settled" lists follow from the sources named next to them. The points
the 2020 paper and the service documentation left open, DQ-1 to DQ-21, are
now closed: most from the DSCTool core library [dsc-core] and the public
example of [Eftimov2020] section 5, the rest by a decision marked
"(minimal)", which covers what IOHanalyzer does with DSC and nothing more.
Two points can still be checked against the live service (see "Optional
recordings"); without them, this page is the reference.
Parity-matrix row: "DSC" (local implementation, decision in the plan's I18).
Implemented in: I18.
Sources: [Eftimov2020], [DSCTool2022], [dsc-core], [IOHanalyzer]; see
[references](references.md).

## Sources

- **[Eftimov2020]** is the supplied `paper.pdf` (*Applied Soft Computing* 87,
  2020, article 105977). Read in full. Page numbers are the journal's.
- **[DSCTool2022]** is the supplied `documentation.pdf`, the DSCTool v1.5
  service documentation of 20 January 2022. Read in full. Page numbers are PDF
  pages. From page 2 on, the printed page number is one lower.
- Both are copyrighted and are not in the repository. This page paraphrases
  them and quotes only short phrases.
- **[dsc-core]**, the public source of DSCTool's core library (Java, GPL-3.0-or-
  later), read in full at commit 4b2f2d9 (19 August 2019, library version 1.4)
  with the user's permission (Q-DS-3). This page states facts about the
  algorithm in its own words. No code was copied or translated, and I18 must
  not do so either: iohinspector is MIT. Where the page says "dsc-core does
  X", it describes that commit, not necessarily service version 1.5.
- **[DSCExample]**: the example requests and responses of [Eftimov2020]
  section 5 and [DSCTool2022] section 3 (3 algorithms, the 22 BBOB functions
  in dimension 10). They are gone from `https://cs.ijs.si/dl/dsctool/` (404
  on 10 November 2026) but archived by the Wayback Machine (snapshots of
  15 June 2022). They carry no licence. This page cites their numbers; the
  files are not in the repository (Q-DS-4).
- **Not read:** [Eftimov2017], [Eftimov2019] and [Eftimov2017b] (no open copy
  found; the repository copy at DiRROS did not answer), and the DSCTool R
  client manual. dsc-core and the example settle what they would.

## Notation

k algorithms, n problems. A problem is one (function, dimension) pair, which
IOHanalyzer names `F{funcId}_{DIM}D` (`R/stats.R:964`). x(a, p) is the sample
of algorithm a on problem p: one value per run, smaller is better (step 1).
r(a, p) is the DSC rank of a on p. m = k(k − 1)/2 is the number of pairs.

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
([DSCTool2022] p. 3). The sources say nothing about missing values or sample
sizes. All of this is the client's job. IOHanalyzer's
`convert_to_dsc_compliant` (`R/stats.R` 901–973) builds one vector of per-run
values per algorithm and (function, dimension) pair:

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

- **Orientation.** Smaller values rank first: dsc-core orders algorithms by
  ascending sample mean (step 2). In [DSCExample], CMA-MSR has the lowest
  mean rank and is the significantly better control ([Eftimov2020] p. 10).
  IOHanalyzer's sign flip under maximisation is therefore right (DQ-1).
- **Missing values.** The documented format has no missing value, so the
  PAR-k, ERT and Remove-na options are entirely the client's, as the plan
  says. Nothing in the sources conflicts with them.
- **Ties.** PAR-1, PAR-10 and ERT give every failed run of a set the same
  value (with equal budgets), so samples with failures contain ties. Step 2
  says how the tests treat them (DQ-2).
- **Problems are matched by position.** dsc-core takes the problem names
  from the first algorithm and pairs the i-th problem of every algorithm with
  the i-th problem of the first. A client that leaves a problem out for one
  algorithm silently shifts the others.

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
- An algorithm that lacks a (function, dimension) pair gets a shorter problem
  list, which the service then pairs by position (above).

## Step 2: ranking per problem (`rank`)

Settled:

- Each problem is ranked on its own, from its k samples only
  ([Eftimov2020] p. 3).
- Every pair of algorithms is compared with a two-sample test of whether the
  two samples come from the same distribution: KS or AD ([Eftimov2020] p. 3;
  [DSCTool2022] p. 3). The name is matched without regard to case.
- **KS (DQ-2).** dsc-core calls Apache Commons Math 3.6.1
  [CommonsMath361]. The test is two-sided. When the product of the two
  sample sizes is below 10 000, the p-value is the exact P(D ≥ d), and tied
  values are first broken by a small deterministic jitter (fixed seed);
  otherwise it is the asymptotic Kolmogorov p-value on the data as given.
- **AD (DQ-2).** dsc-core takes the standardised two-sample statistic of
  [Scholz1987] in its version for ties (midranks, their A²_akN), as computed
  by ELKI 0.7.1 [ELKI071]: (A² − 1)/σ_N, with the finite-sample σ_N of
  [Scholz1987]. The p-value comes from a table of the statistic's upper
  quantiles for one degree of freedom (k − 1 = 1), interpolated by a cubic
  spline in logit(p). Outside the table it is held at the end values, so p
  lies in [0.00001, 0.99999]. This follows the p-value scheme of the R
  package kSamples, which dsc-core's R script calls.
- **Correction (DQ-3).** Bonferroni on the level, not on the p-values: two
  algorithms count as equal on a problem when p ≥ α / m, with m = k(k − 1)/2
  over all k algorithms of the request, and α the request's `method.alpha`.
  The equality relation includes each algorithm with itself.
- **Transitivity (DQ-4).** The relation "equal" is checked to be an
  equivalence: whenever A ~ B and B ~ C, also A ~ C. If it is, the
  algorithms fall into disjoint groups of equal algorithms. If it is not, the
  test results are dropped for that problem and the algorithms are ranked by
  their sample means alone.
- **Rank values (DQ-5).** Groups (or, without transitivity, single
  algorithms) are ordered by ascending sample mean. Each gets the average of
  the positions it spans: two equal algorithms in first place both get 1.5.
  Without transitivity, algorithms with exactly equal means share the average
  of their positions. dsc-core gives a group the mean of one of its members,
  taken from an unordered set, so the member is arbitrary.
- [DSCExample] (KS, α = 0.05, ε = 0, 15 runs each) has these rank patterns
  over its 22 problems: (1, 2.5, 2.5) eleven times, (1, 2, 3) six times,
  (1.5, 1.5, 3) three times and (2, 2, 2) twice. On problem f1 two
  algorithms reach 0 in every run and share rank 1.5: identical samples are
  equal.
- The response gives one rank per (problem, algorithm) as a double
  ([DSCTool2022] p. 4). It contains no p-values.
- With more than four algorithms the authors recommend comparisons with a
  control instead ([Eftimov2020] p. 6). Neither the documentation nor
  dsc-core limits k.

Decided (minimal):

- **DS-9. The pairwise tests.** KS: scipy's two-sided `ks_2samp`, exact
  when n_a · n_b < 10 000 and asymptotic otherwise, on the data as given (no
  jitter). With this rule and the rest of step 2, an independent check with
  scipy 1.18.1 reproduced all 22 problem rankings of [DSCExample]. AD: the
  standardised statistic T of [Scholz1987] with midranks (scipy's
  `anderson_ksamp`, `midrank=True`, which returns exactly this T). The
  p-value is the upper tail of T's limit distribution: p = 1 − F(1 + σ∞ T),
  where F is the limiting Anderson–Darling distribution, computed as in
  [Marsaglia2004], and σ∞² = 2(π² − 9)/3 its variance [Scholz1987]. p is
  held in [0.00001, 0.99999] like the service's table. Between p = 0.1 and
  p = 0.0001 this agrees with the service's table to within 2 % of p, so
  decisions differ only for p-values within that margin of α / m. scipy's own
  `anderson_ksamp` p-value is not used: it is capped to [0.001, 0.25], which
  would wrongly equate pairs once α / m < 0.001 (more than ten algorithms at
  α = 0.05).
- **DS-10. Edge cases (DQ-6).** Only problems that every algorithm has enter
  the analysis, matched by name (DS-2). Every sample needs at least two
  values; otherwise the call fails and names the algorithm and problem (the
  AD variance is undefined for fewer than four values in total, and an empty
  sample has no mean). Two samples that are equal as multisets, constant or
  not, have p = 1, as the service's ranks show. k = 1 is refused, as in
  IOHanalyzer (`RT_DSCinterface.R` 189–193). A group's mean is the mean of
  its members' sample means.

## Practical DSC (`epsilon`, `monte_carlo_iterations`)

Settled:

- pDSC separates practical from statistical significance ([Eftimov2020]
  p. 2). Its threshold ε is chosen by the user. With ε = 0 the scheme reduces
  to standard DSC ([Eftimov2020] p. 4).
- **Sequential pDSC (DQ-7).** dsc-core works on each pair of algorithms
  separately, on fresh copies of the two samples. It walks through the runs
  in their given order and compares run j of one algorithm with run j of the
  other. When the absolute difference is strictly below ε, both values are
  replaced by their midpoint. The pairwise test then runs on the modified
  samples. ε is absolute, in the units of the samples: evaluations (after the
  PAR-k substitution) on the runtime page, negated function values on the
  value page. The same algorithm's sample can therefore be modified
  differently against each opponent.
- The service takes one `epsilon` per request, for all problems
  ([DSCTool2022] p. 3), although [Eftimov2020] calls the threshold problem
  specific (p. 4). IOHanalyzer passes it unscaled for every (function,
  dimension) pair.
- **Switches (DQ-9).** In dsc-core, `monte_carlo_iterations` > 0 selects the
  Monte Carlo branch whatever ε is; ε ≠ 0 switches the midpoint step on.
  Since KS and AD ignore the order of a sample, permutations with ε = 0
  change nothing, and the paper's statement that ε = 0 is standard DSC holds
  in principle.
- **Monte Carlo pDSC (DQ-8, DQ-10).** The idea is to repeat the comparison
  of each pair over random reorderings of the runs, so that the result no
  longer depends on the run order ([Eftimov2020] pp. 2, 4). dsc-core shuffles
  each sample without a seed, so repeated requests can differ. Its 2019 code
  does not do what the paper describes: the midpoint step is applied to the
  unshuffled copies, while the test runs on the shuffled samples, and the
  share of significant iterations is computed in integer arithmetic, so the
  rule that picks the final p-value (the most frequent p-value among the
  significant or among the non-significant iterations) ends in most cases
  at p = 0, which marks the pair as different. Whether service 1.5 fixed
  this is not documented.
- Unequal sample sizes under pDSC (DQ-11): dsc-core pairs runs up to the
  first sample's length, and fails when the second sample is shorter.
- Monte Carlo iterations dominate the service's run time ([Eftimov2020]
  p. 6). The API has no seed parameter ([DSCTool2022] p. 3).

Decided (minimal):

- **DS-11. pDSC.** Sequential pDSC as dsc-core does it, with runs paired up
  to the shorter sample's length (the extra runs of the longer sample are
  left as they are). ε = 0 gives DSC whatever the iteration count. Monte
  Carlo pDSC, for ε > 0 and I > 0 iterations: in each iteration, shuffle the
  run order of both samples independently (seeded), apply the midpoint step,
  and test. The pair's p-value is the median of the I p-values, so the pair
  counts as different exactly when more than half of the iterations say so
  at α / m. The median is the simplest rule that follows the paper's aim; it
  is not the service's rule, which the 2019 code does not implement as
  described. IOHanalyzer's default is 0 iterations, so this matters only
  when a user asks for it. No service comparison is planned for Monte Carlo
  pDSC.

## Step 3: omnibus test (`omnibus`)

Settled:

- The request carries the rank response's `ranked_matrix`,
  `number_algorithms` and `parametric_tests`, plus `method = {name, alpha}`
  ([DSCTool2022] pp. 6–7).
- **Valid methods (DQ-12).** dsc-core offers, for k = 2, the Wilcoxon
  signed-rank test (`wilcoxon-signed-rank`) or, when `parametric_tests` is
  true, the paired t-test (`paired-t`); for k ≥ 3, `friedman`,
  `friedman-aligned-rank` and `iman-davenport` in that order, or `ANOVA`
  when `parametric_tests` is true. Service 1.5 names the two-algorithm test
  `wilcoxon-signed-rank-less` and describes it as one-sided: the algorithm
  with the lower mean against the one with the higher mean ([DSCTool2022]
  p. 7). [DSCExample] (k = 3) lists `friedman`, `friedman-aligned-rank`,
  `iman-davenport`.
- **Parametric check (DQ-13).** dsc-core collects each algorithm's ranks over
  the problems and tests them for normality with a one-sample KS test
  against the standard normal N(0, 1), not against a fitted normal, at the
  rank request's α. If every algorithm passes, Levene's test for equal
  variances decides at the same α. Since every rank is at least 1, the KS
  distance is at least Φ(1) ≈ 0.84, and the exact p-value is then at most
  2(1 − Φ(1))ⁿ, which is 0.008 for n = 3 problems. At α = 0.05 the check
  therefore fails for every n ≥ 3. [DSCExample] has `parametric_tests:
  false`.
- **Too few problems (DQ-14).** dsc-core has no such case; service 1.5
  returns the message "Not enough problems for multiple-problem analysis!"
  in `valid_methods` ([DSCTool2022] p. 4). The threshold is not documented.
- **Statistics (DQ-15).** The data are the k × n matrix of DSC ranks, with
  algorithms in name order.
  - `friedman`: χ²_F = 12n / (k(k + 1)) · (Σ_j R̄_j² − k(k + 1)²/4), R̄_j the
    mean rank of algorithm j, with no correction for ties; p from χ² with
    k − 1 degrees of freedom; `t` = χ²_F. [DSCExample]: χ²_F =
    9.909090909090903, p = 0.007051284680703929.
  - `iman-davenport`: F_ID = (n − 1) χ²_F / (n(k − 1) − χ²_F); p from the F
    distribution with k − 1 and (k − 1)(n − 1) degrees of freedom; `t` = F_ID
    ([Derrac2011]).
  - `friedman-aligned-rank`: subtract each problem's mean rank from its
    ranks, rank all kn aligned values together, and compute the aligned-ranks
    statistic of [Garcia2010] and [Derrac2011] from the column and row sums
    of these ranks; p from χ² with k − 1 degrees of freedom; `t` is the
    statistic. (dsc-core ranks the negated aligned values; the statistic is
    the same either way.)
  - `wilcoxon-signed-rank` (dsc-core): Commons Math's exact two-sided
    p-value, with zero differences kept in the ranking, and an error above
    30 problems; `t` = 0. Service 1.5's renamed one-sided test (above)
    suggests this changed.
  - The service takes p as 1 minus the CDF, so p-values below about 1e-16
    come out as 0.
- **`algorithm_means` (DQ-16).** The mean DSC rank of each algorithm over the
  problems, in algorithm-name order, whatever the method. [DSCExample]:
  BSif 2.3181818181818183, BSqi 2.227272727272727, CMA-MSR
  1.4545454545454546. The post-hoc step uses only these means.
- **Input checks (DQ-17).** α only sets `message`: "rejected" when p ≤ α,
  "not rejected" when p > α. The method name must belong to the family that
  k and the request's `parametric_tests` select; otherwise the response
  carries the error text in `message` and no p-value. The service does not
  recompute `parametric_tests`.

Decided (minimal):

- **DS-12. Methods offered.** `parametric_tests` is always false, so only the
  non-parametric methods exist: for k = 2 the one-sided Wilcoxon test of
  service 1.5, for k ≥ 3 Friedman, Friedman aligned ranks and
  Iman–Davenport, as specified above. The paired t-test and ANOVA are not
  implemented: by the bound above, the service can offer them only when
  n ≤ 2 or α < 2(1 − Φ(1))ⁿ (0.008 for n = 3, 0.0013 for n = 4), and
  IOHanalyzer only ever passes the service's list through.
- **DS-13. Wilcoxon (k = 2).** scipy's `wilcoxon` on the differences
  d = r(lower-mean algorithm) − r(other) (equal means: the first by
  name), `alternative="less"`, zero
  differences dropped (scipy's default), scipy's default choice between the
  exact and normal p-value. If every difference is zero, p = 1. `t` = 0, as
  the service documents for tests without a statistic. Name
  `wilcoxon-signed-rank-less`.
- **DS-14. Too few problems.** With n < 2 there are no omnibus methods and
  the result says "Not enough problems for multiple-problem analysis!"; the
  ranking is still returned. n = 2 is the smallest number of problems for
  which every method above is defined (Iman–Davenport has
  (k − 1)(n − 1) degrees of freedom).

## Step 4: post-hoc test (`posthoc`)

Settled:

- Every algorithm is compared with one control algorithm, `base_algorithm`,
  which must be one of the algorithms in `algorithm_means` ([DSCTool2022]
  p. 8). An unknown name is an error.
- The request carries `algorithm_means` from the omnibus response, `k`, `n`,
  `base_algorithm` and `method = {name, alpha}` ([DSCTool2022] p. 8). The
  documentation says the method must match the omnibus test: `friedman` after
  Friedman or Iman–Davenport, `friedman-aligned-rank` after aligned ranks.
- **z-value (DQ-18).** z_a = (m_0 − m_a) / SE, with m_0 the control's mean
  rank and m_a the other algorithm's. SE = √(k(k + 1)/(6n)) for `friedman`
  and SE = √(k(n + 1)/6) for `friedman-aligned-rank`, the comparisons with a
  control of [Derrac2011]. Both use the mean DSC ranks of the omnibus
  response, also for aligned ranks.
- **One-sided (DQ-19).** The unadjusted p-value is Φ(z), the lower tail. A
  small p-value means the control has a significantly lower, that is better,
  mean rank than the other algorithm. IOHanalyzer's heat map
  (`RT_DSCinterface.R` 143–150) is therefore right: (i, j) is marked when i
  beats j, and subtracting the transpose gives +1 and −1.
- **Adjusted values (DQ-20).** The control is left out; the k − 1 p-values
  are adjusted as R's `p.adjust` does: Holm step-down [Holm1979] and
  Hochberg step-up [Hochberg1988], both capped at 1. All four lists have the algorithms in
  the same order, which IOHanalyzer relies on (it reads them by position).
- **Unchecked combinations (DQ-21).** dsc-core does not check the post-hoc
  method against the omnibus test, nor whether the omnibus test rejected. It
  uses the aligned-ranks SE when the name is `friedman-aligned-rank` (or
  `FAR`) and the Friedman SE for any other name, including after a Wilcoxon
  or parametric omnibus test.
- [DSCExample]: control CMA-MSR, k = 3, n = 22, `friedman`. z: BSif
  −2.864357773488755, BSqi −2.56284642891099. Unadjusted: 0.0020892787840832367,
  0.005190897894850896. Holm and Hochberg: 0.004178557568166473,
  0.005190897894850896. These follow from the formulas above to the printed
  precision.

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
v1.4 URL on port 8443. On 10 November 2026 an empty, unauthenticated POST to
`rank` under each of the three base URLs answered HTTP 401: the service is
up and needs an account.

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
  omnibus test rejects. The service does not object (DQ-21).
- `n` is the number of rows of the target table, not the number of problems
  in the ranked matrix. They differ when an algorithm lacks a problem, and
  then the service also pairs the remaining problems wrongly (step 1).
- The "not enough problems" message is offered as an omnibus choice
  (`RT_DSCinterface.R` 20–21) and then sent as a method name.
- The `epsilon` parameter is documented as being for "eDSC"
  (`R/stats.R:983`). In the sources ε belongs to pDSC, and eDSC is the
  multivariate service ([Eftimov2020] p. 4).

iohinspector against the service (all deliberate):

- KS ties are not jittered (DS-9). The ranks of [DSCExample] are unchanged.
- AD p-values come from the limit distribution instead of the interpolated
  table (DS-9): decisions can differ only when p is within about 2 % of
  α / m.
- Groups are ordered by the mean of their members' means, not by an
  arbitrary member's mean (DS-10).
- Samples with fewer than two values are refused; identical samples have
  p = 1 by rule (DS-10).
- Monte Carlo pDSC uses the median rule and a seed (DS-11).
- `parametric_tests` is always false; no paired t-test or ANOVA (DS-12).
- Problems are matched by name, not position (DS-2).

## Decisions

- **DS-1.** iohinspector implements DSC locally (no network), as the plan's
  I18 decides.
- **DS-2. Inputs.** As step 1, with the run's budget taken from run metadata
  (fixing the PAR-1/PAR-10 NA defect), fixed-budget values oriented so that
  smaller is better, and default targets from TG-4 (B1 not ported). A
  fixed-budget run without a value at the budget (the budget is below its
  first evaluation) is an error that names the problem, instead of an NA
  sent to the ranking. Only problems that every algorithm has enter the
  analysis, matched by name, and the dropped problems are reported, because
  the omnibus tests are tests for paired samples ([Eftimov2020] Fig. 1,
  p. 5).
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
- **DS-4. Ranking.** Step 2 as settled: all pairs, KS or AD (default AD,
  α = 0.05, as in Shiny), p ≥ α / m means equal, the equivalence check, mean
  order and average-position ranks, with the tests of DS-9 and the edge cases
  of DS-10. iohinspector warns when k > 4 ([Eftimov2020] p. 6) but does not
  refuse.
- **DS-5. Practical DSC.** One ε for all problems and an iteration count, as
  the service and IOHanalyzer have it (Q-DS-5), with the rules of DS-11.
  Monte Carlo pDSC takes a seed.
- **DS-6. Omnibus.** The service's method names, statistics, `message` rule
  and `algorithm_means` (step 3), restricted by DS-12 to DS-14. A method
  name outside the valid list is an error.
- **DS-7. Post-hoc.** z, the one-sided unadjusted p-value, Holm and Hochberg
  as settled in step 4. The post-hoc statistic follows from the omnibus
  method (`friedman` for Friedman, Iman–Davenport and, for k = 2, Wilcoxon;
  `friedman-aligned-rank` for aligned ranks) instead of a free choice. The
  k = 2 case keeps IOHanalyzer's page working, which calls `friedman` there.
  One call returns the four value lists for every algorithm as control,
  which is the k × k matrix the Shiny page builds from k calls. n and k come
  from the ranked matrix. The omnibus result is returned with it; the
  post-hoc is not refused when the omnibus test does not reject, but the
  result says so.
- **DS-8. Names.** Result objects keep the service's field names
  (`ranked_matrix`, `valid_methods`, `parametric_tests`, `algorithm_means`,
  `ZValue`, `UnadjustedPValue`, `Holm`, `Hochberg`), so that tests compare
  with [DSCExample] field by field.
- **DS-9 to DS-14**: the minimal decisions in steps 2 and 3 and under
  practical DSC.

## Optional recordings for I18 step 2

The two rows below concern behaviour of service 1.5 that dsc-core 1.4 does
not show. They need an account, which a person registers (a **HUMAN** step;
agents never read IOHanalyzer's stored credentials). If the service cannot
be reached or no account is available, I18 implements DS-13 and DS-14 as
written and this page is the reference. A recording that disagrees changes
the decision only for the point it covers, recorded on this page first.
Each recording names the base URL and service version.

| ID | Point | Cases that decide it |
|---|---|---|
| DQ-12, DQ-15 | Name, sidedness and p-value of the two-algorithm test (DS-13) | k = 2, n = 10 and 30, one algorithm better on most problems; compare with scipy `wilcoxon` one- and two-sided |
| DQ-14 | Smallest n with omnibus methods (DS-14) | k = 2 and 3, n = 1, 2, 3, 5 |

## Open questions for the user

None. Answered or decided:

- Q-DS-1 (answered): the user supplied [Eftimov2020] and [DSCTool2022].
  [Eftimov2017] and [Eftimov2019] were not supplied and no open copy was
  found; dsc-core settles what they would have.
- Q-DS-2 (answered): the two PAR-k conventions are unified, user-customisable
  with a sensible default. Decided by the D6 review (dvermetten, 2026-11-06).
  The default is k = 10, matching the answer to Q-RT-1, and the definition is
  the summary's (RT-3). See DS-3. Remaining point for H1: the shared
  definition uses one cutoff per run set rather than each run's own budget; the
  review did not say which of the two to keep.
- Q-DS-3 (answered): the user allows reading [dsc-core] to settle the
  algorithm, with no copying or translation of its code into iohinspector.
  This page describes it in its own words.
- Q-DS-4 (decided): the example files state no licence, so they are not
  committed. This page cites their numbers as facts. I18 tests may download
  them at test time from the pinned Wayback Machine snapshots and skip when
  offline; the numbers cited here are the committed reference.
- Q-DS-5 (decided): no. One ε per call, as the service and IOHanalyzer do.

## Test tolerance

- Ranks, rank groups, `valid_methods`, `parametric_tests`, the omnibus
  `message` and post-hoc significance at α match [DSCExample] exactly.
- Omnibus `p_value` and `t`, and post-hoc `ZValue`, `UnadjustedPValue`,
  `Holm` and `Hochberg`: closed forms, so relative error 1e-12 against an
  independent computation of the formulas, and 1e-9 against the numbers of
  [DSCExample]. Below p = 1e-10 compare with an absolute 1e-15, because the
  service computes upper tails as 1 minus the CDF.
- Pairwise p-values do not appear in any response. KS p-values are compared
  with scipy (relative 1e-12). AD p-values are tested at the end points of
  their range and against known quantiles of the limit distribution
  ([Marsaglia2004]).
- Monte Carlo pDSC: identical results for the same seed, ε = 0 gives the
  DSC ranks, and the median rule is tested on fixed lists of p-values. No
  service comparison.
