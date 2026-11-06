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
  ([Eftimov2020] p. 10; [DSCTool2022] p. 11), so rank 1 is best. That the
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
- In IOHanalyzer, ε is in the units of the samples: evaluations (after the
  PAR-k substitution) on the runtime page, negated function values on the
  value page. The same ε applies to every (function, dimension) pair.
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
