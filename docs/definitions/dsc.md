# Deep Statistical Comparison (DSC)

Status: **partial specification**, for sign-off at checkpoint H1 and
completion in I18 step 1.
Parity-matrix row: "DSC" (local implementation, decision in the plan's I18).
Implemented in: I18.
Sources: [Eftimov2017], [Eftimov2019], [Eftimov2020], [IOHanalyzer]; see
[references](references.md).

**Limits of this audit.** The three DSC papers are behind publisher pages
that could not be downloaded here (two are open access on the publisher's
site, but its PDF links return an HTML page to scripts), and the DSCtool
service (`ws.ijs.si:8443`) did not answer. This page therefore fixes only what
was verified: the papers' abstracts (OpenAlex), and IOHanalyzer's client code,
which shows the service's inputs, outputs and the client-side preparation.
The step-by-step ranking algorithm is listed as questions for I18, not
guessed. I18 needs the papers' full text (the user can supply the PDFs) and
the recorded service responses (I18 step 2).

## What the papers establish (abstracts)

- **DSC** [Eftimov2017; summarised in Eftimov2020]: per problem, algorithms are
  ranked "by comparing distributions of obtained solutions for a problem
  instead of using a simple descriptive statistic such as the mean or the
  median". For several problems, "an appropriate statistical test must be
  applied to the rankings obtained for a set of problems" (the omnibus test).
- **Practical DSC (pDSC)** [Eftimov2019]: takes practical significance into
  account through preprocessing of the runs. Two variants: **sequential
  pDSC**, which preprocesses the independent runs in a sequential order, and
  **Monte Carlo pDSC**, which avoids the dependence on the order of the runs.
  In DSCtool these appear as the `epsilon` and `monte_carlo_iterations`
  parameters of the ranking service.
- **DSCTool** [Eftimov2020]: REST web services for single-problem and
  multiple-problem analysis.

## Pipeline as used by IOHanalyzer

### 1. Client-side data preparation (`convert_to_dsc_compliant`)

`R/stats.R` 901–973. For every algorithm and every (function, dimension)
pair, a vector of per-run values named `F{funcId}_{DIM}D` (964):

- **fixed target**: the runtime sample at the pair's target. Failed runs are
  NA and are handled by `na.correction` (Shiny default `PAR-10`, the first
  choice in `inst/shiny-server/ui/DSC_interface.R` 155–156):
  - `PAR-1`: the run's own budget (929–932);
  - `PAR-10`: 10 × the run's own budget (933–936);
  - `ERT`: the run set's ERT, or the sum of all budgets when the ERT is
    infinite (937–943);
  - `Remove-na`: the failed runs are dropped (944–945).
- **fixed budget**: the value sample at the pair's budget, negated under
  maximisation so that smaller is always better (955).

Defect: for IOHexperimenter JSON data `maxRT` is a single number
([runtime](runtime.md)), and `budgets[is.na(data)]` then indexes a length-1
vector with a logical vector, which yields NA. PAR-1 and PAR-10 therefore
leave the NA values in place for all JSON data, and the service call fails
(checked in R 4.3.3: `budgets <- 100; data <- c(5, NA, 7, NA);
data[is.na(data)] <- budgets[is.na(data)]` leaves `5 NA 7 NA`).

Defect B1: when no targets are given, `convert_to_dsc_compliant` passes its
fixed-target `which = 'by_RT'` to `get_target_dt` (912), which returns
budgets; the Shiny runtime page does the same explicitly
(`inst/shiny-server/server/RT_DSCinterface.R:200`). See
[targets](targets.md).

### 2. Ranking (`rank` endpoint)

`get_dsc_rank` (`R/stats.R` 1003–1037) posts to `/dsc-1.5/service/rank`:
`method = {name: "AD" | "KS", alpha}`, `epsilon`, `monte_carlo_iterations`
and the data (1014–1023). Shiny defaults: Anderson–Darling, α = 0.05,
ε = 0 and 0 Monte Carlo iterations (`inst/shiny-server/ui/DSC_interface.R`
157–182 for the runtime page, 15–40 for the value page). The response fields the client uses are `ranked_matrix`,
`number_algorithms`, `parametric_tests` and `valid_methods`.

### 3. Omnibus test (`omnibus` endpoint)

`get_dsc_omnibus` (1054–1080): `method = {name, alpha}` with the name taken
from the ranking response's `valid_methods` (first one by default), plus
`ranked_matrix`, `number_algorithms` and `parametric_tests` (1061–1067). The
response's `algorithm_means` feeds the post-hoc step.

### 4. Post-hoc test (`posthoc` endpoint)

`get_dsc_posthoc` (1102–1135): `method = {name, alpha}` with name
`friedman` or `friedman-aligned-rank` (Shiny choices,
`inst/shiny-server/ui/DSC_interface.R` 253–255), `algorithm_means`,
`n` = number of problems, `k` = number of algorithms and a
`base_algorithm` (default the first in `algorithm_means`, 1112). The Shiny
page runs it once per algorithm as base and reads four p-value lists from
`adjusted_p_values`: z, unadjusted, Holm and Hochberg
(`inst/shiny-server/server/RT_DSCinterface.R` 106–125).

## Questions I18 must answer from the papers and the recorded responses

1. Pairwise test per problem: KS and AD as which exact tests (two-sample,
   two-sided?), and which p-value computation (for AD, [Scholz1987] k-sample
   or two-sample; asymptotic or tabulated)?
2. Multiple-comparison handling of the pairwise p-values within a problem
   (for example a Bonferroni-type correction), and the decision rule.
3. The transitivity check on the "same distribution" relation, and what
   happens when it fails.
4. How ranks are assigned: the order of groups (by which statistic), and
   ties (fractional ranks).
5. Sequential pDSC: the exact preprocessing with `epsilon` (on raw values or
   relative?) and the run order it depends on. Monte Carlo pDSC: what is
   resampled per iteration and how the iterations are combined.
6. How the service fills `parametric_tests` and `valid_methods`: the
   normality and equal-variance checks it applies to the ranked matrix and the
   omnibus tests it may offer (the plan names Friedman, Iman–Davenport and
   others; [Garcia2010] and [Derrac2011] describe the usual set).
7. The post-hoc statistics: the z formula for Friedman and Friedman
   aligned ranks with n problems and k algorithms, and the Holm and Hochberg
   adjustments ([Holm1979]; [Garcia2010]).

## How iohinspector 0.0.8 implements it

No DSC.

## Proposed decisions

- **DS-1.** iohinspector implements DSC locally (no network), as the plan's
  I18 decides; DSCtool serves only to record reference responses.
- **DS-2.** Inputs as in step 1 above, with the run's budget taken from run
  metadata (fixing the PAR-1/PAR-10 NA defect), fixed-budget values oriented
  so that smaller is better, and default targets from TG-4 (B1 not ported).
- **DS-3.** Missing-value options PAR-1, PAR-10, ERT and Remove-na, default
  PAR-10 as in Shiny. PAR-k here uses each run's own budget, unlike the
  summary's PAR-k (RT-3), because that is what IOHanalyzer sends.
- **DS-4.** The ranking, omnibus and post-hoc steps are specified in I18 step
  1 from the papers and the recorded responses, answering the questions above.

## Open questions for the user

- Q-DS-1: Can you provide the PDFs of [Eftimov2017], [Eftimov2019] and
  [Eftimov2020] (or the DSCtool API documentation) for I18? Without them the
  ranking algorithm cannot be specified from the source.
- Q-DS-2: DS-3 keeps two PAR-k conventions (own budget for DSC, a common
  cutoff for the summary table). Should they be unified?

## Test tolerance

Set in I18: ranks and test decisions must match the recorded responses
exactly; p-values within a tolerance stated there.
