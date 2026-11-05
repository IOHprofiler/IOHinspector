# Cumulative difference plot

Status: **proposed**, for sign-off at checkpoint H1.
Parity-matrix row: "Cumulative difference plot".
Implemented in: I16.
Sources: [IOHanalyzer], [Arza2023], RVCompare 0.1.8 (CRAN, read from the
source tarball); see [references](references.md).

The cumulative difference plot compares the samples of two algorithms
through stochastic dominance [Arza2023]. RVCompare, the authors' R package,
is the reference implementation; IOHanalyzer calls it.

## Definitions

Inputs: one (function, dimension) pair, exactly two algorithms A and B, and a
target (runtimes) or a budget (values). X_A, X_B are the samples, oriented so
that smaller is better: runtimes as they are, with **failed runs as +∞**;
values negated under maximisation. The samples must have the same size.

Following RVCompare's `get_Y_AB_bounds_bootstrap`: the pooled observations
are ranked (values closer than ε, default 1e-20 in RVCompare and 1e-80 in
IOHanalyzer, count as equal), the ranks of each sample define the cumulative
functions of Y_A and Y_B on 2000 points p ∈ [0, 1], and the plot shows
Y_A − Y_B with a bootstrap confidence band:

- 1000 bootstrap samples, the first being the observed samples themselves;
- each bootstrap draws n values with replacement from X_A and from X_B;
- band at level α' = 1 − sqrt(1 − α) per curve, from the α'/2 and 1 − α'/2
  quantiles (type 7) of the bootstrap curves; α default 0.05;
- output per p: Y_A, Y_B and their bands, and the difference with its band.

The precise definition of Y_A and Y_B (the rank-based random variables of
[Arza2023]) is to be checked against the paper in I16; this page fixes the
estimator to RVCompare's code.

## How IOHanalyzer implements it

- `generate_data.CDP` (`R/DataSetList.R` 1942–2001): only for one pair and
  two IDs; fixed target uses `get_RT_sample`, so failed runs are **NA**;
  fixed budget uses `get_FV_sample` and negates under maximisation; then
  `RVCompare::get_Y_AB_bounds_bootstrap(X_A, X_B, ignoreMinimumLengthCheck =
  TRUE, alpha, EPSILON = 1e-80, nOfBootstrapSamples = 1e3)`.
- RVCompare 0.1.8 (`R/RVCompare.R` 312–450) rejects NaN values only
  (`is.nan`, 316), so NA runtimes are passed on; how its rank routine treats
  them is not defined. The length check compares `length(n)` with `length(m)`
  of two scalars (353), so it never fires, and B is resampled with size n
  (378). The bootstrap uses no seed.
- The Shiny pages call `generate_data.CDP(..., isMinimizationProblem = ...)`
  (`inst/shiny-server/server/RTPMF.R:86`, `FCEPDF.R:81`), an argument the
  function does not have and cannot absorb (no `...`), so at HEAD 20e14e4
  these calls stop with "unused argument". I03's pinned build (`caecada^`)
  must confirm whether this already applied there.

## How iohinspector 0.0.8 implements it

No cumulative difference plot.

## Proposed decisions

- **CD-1.** Port RVCompare 0.1.8's bootstrap estimator as specified above,
  with a seed.
- **CD-2.** Failed runs count as +∞ (the worst value), not NA.
- **CD-3.** Unequal sample sizes are an error with a clear message.
- **CD-4.** ε defaults to 1e-80 as in IOHanalyzer.

## Open questions for the user

- Q-CD-1: The plot needs equal run counts. Should the website subsample the
  larger sample (seeded) instead of refusing?

## Test tolerance

With the observed samples (the first "bootstrap" sample) the estimates
Y_A, Y_B and the difference must match RVCompare to 1e-12. Bands depend on the
random stream and are compared as distributions: the band edges from 1000
seeded bootstraps must lie within 0.02 of RVCompare's on the I03 fixtures.
