# Cumulative difference plot

Status: **proposed**, for sign-off at checkpoint H1; the open questions were
answered by the D6 review (dvermetten, 2026-11-06).
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
values negated under maximisation. The samples must have the same size: when
they differ, the larger one is subsampled (CD-3).

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

Y_A and Y_B are the *quantile random variables* of [Arza2023] (checked in
I16 against the arXiv version, 2203.07889): random variables on [0, 1] whose
densities are mixtures of uniform kernels, one per observation of the pooled
sample, placed by the observation's rank, so that the two densities add up to
2 everywhere. Tied observations share their kernels in proportion to how many
come from each sample. In the paper ties are values closer than a threshold,
as here. RVCompare evaluates these densities on the 2000 points, normalises
each by its sum (not by its trapezoid integral, so the cumulative functions
end just below 1, by less than 1e-3) and integrates with the trapezoid rule.
The paper's bands are the bootstrap bands above, with level sqrt(1 − α) per
band for joint confidence 1 − α. This page follows RVCompare's code.

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

## Decisions

- **CD-1.** Port RVCompare 0.1.8's bootstrap estimator as specified above,
  with a seed.
- **CD-2.** Failed runs count as +∞ (the worst value), not NA.
- **CD-3.** Unequal sample sizes: by default the larger sample is subsampled
  without replacement to the size of the smaller one, with the caller's seed.
  The user is always told: the view and the download state both original
  sizes, the size used and which runs were dropped (`run_id`s). `subsample=False`
  restores the error with a clear message. Decided by the D6 review (dvermetten, 2026-11-06).
- **CD-4.** ε defaults to 1e-80 as in IOHanalyzer.

## Questions answered by the D6 review

- Q-CD-1 (subsample unequal run counts): yes, subsample by default, and show
  it to the user. See CD-3.

Remaining points for H1: the subsample is drawn by seed from the larger set
only (the smaller set is used whole). Confirm, or ask for repeated subsamples.

## Implementation (I16)

`iohinspector.metrics`, in `metrics/cdp.py`:

- `cumulative_difference(x_a, x_b, *, alpha=0.05, epsilon=1e-80,
  n_bootstrap=1000, seed=None)`: RVCompare 0.1.8's estimator (CD-1) on two
  equally large samples, smaller is better, vectorised over the bootstrap
  samples. Infinite values tie with each other (CD-2); NaN is rejected. Returns
  `p` and `y_a`, `y_b`, `diff` with their `_lower` and `_upper` bands.
- `get_cumulative_difference(data, *, target=None, budget=None, by, maximization,
  subsample=True, seed=None, ...)`: one (function, dimension) pair, `by` makes
  exactly two groups (A first in sorted order). Fixed target: runtimes, a failed
  run is `inf` (CD-2). Fixed budget: the best value at the budget, negated when
  maximising; a run with no evaluation yet at the budget (AL-5) is `inf` too.
  Unequal groups: the larger is subsampled with the seed and a warning names the
  sizes (CD-3). The result has `curves` and `runs`: every run with its group,
  sample value and `used`, so the view and the download can state the original
  sizes, the size used and the `run_id`s left out.

RVCompare's code is CC0, so it is ported directly.

## Test tolerance

With the observed samples (the first "bootstrap" sample) the estimates
Y_A, Y_B and the difference must match RVCompare to 1e-12. Bands depend on the
random stream and are compared as distributions: the band edges from 1000
seeded bootstraps must lie within 0.02 of RVCompare's for Y_A and Y_B, and
within 0.04 for the difference, whose band varies about twice as much between
seeds (up to 0.037 between two of our own seeds). The I03 fixtures have 5 runs
per algorithm, too few for their bands to be compared across random streams,
so only their estimates are compared (to 1e-12); the bands are compared on
`tests/golden/rvcompare/`, RVCompare itself on two samples of 100 values.
