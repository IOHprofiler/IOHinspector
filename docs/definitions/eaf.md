# Empirical attainment function (EAF) and EAF differences

Status: **proposed**, for sign-off at checkpoint H1.
Parity-matrix row: "EAF and EAF differences".
Implemented in: I17.
Sources: [IOHanalyzer], [iohinspector], the R package `eaf` and the Python
package `moocore` (same maintainer); see [references](references.md).

## Definitions

### Single-objective EAF

A run i attains the point (b, y) when its best-so-far value at budget b is at
least as good as y: V_i(b) ≤ y (minimisation). The EAF of a run set is

    α(b, y) = (1/n) Σ_i 1[ V_i(b) ≤ y ],

the fraction of runs that reached quality y within b evaluations
[GrunertdaFonseca2001, LopezIbanez2010, LopezIbanez2025]. Treating
(evaluations, f) as two objectives to minimise, the runs' best-so-far traces
are their attainment sets, and the EAF is the bi-objective EAF of those sets.
Under maximisation f is negated before and after.

### Attainment surfaces (levels)

The k-th attainment surface is the boundary of { (b, y) : α(b, y) ≥ k/n }.
IOHanalyzer shows the surfaces at the percentages 0, 10, …, 100 (11 levels)
by default, as computed by `eaf::eafs(..., percentiles = ...)`; I17 must check
that `moocore` maps percentages to levels in the same way before using it.

### EAF difference

For two run sets A and B, Δ(b, y) = α_A(b, y) − α_B(b, y). It is reported
either exactly, as the rectangles where Δ is constant (`eafdiff` with
`rectangles = TRUE`), or on a grid of 50 log-spaced budgets × 50 log-spaced
values. With more than two algorithms, each algorithm is compared with the
pointwise maximum of the others: Δ_a = α_a − max_{c ≠ a} α_c.

### Several functions

Raw f values of different functions are not comparable, so a multi-function
EAF uses the normalised quality q of [ecdf](ecdf.md) (EC-5, bounds per pair)
instead of f. Its area is the AOCC (equivalence in [ecdf](ecdf.md)).

## How IOHanalyzer implements it

- `generate_data.EAF` (`R/DataSetList.R` 2020–2086): values from
  `get_FV_sample` at 100 subsampled budgets (`seq_RT`, linear or log,
  2045–2053); runs of different functions get distinct run ids
  (`max_runs * funcId + run`, 2066), so raw f values of different functions
  are pooled as if they were comparable; maximisation negates f around
  `eaf::eafs(quals, runs, percentiles = seq(0, 100, length.out = n_sets))`
  with `n_sets = 11` (2068–2079).
- `generate_data.ECDF_From_EAF` (2102–2163): the area under the EAF between
  user bounds (optionally log10, values clipped) per budget, normalised by
  the bound range: an EAF-based ECDF on the 11 levels.
- `generate_data.EAF_Difference` (2164–2186): `eaf::eafdiff(x, y, rectangles
  = TRUE)` on the FV samples at every logged evaluation of the two sets.
- `generate_data.EAF_diff_Approximate` (2205–2250): 50 log-spaced budgets ×
  50 log-spaced values; for each algorithm the success rate `ps` of
  `get_RT_summary(ds, y, xval)` (inclusive, budget cap xval) minus the
  pointwise maximum over the other algorithms.
- Shiny: the multi-function EAF page passes `xmax = input$EAF.Multi.Min`
  (`inst/shiny-server/server/EAF_plots.R:168`), so its upper budget is the
  lower one ("the multi-function EAF uses xmax = Min" in the rebuild plan).

## How iohinspector 0.0.8 implements it

`metrics/eaf.py`:

- `get_discritized_eaf_single_objective` (9–72): `turbo_align` on 10
  log-spaced integer budgets (no best-so-far; see [alignment](alignment.md)),
  quality normalised with `transform_fval(lb=1e-8, ub=1e2)` and the
  maximisation flag not passed, then the fraction of runs with normalised
  quality ≥ each of 101 levels in [0, 1].
- `get_eaf_data` (75–107): aligned data on 50 budgets for plotting.
- `get_eaf_pareto_data` (110–135) and `get_eaf_diff_data` (138–174) call
  `moocore.eaf` and `moocore.eafdiff(rectangles=True)`; identical inputs give
  an empty frame.

## Differences

| Item | IOHanalyzer | iohinspector 0.0.8 | Proposed |
|---|---|---|---|
| Quantity | raw f | normalised log quality | raw f per pair; q across pairs |
| Best-so-far | yes | not in `turbo_align` | always |
| Levels | 11 percentages via `eaf` | 101 quality levels | 11 percentages, configurable |
| Difference | exact rectangles, or 50 × 50 grid | moocore rectangles | both |
| Multi-function | raw f pooled; xmax bug | – | q per pair; bug not ported |
| Maximisation | negated | not passed in the discretised EAF | negated |

## Proposed decisions

- **EA-1.** α(b, y) on best-so-far values, with levels at 0, 10, …, 100 % by
  default, computed with `moocore` once its level convention is checked
  against `eaf` (I17).
- **EA-2.** EAF differences exactly (rectangles) and on the 50 × 50 log grid;
  with more than two algorithms, against the pointwise maximum of the others.
- **EA-3.** Multi-function EAFs use the normalised quality q with fixed
  bounds per pair.
- **EA-4.** The Shiny `xmax = Min` bug is not ported.

## Open questions for the user

- Q-EA-1: For the multi-function EAF, are the observed ranges per pair the
  right default bounds, or should bbob data use 1e-8 and 1e2 as COCO does?

## Test tolerance

Attainment surfaces and difference rectangles must match `eaf`'s output
exactly (they are data points). Grid differences agree to 1e-12.
