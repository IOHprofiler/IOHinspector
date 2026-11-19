# Empirical attainment function (EAF) and EAF differences

Status: **proposed**, for sign-off at checkpoint H1; the open questions were
answered by the D6 review (dvermetten, 2026-11-06).
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

### Two modes: precise and discretised

Every EAF output, the EAF and the EAF difference alike, comes in two modes
(EA-5):

- **precise**: the true EAF, as `moocore` computes it, valid at any budget and
  any value: the attainment surfaces, and for differences the rectangles where
  Δ is constant. Nothing is sampled.
- **discretised**: both the evaluation axis and the target (value) axis are
  discretised, so the EAF becomes a matrix α[b_l, y_m] of 50 log-spaced budgets
  by 50 log-spaced values by default (IOHanalyzer's difference grid), each
  adjustable. The EAF difference is the difference of two such matrices.

### EAF difference

For two run sets A and B, Δ(b, y) = α_A(b, y) − α_B(b, y). It is reported
either exactly, as the rectangles where Δ is constant (`eafdiff` with
`rectangles = TRUE`), or on a grid of 50 log-spaced budgets × 50 log-spaced
values.

### EAF margin (more than two groups)

With more than two algorithms there is no single difference. Each algorithm
is compared with the pointwise maximum of the others, its *margin*:
M_a = α_a − max_{c ≠ a} α_c. M_a > 0 where a attains more often than every
other algorithm (by its lead over the next best) and M_a < 0 where another
attains more often (by how far a is behind the best). This is a comparison of
several EAFs rather than a difference of two, so it has its own name and
function; for two algorithms it equals the difference. It is discretised only
(an exact version would need every group's EAF on the union of all
coordinates).

### Several functions

Raw f values of different functions are not comparable, so a multi-function
EAF uses the normalised quality q of [ecdf](ecdf.md) (EC-5, bounds per pair)
instead of f. Its area is the AOCC (equivalence in [ecdf](ecdf.md)). The
default bounds are 10² and 10⁻⁸ on Δf = f − f_opt for BBOB data (AL-7), as in
COCO's grid; for other data they are the observed range of each pair. The user
can set both.

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
| Modes | EAF: levels; difference: exact or 50 × 50 grid | moocore; discretised version of the EAF | precise and discretised, for the EAF and the difference |
| Multi-function | raw f pooled; xmax bug | – | q per pair; bug not ported |
| Maximisation | negated | not passed in the discretised EAF | negated |

## Decisions

- **EA-1.** α(b, y) on best-so-far values, with levels at 0, 10, …, 100 % by
  default, computed with `moocore` once its level convention is checked
  against `eaf` (I17).
- **EA-2.** The EAF difference is of exactly two groups, in the two modes of
  EA-5 (rectangles, and the discretised matrix with a 50 × 50 log grid by
  default). More than two groups are compared by the EAF margin, each against
  the pointwise maximum of the others, discretised only. Decided by the review
  of #45 (dvermetten, 2026-11-19): the multi-group comparison is not a
  difference of EAFs and is a separate function with a clearer name.
- **EA-3.** Multi-function EAFs use the normalised quality q with fixed
  bounds per pair. For BBOB data the default bounds are 10² (upper) and 10⁻⁸
  (lower) on Δf; otherwise the observed range per pair. Decided by the D6 review (dvermetten, 2026-11-06).
- **EA-4.** The Shiny `xmax = Min` bug is not ported.
- **EA-5.** Two modes, for both the EAF and the EAF difference. *Precise*: the
  true EAF as `moocore` returns it, at any budget. *Discretised*: evaluation
  and target axes are both discretised so the EAF is a matrix; default 50
  log-spaced budgets × 50 log-spaced values, both axes adjustable. Decided by the D6 review (dvermetten, 2026-11-06).

## Questions answered by the D6 review

- Q-EA-1 (default bounds for the multi-function EAF): for BBOB data 10² and
  10⁻⁸. See EA-3.
- Added requirement: precise and discretised modes, for the EAF and the EAF
  difference. See EA-5.

Remaining points for H1: the default 50 × 50 grid for the discretised mode is
IOHanalyzer's difference grid, chosen because the review named no size; the
review did not say how BBOB data is recognised (proposed: the suite name in the
run metadata). The same bounds apply to AOCC through EC-5 (see
[ecdf](ecdf.md)).

## Implementation (I17)

`iohinspector.metrics`, in `metrics/eaf.py`:

- `get_eaf(data, by, mode=...)`: the EAF of each group of `by` (EA-1). Each
  run's attainment set is its best-so-far trace: its first row and every row
  where the best-so-far value improves; before its first row a run has
  attained nothing (AL-5). `mode="precise"` gives the attainment surfaces at
  `percentiles` (default 0, 10, …, 100) from `moocore.eaf`;
  `mode="discretised"` gives α on a grid of budgets × values (EA-5): 50
  log-spaced budgets over each pair's logged evaluations and 50 log-spaced
  values over its logged values by default (`eval_values`, `f_values`: a
  count or the values themselves).
- `get_eaf_difference(data, by, mode=...)`: Δ of EA-2 for exactly two
  groups, once for each (Δ_B = −Δ_A), in the same two modes. Precise: the
  rectangles of `moocore.eafdiff` where Δ is constant. Discretised: Δ on the
  grid of `get_eaf`. Other numbers of groups raise `ValueError`.
- `get_eaf_margin(data, by, **options)`: the margin M_a of EA-2 for two or
  more groups, column `eaf_margin`, on the grid of `get_eaf`; for two groups
  it is the discretised `get_eaf_difference`.
- Several pairs (EA-3): when a group of `by` spans more than one (function,
  dimension) pair, f is replaced by the normalised quality q of the continuum
  ECDF (EC-5) in a column `quality`, with the same default bounds
  (`targets.get_ecdf_targets`, so the suite bounds of TG-8 apply), and the
  discretised value grid is linear in q over [0, 1]. Each run is one attainment
  set, so a pair weighs in proportion to its runs. With `by` naming the pair
  columns every group has one pair and the values stay f; differences are then
  taken within each pair.
- The EA-4 bug cannot occur: there is no separate budget bound for the
  multi-function EAF.

Checks of `moocore` 0.1.8 and 0.3.2, as EA-1 requires (the golden tests
repeat them for the installed version):

- On IOHanalyzer's own inputs (each run's best-so-far value at 20 budgets of
  the pair) `moocore.eaf` returns IOHanalyzer's surfaces exactly, so it maps
  percentages to levels as `eaf::eafs` does: the surface at p % is attained by
  at least ⌈p n / 100⌉ of the n runs, and by at least one. With 5 runs, 0, 10
  and 20 % are the same surface.
- `moocore.eaf` reads a set from consecutive rows; interleaved rows give a
  wrong EAF. iohinspector sorts by run first. The 0.0.8 functions
  `get_eaf_pareto_data` and `get_eaf_diff_data` did not, and now do.
- `moocore.eafdiff` assumes both sides have equally many sets (it splits the
  sets in half), and counts Δ in sets. With n_A ≠ n_B runs each side's runs
  are repeated up to lcm(n_A, n_B), and Δ is the count divided by that.
- `moocore.eafdiff` fails when there is no rectangle at all; that is an empty
  result. Rectangles with Δ = 0 are kept, as `eaf::eafdiff` keeps them.

Golden tests (`tests/test_metrics/test_eaf.py`) on `ioh040_min`,
`ioh040_max`, `ioh040_cons` and `ioh040_pos`:

- `EAF`, `EAF_logx`: equal, except one value of `ioh040_min` in the 17th
  digit, because IOHanalyzer takes a run's final best value from the JSON
  summary rather than the `.dat` row.
- `EAF_Difference`: equal on the minimisation fixtures. IOHanalyzer's
  `generate_data.EAF_Difference` ignores maximisation (it always minimises), so
  on `ioh040_max` the test reproduces it with `maximization=False` and checks
  that the agreed result differs.
- `EAF_diff_Approximate`, which with more than two algorithms is the margin
  (`get_eaf_margin`): equal to 1e-12 on IOHanalyzer's grid. Its value grid
  differs from `seq_fv` at the ends (`10^log10(x)` is 1 ulp off; a lower end of
  0 becomes 1e-12, TG-3), which matters where a grid value equals a logged
  value, so the test passes IOHanalyzer's grid. Under maximisation IOHanalyzer
  labels the values in reverse order, because `get_RT_summary` returns the
  success rates sorted by target; the test undoes that.
- `ECDF_From_EAF` (the EAF-based ECDF) is not ported; the continuum ECDF of
  [ecdf](ecdf.md) is that view.

## Test tolerance

Attainment surfaces and difference rectangles must match `eaf`'s output
exactly (they are data points). Grid differences agree to 1e-12.
