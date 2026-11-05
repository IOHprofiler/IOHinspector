# Target and budget grids

Status: **proposed**, for sign-off at checkpoint H1.
Parity-matrix row: "Target and budget grid generation".
Implemented in: I10 (`iohinspector.targets`).
Sources: [IOHanalyzer], [cocopp], [iohinspector]; see [references](references.md).

Targets are the f values a fixed-target view asks about; budgets are the
evaluation counts a fixed-budget view asks about. The ECDF, ERT tables,
ranking, portfolio and DSC views all take a target table: one or more
targets per (function, dimension) pair.

## Definitions

### Observed range

For a selection of run sets on one (function, dimension) pair, F is the set of
every distinct finite objective value logged by any run of any algorithm in
the selection, and E the set of every distinct logged evaluation count. This
is what IOHanalyzer's `get_funvals` and `get_runtimes` return
(`R/DataSetList.R` 806–836).

### `log-linear` and `linear` targets (`seq_FV`)

n targets, default n = 10, evenly spaced between f_lo = min F and
f_hi = max F, both ends included:

- `linear`: t_k = f_lo + k (f_hi − f_lo)/(n − 1), k = 0..n−1;
- `log-linear`: the same in log10, t_k = 10^(log10 f_lo + k (log10 f_hi −
  log10 f_lo)/(n − 1)). It needs f_lo > 0. If min F ≤ 0, f_lo is the
  smallest positive value in F.

Duplicates are removed, so n can shrink when f_lo = f_hi. Targets are
generated **per (function, dimension) pair** from that pair's observed range.

### `bbob` targets

The 51 COCO targets t_k = 10^(2 − k/5), k = 0..50, from 1e2 down to 1e-8,
identical for every pair [Hansen2016pa]. cocopp: `10 ** np.arange(2, -8.2,
-0.2)` (`cocopp/testbedsettings.py:295`); IOHanalyzer:
`rev(seq_FV(c(100, 1e-8), length.out = 51, scale = 'log'))`
(`R/DataSetList.R:1092`). They are precision targets (Δf = f − f_opt) and are
only meaningful when the stored value is Δf (see AL-7 in
[alignment](alignment.md)). iohinspector computes each one as
`10.0 ** (2 - k / 5)` with integer k so that decade targets such as 1e-8 are
the exact float64 value of the literal.

### One target per pair (`get_target_dt`)

Some views (multi-function ERT, Glicko-2, DSC, portfolio) need a single
default target or budget per pair:

- fixed target: the 2 % quantile (type 7) of each algorithm's final values,
  then the best of these over the algorithms. Under minimisation that is
  min over algorithms of the 2 % quantile; under maximisation it is the
  max over algorithms of the **98 %** quantile, the mirrored rule;
- fixed budget: max E, the largest logged evaluation count.

### Budget grids (`seq_RT`)

n budgets, default 50 for plots, evenly spaced (linear or log10) between
min E and max E, both ends included, **rounded to the nearest integer** (ties
to even, as R's `round` and numpy's `round`), duplicates removed.

### Custom targets

A user-supplied table (function, dimension, target). The website's single
target editor (plan, section "Edit targets in one place") produces this table;
generators only fill it.

## How IOHanalyzer implements it

- `seq_FV` (`R/stats.R` 209–285): `from`/`to` default to min/max of the
  values (217–222). If no scale is given it guesses: linear when a bound is
  negative, log when |log10(mean) − log10(median)| > 1, else linear
  (229–235). Log scale: `from <- max(1e-12, from[from > 0])` (244), so a
  minimum ≤ 0 becomes 1e-12, an arbitrary floor marked TODO in the source.
  The ends are always appended (`c(from, ., to)`, 279). The result goes
  through 10^log10(x); in the worked example of [ecdf](ecdf.md) the end
  target 1e-4 came back exact, but this is not guaranteed in general.
- `seq_RT` (`R/stats.R` 304–363): the same for evaluation counts, rounded
  with `round` (360–363).
- `get_ECDF_targets` (`R/DataSetList.R` 1083–1112): per (function,
  dimension): `bbob` 51 targets, or `seq_FV(get_funvals(subset), 10, log or
  linear)`.
- `get_target_dt` (`R/stats.R` 529–556), with `get_target_FV` (558–568) and
  `get_target_RT` (572–574):
  - `get_target_FV` takes the 2 % column of `get_FV_summary(dsList, Inf)` and
    returns its max when maximising (or when the `maximization` attribute is
    missing) and its min otherwise (559–567). Under maximisation the 2 %
    quantile is a **poor** final value, so the default maximisation target is
    one that almost every run of the best algorithm reaches, while the
    minimisation target is reached by about 2 % of them. This asymmetry is
    not ported (TG-4).
  - **Inverted argument.** `get_target_dt(which = 'by_FV')` returns f
    targets and `'by_RT'` returns budgets. IOHanalyzer uses both conventions
    for `which`: `generate_data.Heatmaps` and `glicko2_ranking` take
    `'by_FV'` to mean fixed target, as `get_target_dt` does, but
    `generate_data.ECDF`, `generate_data.Aggr` and `convert_to_dsc_compliant`
    take `'by_RT'` to mean fixed target. The last two pass their `which`
    straight to `get_target_dt` when no targets are given
    (`R/DataSetList.R:1679`, `R/stats.R:912`), so their default fixed-target
    view uses a budget as an f target. The Shiny pages pass explicit values; the DSC page passes
    `"by_RT"` for its runtime view (`inst/shiny-server/server/RT_DSCinterface.R:200`),
    which is the "DSC default targets are swapped" bug B1.

## How cocopp implements it

- The 51-target ECDF grid (`cocopp/testbedsettings.py:295`), plus smaller
  fixed sets for single-target figures and tables (`testbedsettings.py`
  293–304).
- Runlength-based targets (`RunlengthBasedTargetValues`,
  `cocopp/pproc.py` 253–559): per (function, dimension) the target that a
  reference algorithm (best 2009) reaches within a given budget factor times
  the dimension. Used in COCO's "expensive" setting. Not part of IOHanalyzer
  and not on the parity list.
- Hitting times are computed on an internal grid of 5 levels per decade and
  a requested target is rounded to the next stricter level (see
  [alignment](alignment.md)).

## How iohinspector 0.0.8 implements it

`get_sequence` (`metrics/utils.py` 11–52) is the only generator:

- log scale asserts `min > 0` (32), so a run that reaches f = 0 makes the
  default target grid fail;
- the step is computed with `np.arange` (42–47), so the end point depends on
  floating-point accumulation;
- `cast_to_int=True` truncates with `np.array(values, dtype=int)` (51), so
  10^2 computed as 99.99… becomes 99. On a run set whose largest evaluation is
  100 the ECDF grid ends at 99, and on one ending at 10 it ends at 9 (checked
  with `get_data_ecdf` on the three-run example of [ecdf](ecdf.md));
- targets are 50 log-spaced values between the data's global min and max,
  over all functions in the frame together (`metrics/fixed_target.py:45`),
  not per (function, dimension);
- no `bbob` grid, no per-pair default target.

## Differences

| Item | IOHanalyzer | cocopp | iohinspector 0.0.8 | Proposed |
|---|---|---|---|---|
| Default ECDF targets | 10 log-linear per pair | 51 bbob | 50 log over all data | 10 log-linear per pair |
| Range | observed per pair | fixed | observed, global | observed per pair |
| Log scale with min ≤ 0 | floor 1e-12 | – | assertion error | smallest positive observed value |
| Budget rounding | round | – | truncate | round (half to even) |
| bbob grid | `seq_FV` of 1e2..1e-8 | `10 ** arange` | none | 10^(2 − k/5), exact decades |
| Default target per pair, maximisation | max of 2 % quantiles | – | – | max of 98 % quantiles |
| `which` naming | inverted in defaults | – | – | explicit `perspective` argument |

## Proposed decisions

- **TG-1.** Generators: `log-linear` (default, 10 per pair), `linear`,
  `bbob` (51) and custom tables. This answers the plan's question: log-spaced
  by default, per problem, with the COCO 51-target grid available.
- **TG-2.** Ranges come from the observed values of the current selection,
  per (function, dimension) pair, including both ends.
- **TG-3.** Under log spacing a minimum ≤ 0 is replaced by the smallest
  positive observed value, not by 1e-12.
- **TG-4.** The default single target per pair mirrors under maximisation
  (98 % quantile, max over algorithms).
- **TG-5.** Budget grids are rounded to the nearest integer, never truncated,
  and always include max E.
- **TG-6.** Functions take an explicit `perspective = "fixed_target" |
  "fixed_budget"` argument; IOHanalyzer's inverted `which` default and bug B1
  are not ported.

## Open questions for the user

- Q-TG-1: TG-3 changes IOHanalyzer's target grid whenever a run reaches
  f ≤ 0. Is the smallest positive observed value acceptable, or should the
  1e-12 floor stay for parity?
- Q-TG-2: Should cocopp's runlength-based targets be offered after launch?

## Test tolerance

Generated targets and budgets must match the golden grids exactly, except
where TG-3, TG-4 or TG-5 apply; those tests name the decision. Log-linear
targets are compared with a relative tolerance of 1e-15 because R and numpy
may round 10^x differently in the last bit.
