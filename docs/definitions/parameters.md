# Parameters, positions and constraint penalties

Status: **proposed**, for sign-off at checkpoint H1; the open questions were
answered by the D6 review (dvermetten, 2026-11-06).
Parity-matrix rows: "Parameters, correlations, positions" and "Constraint
penalties". These rows are on the parity matrix but not named in I02's step
list; they are covered here because I14 depends on them.
Implemented in: I14.
Sources: [IOHanalyzer], [iohinspector]; see [references](references.md).

## Notation

- Each logged row of run i may carry extra columns: algorithm parameters
  (for example a step size σ), positions x0…x(d−1) (stored apart in
  `positions.parquet`), and a constraint `violation` v_ij, a signed real number
  as logged (see "Constraint violations in IOHexperimenter" below). These
  columns need not be monotone.
- j*(i, t) is the row at which run i first reaches target t, the row of
  T_i(t) ([alignment](alignment.md)); j°(i, b) is the last row with
  e_ij ≤ b.

## Definitions

### Parameter value at a target or a budget

- At target t: the parameter's value on row j*(i, t), the row where the run
  first reached t. Undefined (null) when the run never reaches t.
- At budget b: the value on row j°(i, b), the most recent logged row at or
  before b (an as-of join backward in evaluations). After the run stopped
  (b > B_i) the last value is carried forward, as for V_i(b) (AL-4). Before
  the first row it is null.

Both are lookups on the raw rows; no alignment matrix is built.

### Parameter summary

At each target or budget, over the runs with a defined value: `n`, `mean`,
`median`, `sd` (denominator n − 1) and quantiles with Hyndman–Fan type 7 at the
nine IOHanalyzer probabilities, as for function values
([fixed-budget](fixed-budget.md)).

### Correlation and parallel-coordinate data

The raw logged rows of two columns (two parameters, or a parameter and the
raw f value), matched on (run, evaluation). No coefficient is computed; the
website draws the scatter. This is what IOHanalyzer provides.

### Positions

The logged positions of a run, joined to the trace on (`run_id`, `row`). They
are shown per instance and per run, unaligned.

### Constraint penalties

A penalised value g_ij = g(f_ij, v_ij) replaces f_ij **before** best-so-far,
hitting times and every metric are computed. With factor c ≥ 0, exponent
p > 0 (default 1) and the sign s = +1 when minimising and −1 when maximising (a
penalty always makes a value worse). A row is **infeasible** when the
feasibility rule says so: by default v ≠ 0 (`feasible = "zero"`), so negative
violations are supported and count as infeasible; `feasible = "nonpositive"`
(v ≤ 0 is feasible) reproduces IOHanalyzer's `y <= 0` test. The size of the
violation is |v|:

| Method | g(f, v) |
|---|---|
| ignore | f |
| relative | f + s · c · \|v\|^p if infeasible, else f |
| absolute | f + s · c if infeasible, else f |
| set to value | f if feasible, else c |

With p = 1 and v ≥ 0 this is IOHanalyzer's relative penalty on infeasible
rows. With general c and p it reproduces IOHexperimenter's own penalty
`weight · violation^exponent` (where the sign s is the user's choice of
direction, see below).

#### Constraint violations in IOHexperimenter

Read from `source/iohexperimenter` at 1e44e067 (the sources; no logged data
was needed). This answers Q-PA-1.

- A constraint has an enforcement `Enforced` (`include/ioh/problem/constraints.hpp`
  9–18): `NOT` (nothing computed; the logged violation is always 0), `HIDDEN`
  (violation computed, y not penalised), `SOFT` (y + penalty, the penalties of
  all constraints summed), `HARD` and `OVERRIDE` (y replaced by the penalty).
  Only `HIDDEN` and `NOT` leave the logged y unpenalised.
- The penalty is `weight · pow(violation, exponent)` (line 112) and is **added**
  to y (`y + penalty()`, lines 106 and 215), whatever the optimisation
  direction. For a maximisation problem a positive weight therefore improves y;
  the sign is the user's weight.
- `BoxConstraint` (lines 283–336): the violation is the sum over coordinates of
  the squared distance outside the bounds, so v ≥ 0 and 0 means feasible.
- `FunctionalConstraint` (lines 370–397): the violation is whatever the user's
  function returns, and the point is infeasible when `abs(violation) > 0`
  (line 392). So **negative values are possible and infeasible**, and zero is
  feasible.
- `GraphConstraint` (`problem/submodular/graph_problem.hpp` 245–278) and
  `PWTConstraint` (`problem/submodular/pack_while_travel.hpp` 195–225), both
  `HARD`: the "violation" is the load (the weighted sum of the selected
  items), which is positive also for feasible points; the point is infeasible
  only when the load exceeds the limit or capacity, and the penalty is
  `weight · (limit − load [+ penalty term])^exponent`, so it can be negative.
  For these problems v = 0 does **not** mean feasible, and the logged y is
  already penalised.
- The logger's `violation` and `penalty` properties
  (`include/ioh/logger/properties.hpp` 314–377) log index 0 by default, the
  **sum** over all constraints of the set (`logger/loginfo.hpp` 91–92); index i
  logs one constraint.

Consequences: the logged y already contains the penalty unless the constraint
was `HIDDEN` or `NOT`, so the post-hoc penalties here are meant for those;
`ignore` is right for `SOFT`, `HARD` and `OVERRIDE` data. The default rule
`feasible = "zero"` is correct for box and functional constraints; load-type
violations (graph, pack-while-travel) need the user to give their threshold
(a callable `feasible`). The reader (I05) must keep the signed violation
unchanged.

## How IOHanalyzer implements it

- **Storage.** Old `.dat` data align parameters with the RT and FV matrices
  (`align_running_time` and `align_function_value`, `R/readFiles.R`
  754–874): by target, the value on the hitting row; by budget, carried
  forward. IOHexperimenter JSON data use `dcast` instead
  (`read_IOH_v1plus`, `R/readFiles.R` 1612–1626): `by_RT` on
  `evaluations ~ runnr`, and `by_FV` on `raw_y ~ runnr`, a matrix indexed by
  the exact f value each run logged.
- **Lookup.** `get_PAR_summary.DataSet` (`R/DataSet.R` 1178–1251) and
  `get_PAR_sample.DataSet` (1253–1316) pick a row of that matrix with
  `op <- ifelse(maximization || which == 'by_RT', >=, <=)` (1205–1206,
  1282–1283). By budget this is the first row with evaluation ≥ b, the
  *next* logged row, while function values use the last row ≤ b. Summaries
  use `.mean`, `.median`, `.sd` and `C_quantile` (type 7) over non-NA values.
- **Defects on JSON data**, reproduced with IOHanalyzer's own functions on a
  two-run file with a parameter column `p` (R 4.3.3, sources from the
  reference copy):
  - the `by_FV` `dcast` has no `fun.aggregate`; when a run logs the same f
    value twice, data.table falls back to `length`, and every entry of the
    matrix becomes a count (0, 1, 2) instead of a parameter value;
  - the row index is computed on the RT-matrix row names (sorted from worst to
    best) but applied to the `by_FV` matrix (sorted ascending by f), so it
    reads the row of a different f value. In the test, targets 6 and 4.5
    returned the rows for f = 9 and f = 12;
  - even with matching rows, only runs that logged exactly that f value have a
    non-NA entry.
  Parameter-by-target views of IOHexperimenter JSON data are therefore not
  reliable in IOHanalyzer, and golden tests against them must be limited to
  old-format data and to by-budget lookups at logged evaluations.
- `generate_data.Parameters` (`R/DataSetList.R` 1577–1620): summaries on 50
  grid points from `seq_RT` or `seq_FV`, plus mean ± sd bands.
  `generate_data.Parameter_correlation` (1622–1660): raw (runtime, run) pairs
  of two columns, merged; f(x) only when the raw FV matrix was kept
  (`contains_full_FV`).
- `get_position_dsl` (`R/DataSetList.R` 1802–1854): positions from the
  `by_RT` parameter matrices x0…x(d−1) per instance, with the FV column, and
  it drops the last row when the evaluation counts are not contiguous (1824).
- **Penalties.** The Shiny settings page sets
  `IOHanalyzer.Violation_Function` (`inst/shiny-server/server/settings_page.R`
  78–87; choices in `inst/shiny-server/ui/settings.R` 68–72, default factor
  1): Ignore `x`; Penalize Relative `x + factor * y`; Set to value
  `ifelse(y <= 0, x, factor)`; Penalize Absolute `x + factor`.
  `get_FV.DataSet` and `get_RT.DataSet` (`R/DataSet.R` 1332–1373) apply it to
  the raw value matrix `FV_raw_mat` and the `violation` column, then recompute
  the running best (`cummin`/`cummax`) and the RT matrix. So the penalty is
  applied before best-so-far, as proposed. Two defects: "Penalize Absolute"
  adds the factor to **every** row, feasible or not; and penalties are always
  added, which under maximisation makes an infeasible value better.

## How iohinspector 0.0.8 implements it

No parameter, position or constraint functions. Extra logged columns are
read but not analysed; positions are used only by the single-run heat map
and trajectory helpers (`metrics/single_run.py`, `metrics/trajectory.py`),
which are outside the parity list.

## Differences

| Item | IOHanalyzer | iohinspector 0.0.8 | Proposed |
|---|---|---|---|
| Parameter at budget | next logged row (≥ b) | – | last row ≤ b, carried forward |
| Parameter at target (JSON data) | wrong rows, counts | – | value on the hitting row |
| Feasibility | `y <= 0` is feasible (`violation` column) | – | v = 0 (IOHexperimenter's rule); `nonpositive` option |
| Penalty "absolute" | added to every row | – | only on infeasible rows |
| Penalty under maximisation | added (improves value) | – | subtracted |
| Penalty then best-so-far | yes | – | yes |

## Decisions

- **PA-1.** Parameter at target = value on the first row that reaches the
  target; at budget = value on the last row at or before the budget, carried
  forward after the run stops; null otherwise.
- **PA-2.** Parameter summaries use type 7 quantiles over defined values.
- **PA-3.** Correlation views return raw matched rows, no coefficient.
- **PA-4.** Penalties are applied to raw values before best-so-far, with the
  four methods above; "absolute" penalises only infeasible rows and every
  penalty makes the value worse under both directions. A row is infeasible when
  v ≠ 0 by default, so negative violations are supported (`feasible` selects
  other rules); the penalty size uses |v|. Decided by the D6 review (dvermetten, 2026-11-06).
- **PA-5.** IOHanalyzer's JSON parameter-by-target lookup is not ported; the
  golden tests name PA-1.
- **PA-6.** The signed violation is stored and passed through as logged,
  including negative values. Its meaning follows IOHexperimenter's constraint
  types, as documented above: 0 is feasible for box and functional constraints,
  and the load-type violations of the graph and pack-while-travel problems need
  a user-given threshold. Logged y values of `SOFT`, `HARD` and `OVERRIDE`
  constraints already contain the penalty. Decided by the D6 review (dvermetten, 2026-11-06).

## Questions answered by the D6 review

- Q-PA-1 (values a constraint violation can take): IOHexperimenter's sources
  were checked, see "Constraint violations in IOHexperimenter". Negative
  values are possible (functional constraints) and are supported. See PA-4 and
  PA-6.

Remaining points for H1: the default feasibility rule v ≠ 0 and the size |v|^p
(with p = 1), which keep "a penalty never improves a value" while supporting
negative violations; the review asked only that negative values be supported.
Whether load-type violations need a different default is for the user to see.

## Test tolerance

Looked-up values are stored values and must match exactly. Summaries must
agree to a relative error of 1e-12.
