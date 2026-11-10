# Alignment: hit rule, best-so-far, runtime and value samples

Status: **proposed**, for sign-off at checkpoint H1; the open questions were
answered by the D6 review (dvermetten, 2026-11-06).
Parity-matrix rows: "Runtime and value samples"; the alignment part of every
fixed-target and fixed-budget row.
Implemented in: I10 (`iohinspector.align`).
Sources: [IOHanalyzer], [cocopp], [iohinspector]; see [references](references.md).

Every other page builds on two functions defined here: the first hitting time
T_i(t) of a target (fixed-target view) and the best-so-far value V_i(b) at a
budget (fixed-budget view) [Doerr2018, Wang2022].

## Notation

- Run i has logged rows j = 1..m_i with evaluation counts e_ij
  (non-decreasing) and objective values f_ij, in source order. Storage format
  v1 keeps every logged row (Principle 1), so these are the raw values, not a
  best-so-far trace.
- B_i is the number of evaluations run i used (run metadata; see
  [runtime](runtime.md)).
- "Better" means smaller when minimising and larger when maximising. The
  direction comes from the run set's `maximization` flag. Below, ≤ and min are
  written for minimisation; under maximisation read ≥ and max.

## Definitions

### Best-so-far

    V_i(e) = min { f_ij : e_ij ≤ e }    for e ≥ e_i1;   undefined for e < e_i1

It is computed during analysis from all logged rows, never at load time. If
several rows share an evaluation count, all of them count (their best wins),
as IOHanalyzer's `dcast(..., fun.aggregate = min/max)` does
(`R/readFiles.R:1565`).

### Hit rule

A run **reaches** target t at row j when f_ij ≤ t (minimisation) or
f_ij ≥ t (maximisation). The comparison is **inclusive** and exact on the
stored float64 values.

### First hitting time (fixed-target alignment)

    T_i(t) = min { e_ij : f_ij ≤ t },   or ∞ if no row reaches t

Equivalently T_i(t) = min { e : V_i(e) ≤ t }. Ties resolve to the **first** row
that reaches t: later rows with the same value never change T. T_i is
non-increasing in the difficulty of t: a harder target is reached no earlier
than an easier one.

### Value at a budget (fixed-budget alignment)

    V_i(b) = best-so-far at evaluation min(b, B_i)

- **b ≥ e_i1 and b ≤ B_i**: the best value among rows with e_ij ≤ b.
- **b > B_i** (the run stopped before b): the run's final best value is
  **carried forward**. The fixed-budget summary reports how many runs had
  B_i ≥ b (see [fixed-budget](fixed-budget.md)).
- **b < e_i1** (before the first logged row): **undefined** (null). It is not
  replaced by +∞, −∞ or any value of the run. In IOHexperimenter data
  e_i1 = 1, so this only arises for b < 1 or for formats that start logging
  later.
- **The final evaluation is always kept.** The aligned table keeps, for every
  run, a row at B_i (from run metadata) even when the last improvement came
  earlier. Its best-so-far value is the run's final best value, and a flag
  `is_final` marks it. A run that stopped (B_i < b) and a run that is still
  running but has not improved since some evaluation (B_i ≥ b) are then
  distinguishable, which the fixed-budget restart option needs
  ([fixed-budget](fixed-budget.md), FB-6).

### Runtime and value samples

The runtime sample of a run set at target t is the table of T_i(t), one row
per run, with null where the run never reached t. The value sample at budget b
is the table of V_i(b), one row per run. Both are tidy tables with the run
identifiers (`run_id`, instance) and are what the website offers for download.

### Targets relative to the optimum

The hit rule applies to the stored objective value. For COCO data, cocopp and
IOHanalyzer compare targets with the precision Δf = f − f_opt
(`cocopp/dataformatsettings.py` 44–46; IOHanalyzer reads COCO's
"best noise-free fitness − Fopt" column). For COCO data, targets therefore
apply to Δf; how Δf is stored or derived is decided by the COCO reader (I07).

## How IOHanalyzer implements it

- **RT matrix** (fixed target). `c_impute_running_time` (`src/align.cc`
  76–107) walks each run's best-so-far column and each target in order of
  difficulty, and records the first row j with `index[j] <= FV[i]`
  (`>=` when maximising): the loop advances while `FV[i] < index[j]`
  (lines 91–95). For old IOHprofiler and COCO `.dat` files the column is the
  best-so-far column 3 (`idxTarget <- 3`, `R/readFiles.R` 760–762); for
  IOHexperimenter JSON data it is the raw `raw_y` column, which gives the same
  first hitting time (`R/readFiles.R` 1594–1600). Unreached targets are NA.
  The matrix rows are every distinct observed value, so any user target falls
  between rows; lookups take the first row whose value is ≤ t
  (`R/DataSet.R` 888–900).
- **FV matrix** (fixed budget). Old `.dat` data: `align_function_value`
  (`R/readFiles.R` 811–874). Contiguous runs are padded with their last value
  (`align_contiguous`, 716–731); otherwise `c_impute` (`src/align.cc` 47–73)
  carries the last value forward and leaves NA before a run's first row.
  JSON data: `dcast` on the union of evaluation counts, NA replaced by the
  run's worst raw value, then `cummin`/`cummax` (`R/readFiles.R` 1564–1590),
  which also carries values forward after a run ends.
- **Value lookup** at budget b: the last FV-matrix row with evaluation ≤ b
  (`rev(idx[r >= RT])[1]`, `R/DataSet.R:1096` and `:1147`); NA when b is below
  the first row.
- **The ECDF is the exception**: it counts a target as reached only when
  f < t (`temp < targets_`, `R/DataSetList.R:1409`; `temp > targets_` when
  maximising, `:1405`), and in its fixed-budget form only when T < b
  (`:1419`). The rest of IOHanalyzer uses ≤. This is the "strict comparison"
  R-core finding in the rebuild plan.

## How cocopp implements it

- Hit rule `<=`, with a relative tolerance: a row counts if
  `f <= target or is_close(f, target)`, `is_close` having `rel_tol = 1e-9`
  (`cocopp/readalign.py` 249, 252 and 545–546).
- Hitting times are computed on a fixed grid of levels 10^(i/5) for bbob.
  `detEvals` returns the row of the closest grid level **not larger** than the
  requested target (`cocopp/pproc.py` 1912–1916), so a target between grid
  levels is rounded to the next stricter level.
- Fixed budget: the last row with evaluations ≤ the budget; a run that
  stopped earlier keeps its last value (cocopp report, section 7).

## How iohinspector 0.0.8 implements it

- `align_data` (`align.py` 8–67) computes `cum_min`/`cum_max` per run and then
  an as-of join. Fixed target: `join_asof` on the f value, strategy
  `backward` when minimising and `forward` when maximising (43–55); unreached
  targets become `inf`.
- **Ties under minimisation.** The backward join takes the *last* row among
  rows with equal best-so-far, so T is overestimated whenever a run logs rows
  that do not improve. Checked on a run with f = 10, 5, 5, 5, 2 at
  evaluations 1 to 5: `align_data` and `turbo_align` both return T(6) = T(5) = 4,
  the correct value is 2 (polars 1.44.2). The maximisation mirror returns the
  correct 2. The loader hides this by deleting every row whose best-so-far
  equals the previous row, with a global `raw_y.diff()` filter that also
  crosses run boundaries (`data.py` 213–219), a listed known defect.
- **Fixed budget.** Backward join on evaluations, then `fill_null(np.inf)`.
  Before the first logged row this gives `inf`, which under maximisation is
  the best possible value (checked: a run starting at evaluation 5 has value
  `inf` at b = 2 when maximising).
- `turbo_align` (`align.py` 70–128) does **not** compute best-so-far
  ("Assumes the data is monotonic!"). On raw rows f = 10, 3, 7 it returns 7 at
  b = 3 instead of 3.
- Under polars 2.1.0, `align_data` with a fixed-target grid fails with "argument
  in operation 'asof_join' is not sorted" (handled in I01).

## Differences

| Item | IOHanalyzer | cocopp | iohinspector 0.0.8 | Proposed |
|---|---|---|---|---|
| Hit rule | ≤ (ECDF: <) | ≤, rel. tol. 1e-9 | ≤ (max: ≥) | ≤ everywhere, exact |
| Target between levels | exact | rounded to stricter grid level | exact | exact |
| Ties (equal best-so-far rows) | first row | first row | last row (min), first (max) | first row |
| Value after a run stopped | carried forward | carried forward | carried forward | carried forward, count reported |
| Value before first row | NA (JSON data: worst value) | – | `inf` | null |
| Best-so-far | from data / `cummin` | from data | `cum_min` (not in `turbo_align`) | always computed |

## Decisions

- **AL-1.** The hit rule is inclusive, f ≤ t (minimisation) and f ≥ t
  (maximisation), in every metric, the ECDF included. This answers the plan's
  first question. IOHanalyzer's strict ECDF comparison is not ported
  (see [ecdf](ecdf.md)).
- **AL-2.** Comparisons are exact on float64. Targets are used as given, not
  snapped to a grid.
- **AL-3.** Ties resolve to the first row that reaches the target.
- **AL-4.** A run that stopped before budget b contributes its final best value
  at b (carried forward).
- **AL-5.** Before a run's first logged row its value is null.
- **AL-6.** Best-so-far is always computed at analysis time from the raw rows;
  no metric assumes monotone input.
- **AL-7.** For COCO data targets apply to Δf = f − f_opt.
- **AL-8.** No tolerance is added to the hit rule for COCO-derived data;
  cocopp's relative 1e-9 is not ported and AL-2 holds for every source. A
  tolerance can be offered later as an explicit option. Decided by the D6 review (dvermetten, 2026-11-06).
- **AL-9.** Alignment keeps every run's final evaluation: a row at B_i with the
  run's final best value and an `is_final` flag, whether or not the run improved
  there. It is what lets fixed-budget views separate "the run stopped" from
  "the run did not improve" (FB-6). Decided by the D6 review (dvermetten, 2026-11-06).

## Questions answered by the D6 review

- Q-AL-1 (cocopp's 1e-9 tolerance for COCO targets): no additional
  tolerances for now. See AL-8.
- Added by the same comment thread (fixed-budget Q-FB-1): the last evaluation
  stays present in the aligned data. See AL-9.

Remaining points for H1: none.

## Test tolerance

Hitting times and counts must match exactly. Values at a budget are stored
values and must match exactly. Golden tests against IOHanalyzer's RT and FV
matrices (I03) compare on the IOHanalyzer row grid; the ECDF tests name AL-1
where the strict comparison changes a value.
