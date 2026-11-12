# ECDF, aggregation, AUC, AOCC and multi-function ranks

Status: **proposed**, for sign-off at checkpoint H1; the open questions were
answered by the D6 review (dvermetten, 2026-11-06).
Parity-matrix rows: "ECDF and AUC as in IOHanalyzer", "Multi-function
aggregation and ranks"; AOCC from the "New" row.
Implemented in: I12.
Sources: [IOHanalyzer], [cocopp], [iohinspector]; see [references](references.md).

## Notation

- Run set S (one algorithm, one (function, dimension) pair p), n runs.
- Targets t_1..t_m for pair p, from [targets](targets.md).
- T_i(t): first hitting time ([alignment](alignment.md)); V_i(b): best-so-far
  at budget b.
- P_A: the pairs on which algorithm A has runs.

## Definitions

### Target-based ECDF of one run set

    ECDF_S(b) = (1 / (n m)) Σ_i Σ_k 1[ T_i(t_k) ≤ b ]

the fraction of (run, target) pairs reached within b evaluations
[Hansen2016pa, Hansen2022, Wang2022]. Both comparisons are inclusive: the
target is reached when f ≤ t (AL-1), and "within b" means T ≤ b, "not later
than a given runtime" in the words of [LopezIbanez2025]. ECDF_S is a step
function of b. It changes only at the finite hitting times, so it is computed
exactly at those points and sampled on a grid only for display.

### Aggregation over pairs

    ECDF_A(b) = (1 / |P|) Σ_{p ∈ P} ECDF_{A,p}(b)

with P the set of pairs that **every** compared algorithm has (the common
pairs; EC-2). When the algorithms' pairs differ, a clear warning names the
pairs that were left out and for which algorithm. A parameter `pairs = "all"`
averages each algorithm over its own pairs, P_A, instead, and the warning is
still raised, because such curves are hard to interpret. Each (function,
dimension) pair has equal weight; inside a pair each target and
each run has equal weight. When every pair has the same number of targets,
which is the default (10 per pair, or the 51 bbob targets), this is the same as
weighting each (pair, target) combination equally, as cocopp does. Runs of
one algorithm on one pair are first collected into one run set, so a pair
split over several files still counts once.

The budget axis is evaluations, optionally divided by the dimension, which is
cocopp's default (`divide_by_dimension = True`, `cocopp/compall/pprldmany.py:49`)
and the BBOB-style figure's axis (I19).

### Area under the ECDF (AUC)

For a budget interval [b_lo, b_hi] on a linear axis,

    AUC_A = (1 / (b_hi − b_lo)) ∫_{b_lo}^{b_hi} ECDF_A(b) db,

and on a log axis the same with log10 b in place of b. The integral of the step
function is computed exactly. AUC ∈ [0, 1]. Defaults: b_lo = 1 and b_hi = the
largest budget in the selection.

### AOCC

Following [LopezIbanez2025]: choose bounds f_lo < f_hi per pair (default the
observed range of [targets](targets.md); for bbob data 1e-8 and 1e2). For a run
under minimisation,

    q_i(b) = ( log10 f_hi − log10 clip(V_i(b), f_lo, f_hi) ) / ( log10 f_hi − log10 f_lo )

and under maximisation the mirror. q_i(b) ∈ [0, 1] is the normalised
quality of the best-so-far value. The **AOCC** of the run set is the mean over
runs of the normalised area under q_i over [b_lo, b_hi]:

    AOCC_S = (1/n) Σ_i (1 / (b_hi − b_lo)) ∫ q_i(b) db

### Equivalence

q_i(b) is exactly the fraction of a continuum of targets, log-uniform on
[f_lo, f_hi], that run i has reached by budget b. So mean_i q_i(b) is the ECDF
over infinitely many log-spaced targets, AOCC_S is its AUC, and the m-target
ECDF and its AUC converge to these as m grows (the target ECDF approximates the
EAF [LopezIbanez2025]). The bounds must be fixed per pair before alignment;
the equivalence does not hold when they are taken from the aligned data.

### Multi-function rank table

Per (function, dimension) pair, algorithms are ranked by ERT at the pair's
target (fixed target; lower is better) or by mean value at the pair's budget
(fixed budget; direction by maximisation). Ties get the average rank; ERT = ∞
values tie at the bottom; missing values rank last. Default targets and
budgets come from `get_target_dt` ([targets](targets.md), TG-4).

## How IOHanalyzer implements it

- `generate_data.ECDF` (`R/DataSetList.R` 1334–1431), fixed-target branch:
  the x grid is 50 points of `seq_RT` over all logged evaluation counts
  (1357). At each x, each run's value from `get_FV_sample` (1401) is compared
  **strictly** with the pair's targets: `mean(temp < targets_)` (1409;
  `>` when maximising, 1405). This is averaged over runs per DataSet (1411)
  and then over all DataSets with the same ID (1430): equal weight per
  DataSet, so per pair when each pair is one DataSet.
- Default targets: `get_ECDF_targets(dsList, "log-linear", 10)`
  (`R/DataSetList.R` 1083–1112), called by the Shiny page
  (`inst/shiny-server/server/RT_ECDF.R:83`).
- `generate_data.AUC` (`R/DataSetList.R` 1454–1515): trapezoid rule between
  consecutive grid points (1492), cumulative sum divided by the current x,
  **not** by x − x_min (1494). A curve that ends before the largest x is
  extended with its last value (1480–1491).
- `ECDF.DataSet`/`AUC.ECDF` (`R/stats.R` 387–402 and 502–515) are an older
  API that integrates R's `ecdf` with `integrate`; the Shiny app uses
  `generate_data.AUC`.
- An EAF-based ECDF exists too (`generate_data.ECDF_From_EAF`,
  `R/DataSetList.R` 2102–2163), on 11 attainment levels by default; see
  [eaf](eaf.md).
- `generate_data.Aggr` (`R/DataSetList.R` 1668–1719): rank by ERT
  (`get_RT_summary`) or by mean (`get_FV_summary`) per (DIM, funcId) with
  `frank(..., na.last = TRUE)` (1717), average ties. Its default targets
  come from `get_target_dt` with the inverted `which` (1679, see
  [targets](targets.md)).

## How cocopp implements it

`compall/pprldmany.py` (cocopp report, section 4, line numbers spot-checked):
per dimension (dimensions are never pooled), per (function, target,
algorithm). With uniform instances (bbob) each (function, target) pair gets a
fixed number of **simulated restarts** of runtimes (`toolsstats.drawSP`,
`cocopp/compall/pprldmany.py:817`), so unsuccessful runs are turned into
longer, restarted runtimes instead of counting as "not reached". Each
(function, target) pair has equal weight. The x axis is evaluations divided by
the dimension. The "x" markers show a budget estimate per function, not a
data point. Simulated restarts are not part of IOHanalyzer or of the parity
list.

## How iohinspector 0.0.8 implements it

- `get_data_ecdf` (`metrics/ecdf.py` 9–88) is **not** a target-based ECDF. It
  aligns best-so-far values on 50 log-spaced integer budgets (46–71), maps
  each value through `transform_fval` (72–80): log10, min-max normalised,
  clipped and reversed for minimisation (`metrics/utils.py` 102–115), and
  averages per algorithm (81–84). That is mean_i q_i(b), the
  continuum-target ECDF, except that:
  - the bounds default to the min and max **of the aligned data**
    (`metrics/utils.py` 97–100), so they depend on the budget grid and on
    every function in the frame;
  - all functions in the frame share one pair of bounds, and runs, not pairs,
    have equal weight;
  - when the aligned values are all equal the bounds coincide and every value
    is null (0/0).
- `get_aocc` (`metrics/aocc.py` 8–84): per run, area under the given column
  over [0, eval_max] with a step rule, divided by eval_max (19–42); the
  column is used as given, so on data loaded without `monotonic=True` it is
  the raw, not the best-so-far, value.
- No multi-function rank table; ranking is covered in [glicko2](glicko2.md).

## Worked example: why 0.67 and 0.90 differ

The plan records only the two numbers, not the input that produced them, so
this section re-derives the gap from the definitions. Both tools' own code
was run: IOHanalyzer's R functions sourced from `source/IOHanalyzer` (R 4.3.3,
`align.cc` compiled with Rcpp, the package itself not installed), and
iohinspector 0.0.8 from the worktree under polars 1.44.2.

**Data.** One algorithm, one function, minimisation, three runs, each logged
in IOHexperimenter format with evaluations 1, 10 and 100 and budget 100:

| Run | f at 1 | f at 10 | f at 100 |
|---|---|---|---|
| 1 | 1e4 | 1e0 | 1e0 |
| 2 | 1e4 | 1e-2 | 1e-2 |
| 3 | 1e4 | 1e-4 | 1e-4 |

**IOHanalyzer.** `get_ECDF_targets` gives 10 targets, log-spaced from 1e-4
to 1e4 (the end target came back as exactly 1e-4). At the final budget the
runs reach 5, 7 and 9 of them under the strict comparison, so the ECDF is
(0.5 + 0.7 + 0.9)/3 = **0.70**. Run 3 reaches the best observed value, but not
the target equal to it.

**Inclusive hit rule (AL-1).** The same targets with ≤ give 0.5, 0.7 and 1.0,
so **0.733**.

**iohinspector.** `get_data_ecdf` on the data loaded with `monotonic=False`
gives q = 0.5, 0.75 and 1.0, so **0.75**: the continuum of targets instead of
10. With `eval_values=[100]` the bounds shrink to the final values
(1e-4..1e0) and the same data gives **0.50**. Loaded with `monotonic=True`
(as in the README), the final rows are deleted, the largest evaluation becomes
10, the truncated grid ends at 9, and every value is **null**.

**Real fixture.** The same comparison at the final budget on the committed
`tests/test_data` (IOHexperimenter 0.3.18 JSON, dimension 2, 5 runs per set):

| Selection | IOHanalyzer | iohinspector |
|---|---|---|
| f1, algorithms A (two files) and B (README example) | A 0.68, B 0.60 | A 0.718, B 0.619 |
| f1, A and B | A 0.62, B 0.60 | A 0.657, B 0.619 |
| f2, A and B | A 0.64, B 0.76 | A 0.652, B 0.821 (`monotonic=True`) / 0.826 (`False`) |
| f1 and f2 together | A 0.63, B 0.68 | A 0.641, B 0.687 / 0.688 |

The two `monotonic` values for f2 differ because with `monotonic=True` the
data ends at evaluation 87, the truncated grid ends at 86, and one run of B
improves at 87.

**Conclusion.** The gap comes from definitions, plus defects that make
iohinspector's value unstable:

1. *Definition:* 10 discrete targets per pair (IOHanalyzer) against a
   continuum of targets between normalisation bounds (iohinspector). With the
   same bounds these converge as the number of targets grows.
2. *Definition, and an IOHanalyzer artefact:* the strict comparison. With
   default targets the best observed value is itself a target, so no run ever
   reaches it, and the ECDF of a run set in which every run ends at the best
   value is exactly 0.90, never 1.0. A reported 0.90 is consistent with this.
3. *iohinspector defect:* bounds taken from the aligned data, shared across
   functions, so the value depends on the budget grid and on which other
   functions are loaded (0.75 against 0.50 above).
4. *iohinspector defects:* the loader drops the final rows and the integer
   grid truncates its last budget, so the curve can end before the last
   improvement or be null.
5. *Weighting:* IOHanalyzer weights pairs equally, iohinspector weights runs
   equally across all functions in the frame.

I12's golden test encodes the three-run example: 0.70 for the IOHanalyzer
definition, 0.733 for the agreed target ECDF (EC-1) and 0.75 for the
continuum ECDF with bounds 1e-4 and 1e4 (EC-5).

## Differences

| Item | IOHanalyzer | cocopp | iohinspector 0.0.8 | Proposed |
|---|---|---|---|---|
| ECDF kind | 10 targets per pair | 51 targets, simulated restarts | continuum of targets | targets (EC-1); continuum kept as AOCC/EAF view |
| Hit rule | strict | ≤ | ≤ | ≤ |
| Weighting | per DataSet | per (function, target), per dimension | per run, all functions | per pair, then target and run |
| Evaluation | 50 grid points | at its sample points | 50 truncated integer budgets | exact step function |
| AUC | trapezoid, divided by x | – | – | exact, divided by b_hi − b_lo |
| AOCC bounds | – | – | from aligned data | fixed per pair |
| Ranks | `frank`, average ties, NA last | – | – | same |

## Decisions

- **EC-1.** The ECDF is target-based, inclusive in f and in budget, with
  targets per pair from TG-1 (default 10 log-linear). The strict comparison is
  not ported.
- **EC-2.** Aggregation weights each (function, dimension) pair equally, then
  each target and each run within the pair. When the compared algorithms do not
  all have the same pairs, only the common pairs are used and a clear warning
  says which pairs were dropped. A parameter (`pairs = "all"`) averages over
  each algorithm's own pairs; it still raises the warning. This answers the
  plan's question. Decided by the D6 review (dvermetten, 2026-11-06).
- **EC-3.** The ECDF is computed exactly at the hitting times; display grids
  only sample it.
- **EC-4.** The AUC is the exact normalised integral over [b_lo, b_hi] on the
  chosen axis. IOHanalyzer's trapezoid and divisor are reproduced only in the
  golden test that names EC-4.
- **EC-5.** AOCC uses best-so-far values and bounds fixed per pair before
  alignment (for BBOB data 10² and 10⁻⁸ on Δf by default, as in EA-3 of
  [eaf](eaf.md), otherwise the observed range per pair); it equals the AUC of the continuum ECDF, and a test checks the
  equivalence numerically.
- **EC-6.** The current `get_data_ecdf` stays available under a clear name
  (for example `ecdf_continuous`), with EC-5's fixed bounds.
- **EC-7.** No simulated restarts.
- **EC-8.** Rank tables rank per pair with average ties and missing values
  last.

## Open questions for the user

- Q-EC-1: Which data gave 0.67 and 0.90? If it can be found, I12 should add it
  as a second golden case.
- Q-EC-3: IOHanalyzer also has a fixed-budget ECDF (`generate_data.ECDF` with
  `which = 'by_FV'`, `R/DataSetList.R` 1414–1424: over a grid of values x, the
  fraction of (run, budget) pairs in which the run reached x in fewer
  evaluations than the budget, `RT < budget`) and its AUC. This page does not
  define it, so I12 does not implement it. Is it needed, and inclusive (T ≤ b)
  as in EC-1?

## Questions answered by the D6 review

- Q-EC-2 (algorithms with different pairs): use only the common pairs, with a
  clear warning, and a parameter that averages everything but still warns. See
  EC-2.

Remaining points for H1: none. EC-5's bounds follow the EAF answer (Q-EA-1);
that is a consequence of AOCC and the continuum ECDF being equal, not a
separate review answer.

## Implementation (I12)

`iohinspector.metrics`, in `metrics/ecdf.py`:

- `get_ecdf(data, targets, by)`: the target ECDF (EC-1), exact at its steps or
  sampled at `x_values` (EC-3), averaged over pairs as in EC-2
  (`pairs = "common" | "all"`), optionally over evaluations / dimension.
  `continuous=True` is the continuum ECDF (EC-6) with the bounds of EC-5 taken
  from the targets: each pair's smallest and largest target. The default
  targets give the observed range per pair; `get_ecdf_targets(data, "bbob")`
  gives the BBOB bounds 1e-8 and 1e2.
- `get_ecdf_auc(...)`: the AUC of EC-4; with `continuous=True` it is the AOCC
  of EC-5. Both integrate one step function exactly, so the equivalence holds
  by construction; a test checks that the AUC over m log-spaced targets is
  within 1/m of the AOCC.
- `get_rank_table(summary, value)`: the ranks of EC-8 for any per-pair
  summary, such as ERT at `get_target_dt`'s targets.
- `get_data_ecdf` and `get_aocc` are deprecated in 0.2.0 (user, 2026-11-12). They
  keep their signatures and 0.0.8 columns and call `get_ecdf(continuous=True)`
  and `get_ecdf_auc(continuous=True)`, so they follow EC-2 and EC-5 (EC-6):
  `get_data_ecdf` has bounds per pair, `get_aocc` integrates its quality column
  over [1, eval_max] with the best-so-far value and 0 before a run's first row.

The golden tests reproduce IOHanalyzer's ECDF and AUC exactly by moving each
target to the next float (`f < t` is `f ≤ nextafter(t, −∞)`), and the worked
example above gives 0.70, 0.733 and 0.75.

## Test tolerance

ECDF values are ratios of counts and must match exactly (as rationals; float
comparison to 1e-12). AUC and AOCC must agree to a relative error of 1e-12.
Comparisons with IOHanalyzer's sampled ECDF and trapezoid AUC are made on
IOHanalyzer's 50-point grid.
