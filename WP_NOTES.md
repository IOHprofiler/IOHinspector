# WP I33: simulated restarts in every ECDF (on #42; review round of #48)
Branch: wp/I33-fixed-budget-restarts
Status: ready for review

Work package I33 (plan: iohinspector.md), from dvermetten's reviews of #42 and #48.
Metric code: review by dvermetten (D6). Definitions: `docs/definitions/ecdf.md`
("Simulated restarts", EC-7, "Implementation (I33)") and
`docs/definitions/fixed-budget.md` (FB-6). For the reviewer:
`docs/notes/simulated-restarts.md` (the cost problem, its cause and the timings).

## Done
- Round 1 (steps 1 to 4): the fixed-budget ECDF and its AUC restart stopped runs
  by default (`stopped="restart"`, or `"carry_forward"`, IOHanalyzer's rule);
  validation against cocopp's `drawSP` (`tests/golden/cocopp/coco_old/ecdf_restarts_*`).
- Round 2, the review of #48:
  - step 5, target ECDFs: every ECDF restarts by default with COCO's procedure:
    `get_ecdf` and `get_ecdf_auc` (targets and continuum, so the AOCC),
    `get_portfolio_ecdf`/`get_shapley_values`, `get_fixed_budget_ecdf` and its
    AUC. Same option and default everywhere. The deprecated `get_aocc` and
    `get_data_ecdf` keep no restarts, as in 0.0.8. The BBOB-style figure draws
    every curve to the largest budget of all groups (restarted beyond a group's
    own budget, as cocopp); its snapshots are regenerated.
  - step 6, seed: the ECDFs need none (below). `aggregate_convergence`'s restarts
    are seeded with 0 by default.
  - step 7, method: a run that reaches a target is not restarted for it, and the
    ECDF is the expected value over the restarts, computed from the runs'
    runtimes instead of simulated: with F the runs that never reach t,
    E_t(b) = 1/n sum_{i not in F} 1[T_i(t) <= b] + 1/n sum_{j in F} E_t(b - B_j).
    One module, `metrics/restarts.py`, computes it for every ECDF (a pass over
    the integer evaluations in blocks of the smallest B_j, stopped where the
    chance of a further restart is below 1e-17). The fixed-budget ECDF is
    computed from pieces of the runs' traces (one path for both options; the
    IOHanalyzer golden tests check it with `carry_forward`). Simulation remains
    only in `aggregate_convergence`, whose statistics need one value per run;
    there a run with the best final value of its set is not restarted and a
    chain stops once it has drawn one.
  - step 8: `docs/notes/simulated-restarts.md`.

## Validation
- E_t from its definition (a plain recursion in the tests) matches the target
  ECDF, the fixed-budget ECDF and the portfolio game to 1e-12, minimising and
  maximising; with restarts, the AUC over m targets is within 1/m of the AOCC and
  the fixed-budget AUC is the mean continuum ECDF over its budgets; the
  summary's simulated chains average to E_t (30000 chains, within 5 standard
  errors).
- cocopp 2.9.1 (`test_restarts_match_cocopp`), `coco_old` (DIRECT on f6, 2-D and
  3-D), 51 targets x 10 budgets up to 1e6, 510 cells per dimension: each cell
  from `get_ecdf` (one target, at the budgets) and from `get_fixed_budget_ecdf`
  (one budget, at the target); the two agree to 1e-12. Ours is exact, cocopp's a
  sample of 20000, so each cell must be within 5 of cocopp's standard errors.
  Measured: largest difference 0.0054 (2-D) and 0.0056 (3-D), at most 2.6 and 1.6
  standard errors, mean absolute difference 0.00010 and 0.00007; 445 and 466
  cells equal. The fixed-budget ECDF over the ten budgets is within 0.0005 and
  0.0006 of cocopp's (carrying forward: 0.071 and 0.099). Round 1, with 5000
  simulated chains, had 0.0074 and 0.0106.

## Timings (`scripts/bench_restarts.py`, Apple M5, budgets 10 to 1e6)
| Case | Before (s) | After (s) |
|---|---|---|
| `get_fixed_budget_ecdf`, `coco` (15 runs of 13 to 25 evaluations) | 0.40 | 0.015 |
| the same on 1500 copied runs (as a statistical comparison needs) | 702.6 | 0.08 |
| `aggregate_convergence(stopped="restart")`, `coco` | 0.37 | 0.004 |
| `get_fixed_budget_ecdf`, `coco_old` | 0.007 | 0.031 |

## Decisions
- `n_restarted` stays on the fixed-budget results (the runs of the group that
  stopped before their pair's largest budget); the target ECDFs do not get it.
- The target ECDF with restarts is computed up to the largest of `x_values`,
  `b_hi` for the AUC, or else the largest evaluation of the selection.
- `split`'s `explode` gives polars 1.44's `empty_as_null` deprecation warning,
  as the existing explodes in `targets.py` do; its lists are never empty.

## Open questions / Blocked
- For dvermetten: the deprecated `get_aocc` and `get_data_ecdf` keep no
  restarts; should they follow the new default instead?
- Remaining cost: the pass makes one block step per smallest failed B_j. A
  constructed worst case (13 failed runs of 1e5 evaluations, one failed run of
  1 evaluation, one success) takes 1.8 s at budgets to 1e5 and 26 s to 1e6 for
  the fixed-budget ECDF. The passes for the different F of a run set could be
  vectorised if that matters.
