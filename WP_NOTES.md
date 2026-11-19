# WP I33: Fixed-budget ECDF with simulated restarts
Branch: wp/I33-fixed-budget-restarts (stacked on wp/I28-fixed-budget-ecdf, #42)
Status: ready for review

From dvermetten's review of #42 (`docs/definitions/ecdf.md`, the fixed-budget ECDF): simulated restarts become the default of the fixed-budget ECDF and its AUC, carrying the final value forward stays as an option, validated against cocopp.

Definitions: [`docs/definitions/ecdf.md`](docs/definitions/ecdf.md) (fixed-budget ECDF, EC-7, EC-9, Q-EC-4, "Implementation (I33)"), the summary rows in [`docs/definitions/README.md`](docs/definitions/README.md), a note under FB-6 in [`docs/definitions/fixed-budget.md`](docs/definitions/fixed-budget.md).

## Done
- step 1: definitions. EC-7 now says: the fixed-budget ECDF and AUC restart stopped runs by default (FB-6's procedure), `stopped="carry_forward"` is IOHanalyzer's rule; the target ECDF keeps no restarts (Q-EC-4).
- step 2: `get_fixed_budget_ecdf` and `get_fixed_budget_ecdf_auc` take `stopped="restart"` (default) or `"carry_forward"`, and `seed`. They call `fixed_budget._restart`, the procedure of `aggregate_convergence(stopped="restart")`: no second restart helper. With restarts both results have `n_restarted`. The IOHanalyzer golden tests pass `stopped="carry_forward"` (their fixtures have no stopped runs, so the default gives the same values).
- step 3: validation against cocopp 2.9.1 (below).

## Validation against cocopp
cocopp has no fixed-budget ECDF, but its runtime ECDF uses simulated restarts (`toolsstats.drawSP`, from `pprldmany`). As T(t) <= b exactly when V(b) <= t, cocopp's fraction of simulated runtimes to target t that are <= b is our fixed-budget ECDF with the one budget b at v = t.

- Reference: `tests/golden/cocopp/generate.py` now writes `coco_old/ecdf_restarts_{2D,3D}.parquet`: `drawSP` called as `pprldmany` calls it (derandomised first draw, `ds.maxevals` for unsuccessful runs), 20000 samples per target, `np.random.seed(20240423)`, for the 51 targets 10^(2-k/5) and 10 budgets round(10^(1+5k/9)) = 10 ... 1e6: 510 cells per dimension. Regeneration is byte-identical; the existing cocopp files are unchanged.
- Fixture: `coco_old`, DIRECT on f6, 2-D and 3-D, five runs each, B_i between 3e4 and 1e5, mostly unsuccessful for small targets; budgets up to 1e6 force up to about 30 restarts.
- Ours (`test_fixed_budget_restarts_match_cocopp`): each cell from `get_fixed_budget_ecdf` with that one budget, at v = cocopp's grid level of the target, on the five runs copied 1000 times (5000 chains; copies leave the runs that restarts draw from unchanged), seed 1.
- Not a golden test: cocopp's random stream (global `np.random`, drawing whole runtimes) cannot be matched to ours (one `default_rng` chain per run), so the test is statistical.
- Tolerance: every cell within 5 binomial standard errors of both samples, mean absolute difference < 0.002; fixed-budget ECDF over all ten budgets within 0.005 of the mean of cocopp's cells.
- Measured: largest cell difference 0.0074 (2-D) and 0.0106 (3-D); largest in standard errors 2.3 and 2.2; mean absolute difference 0.00015 and 0.00024; 212 and 308 cells equal exactly (targets no run or every run reaches). Fixed-budget ECDF over the ten budgets: at most 0.0005 (2-D) and 0.0012 (3-D) from cocopp. Carrying forward instead differs by up to 0.071 and 0.099, so the check tells the two rules apart.

## Decisions
- B_i without `evals_used` is the run's last logged evaluation, as for the ERT (RT-1, `runs_of`), so the default works on data loaded without `evals_used`. `aggregate_convergence(stopped="restart")` still requires `evals_used` (unchanged, its default is carry forward).
- A chain draws from the runs of its group of `by` within its (function, dimension) pair.
- `n_restarted` per group: the runs that stopped before their pair's largest budget.
- `seed` defaults to None, as in `aggregate_convergence`, so the default result varies between calls; callers that need a fixed curve (the website) pass a seed.
- `cocopp_level` moved from `tests/io/test_coco_cocopp.py` to `tests/golden/load.py`, shared by both tests.

## Open questions / Blocked
- Q-EC-4 for dvermetten (not built): should the target ECDF (`get_ecdf`, EC-1/EC-7) also offer or default to simulated restarts, as cocopp's runtime ECDF does? It would need a largest budget up to which runs are restarted.
- Should the seed default be fixed (for example 0) rather than None, so that the default ECDF is reproducible?
