# WP I33: Fixed-budget ECDF with simulated restarts
Branch: wp/I33-fixed-budget-restarts (stacked on wp/I28-fixed-budget-ecdf, #42)
Status: in progress

## Done
- step 1: definitions (ecdf.md fixed-budget section, EC-7, EC-9, Q-EC-4; README rows; fixed-budget.md note)
- step 2: `stopped="restart"` (default) / `"carry_forward"` and `seed` on get_fixed_budget_ecdf(_auc), reusing fixed_budget._restart; `n_restarted`; unit tests

## Next step
Step 3: cocopp validation. Extend tests/golden/cocopp/generate.py with drawSP restart fractions on coco_old, regenerate with the golden venv, add the statistical test.

## Decisions
- B_i without `evals_used` is the last logged evaluation (runs_of, as the ERT, RT-1), so the default works on any data; aggregate_convergence still requires evals_used for its (non-default) restarts.
- The restart pool of a chain is its group of `by` within its (function, dimension) pair.
- `n_restarted` per group: runs that stopped before their pair's largest budget.
- seed defaults to None, as aggregate_convergence.

## Open questions / Blocked
- Q-EC-4 for dvermetten: simulated restarts for the target ECDF (get_ecdf) too?
