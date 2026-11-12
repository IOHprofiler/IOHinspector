# WP I29: Pre-release review before 0.2.0
Branch: wp/I29-prerelease-review
Status: in progress

## Done
- step 1: align_data chose the fixed-budget view by the column name "evaluations";
  every metric with another `eval_var` was wrong. Keyword `eval_var`. Test, CHANGELOG.

## Next step
Step 2: fix `by=()` (get_overview, get_ecdf, get_ecdf_auc crash) and duplicate
x_values in align_data (counted twice).

## Plan
1. align_data eval_var bug (done)
2. by=() crashes; duplicate targets/budgets double-count runs
3. samples: get_runtime_sample / get_value_sample return only their columns
4. align_data fixed target: keep the rows where best-so-far changes (first row
   that reaches each value) instead of a two-key sort; parameters drop their copy
5. DRY: one type-7 summary (aggregate_convergence, get_parameter_summary), one
   custom_op column; simpler aggregate_running_time post-processing
6. API: keyword-only options of the functions new since 0.1.0; CHANGELOG
7. edge-case tests; line counts; notes

## Line counts (lines / code lines; tokenize, string-only lines excluded)
Before (16e84e0): src/iohinspector 12,546 / 8,433; metrics/ 2,165 / 1,397;
tests 10,176 / 7,929; align.py 87/48, targets.py 145/99, fixed_target.py 133/86,
fixed_budget.py 226/160, ecdf.py 362/224, distributions.py 241/183,
parameters.py 201/118, metrics/utils.py 239/137.

## Decisions
- Counting script: code lines are lines that hold a token other than a string or
  comment (Python tokenize); this reproduces the plan's baseline exactly.

## Open questions / Blocked
- none yet
