# WP I28: Fixed-budget ECDF and default targets from the suite registry
Branch: wp/I28-fixed-budget-ecdf
Status: in progress

## Done
- step 1: definitions: EC-9, EC-10 in `docs/definitions/ecdf.md`, TG-8 and the budget table in `targets.md`

## Next step
Step 2: `get_fixed_budget_ecdf` and `get_fixed_budget_ecdf_auc` in `metrics/ecdf.py`, `get_ecdf_budgets` in `targets.py`, tests in `tests/test_metrics/test_ecdf.py` with golden `ECDF_by_FV` and `AUC_by_FV`.
Then step 3: TG-8 table in `targets.py`, `get_ecdf_targets` uses it, test against the ioh 0.4.2 registry file.

## Decisions
- Fixed-budget ECDF inclusive (V(b) <= v); IOHanalyzer's strict T < b reproduced in golden tests with b - 1.
- Fixed-budget AUC: exact normalised integral over [v_lo, v_hi], log10 axis by default; defaults from per-pair bounds (TG-8 or observed).
- TG-8 table holds only suites whose registry `target_bounds` give both bounds (BBOB, SBOX in ioh 0.4.2). Suite matched case-insensitively (COCO `bbob`). Suite read from the data's `suite` column.
- Workers commit locally (CLAUDE.md); the controller pushes and opens the PR.

## Open questions / Blocked
- none
