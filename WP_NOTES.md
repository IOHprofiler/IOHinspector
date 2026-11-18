# WP I28: Fixed-budget ECDF and default targets from the suite registry
Branch: wp/I28-fixed-budget-ecdf
Status: ready for review

## Done
- step 1: definitions: EC-9, EC-10 in `docs/definitions/ecdf.md`, TG-8 and the budget table in `targets.md`
- step 3: TG-8: `targets.SUITE_BOUNDS` (BBOB, SBOX: 1e-8 to 1e2), used by `get_ecdf_targets` (new `lower`/`upper`); test against the trimmed ioh 0.4.2 `registry.json` (`tests/ioh_registry_0.4.2.json`, with source URL and sha256)
- step 2: `get_fixed_budget_ecdf`, `get_fixed_budget_ecdf_auc` (metrics/ecdf.py), `targets.get_ecdf_budgets`; golden tests against `ECDF_by_FV` (exact with b - 1) and `AUC_by_FV` (trapezoid reproduced)

## Next step
Controller: push, open the PR, request dvermetten's review.

## Decisions
- Fixed-budget ECDF inclusive (V(b) <= v); IOHanalyzer's strict T < b reproduced in golden tests with b - 1.
- Fixed-budget AUC: exact normalised integral over [v_lo, v_hi], log10 axis by default; defaults from per-pair bounds (TG-8 or observed).
- TG-8 table holds only suites whose registry `target_bounds` give both bounds (BBOB, SBOX in ioh 0.4.2). Suite matched case-insensitively (COCO `bbob`). Suite read from the data's `suite` column.
- Workers commit locally (CLAUDE.md); the controller pushes and opens the PR.

## Open questions / Blocked
- none
