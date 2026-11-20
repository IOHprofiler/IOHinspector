# WP I33: simulated restarts in every ECDF, round 3 (#48)
Branch: wp/I33-fixed-budget-restarts
Status: ready for review

Steps 9 to 11 of `plan/iohinspector.md` ("Answers from the user", 2026-11-20).
Add to the #48 body under "Done".

## Done
- step 9: nothing to build (deprecated `get_aocc`/`get_data_ecdf` keep no
  restarts; the 26 s worst case is left).
- step 10: `get_ecdf` and `get_ecdf_auc` report `n_restarted` per group with
  restarts: the runs that stopped before the largest budget computed (the
  horizon, in evaluations per pair). Shares `_stopped_before` with the
  fixed-budget ECDF.
- step 11: `get_eaf`, `get_eaf_difference`, `get_eaf_margin` take
  `stopped="carry_forward"` (default, unchanged) or `"restart"`, discretised
  only (precise raises `ValueError` suggesting `mode="discretised"` or
  `"carry_forward"`). alpha with restarts = alpha + per attaining run the
  renewal gain of `restarts.gained` (now per budget; the fixed-budget ECDF sums
  it). Tests: the volume of the pair-weighted EAF equals the AOCC with runs that
  stop early under both options; alpha at y equals the restarted single-target
  ECDF; difference and margin follow. EA-6, EC-7, docstrings: the defaults differ.

## Decisions
- `n_restarted` counts runs with B_i below the horizon, as the fixed-budget
  results count runs below the pair's largest budget.

## Open questions / Blocked
- (none)
