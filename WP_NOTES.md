# WP I33: simulated restarts in every ECDF, round 3 (#48)
Branch: wp/I33-fixed-budget-restarts
Status: in progress

Steps 9 to 11 of `plan/iohinspector.md` ("Answers from the user", 2026-11-20).

## Done
- definitions: EC-7 (n_restarted for the target ECDFs; EAF default differs),
  EA-6 and "Simulated restarts" in eaf.md, README rows.

## Next step
Code: n_restarted in `get_ecdf`/`get_ecdf_auc`; `stopped` in the EAF functions
(discretised only, via `restarts.gained`); tests; CHANGELOG.

## Decisions
- Step 9: nothing to build.

## Open questions / Blocked
- (none)
