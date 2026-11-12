# WP I10: Alignment and target grids
Branch: wp/I10-alignment
Status: in progress

Definitions: `docs/definitions/alignment.md` (AL-*), `targets.md` (TG-*),
`fixed-budget.md` (FB-*), `ecdf.md` (EC-*).

## Done
- step 2 (AL-5): `align_data` leaves a fixed-budget value null before a run's
  first logged evaluation (fixed target keeps `inf` for a target never reached).
  Callers: `aggregate_convergence` leaves it out and counts it (`n_runs`,
  `n_defined`); `transform_fval` maps null to 0 (reached nothing), so
  `get_data_ecdf` and `get_discritized_eaf_single_objective` count the run;
  the EAF plot ranks it worst; `get_trajectory`, `get_eaf_data` and
  `load(x_values)` return null. Both EAF functions take `maximization`.
  CHANGELOG: "Unreleased" fix entry replaces the 0.1.0 known issue.

## Next step
Step 1: tests for the fixed-target rules (AL-1 inclusive hit, AL-3 first row on
ties, maximisation mirror, AL-6 non-monotone input), then AL-9 (`is_final`).

## Decisions
- `n_runs`/`n_defined` names follow fixed-budget.md; I11 adds the rest of the
  FB summary (`n_active`, quantiles).

## Open questions / Blocked
- none
