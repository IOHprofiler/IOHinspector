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
- step 1: tests for the fixed-target rules (AL-1 inclusive, AL-3 first of
  ties, AL-6 raw input, both directions). `align_data` already met them.

## Next step
Step 3: `iohinspector.targets` (TG-1..TG-6): `seq_FV`, `seq_RT`,
`get_ECDF_targets`, `get_target_dt` and the bbob grid, with golden tests
against `targets_*`, `budgets_*` and `target_dt_*`.

## Decisions
- `n_runs`/`n_defined` names follow fixed-budget.md; I11 adds the rest of the
  FB summary (`n_active`, quantiles).

## Open questions / Blocked
- AL-9 (keep every run's final evaluation, `is_final`) is not specified enough
  to implement without guessing: `align_data` gets a frame of trace rows, and
  B_i is run metadata (`evals_used`), which the frame does not carry unless
  loaded with `include_meta_data`. Unsettled: (a) where B_i comes from in
  `align_data` (a column such as `evals_used`, falling back to the last logged
  evaluation?); (b) whether the extra row at B_i goes into every caller's
  output (it adds a row per run outside the grid, so every per-grid-point
  summary, trajectory and `load(x_values)` would have to filter it out); (c)
  whether a grid point equal to B_i is that row. Proposal: `align_data` adds,
  per run, a row at B_i (`evals_used` if present, else the last logged
  evaluation) with `is_final=True`, grid rows `is_final=False`; summaries
  drop `is_final` rows and I11 uses them for `n_active` (FB-6). Needs
  dvermetten's yes before it is built.
