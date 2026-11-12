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
- step 3: `iohinspector.targets`: `spaced`, `seq_fv` (TG-1, TG-3), `seq_rt`
  (TG-5), `bbob_targets`, `get_ecdf_targets` (per pair, TG-2), `get_target_dt`
  (`perspective`, TG-4, TG-6). `get_sequence` and the metric default grids
  (fixed target, fixed budget, ECDF, EAF) now use them: one generator.
  Golden tests in `tests/test_targets.py` against `targets_*`, `budgets_*`,
  `target_dt_*`; the f = 0 xfail in `tests/io/test_ioh_defects.py` passes.
- step 4: `test_samples_match_iohanalyzer` (`tests/test_align.py`): every run's
  hitting times (`RT_sample`, 750 rows, 164 never reached) and values at
  budgets (`FV_sample`, 750 rows) on `ioh040_min` and `ioh040_max` match
  IOHanalyzer exactly, except one value IOHanalyzer parses 1 ulp off. The
  goldens hold no run without a value yet, so AL-5 is covered by unit tests
  only. Helpers `logged` and `golden_renamed` in `tests/golden/load.py`.
- step 5 (D17): descriptor names already resolve to columns in the data
  layer (`StoredDataset.run_columns`, `DataManager.load(include_columns=by)`);
  every metric already groups on `by`. Added `metrics.group_label(by,
  template)` (`{name}` placeholders, as `tags_from_path`), used by the plots'
  hue for several `by` columns. `tests/test_metrics/test_by.py`: four metrics
  grouped by algorithm, a numeric parameter (`sigma0`), function, a user tag
  and tag + dimension; runs without the descriptor form the null group.

## Next step
All steps are committed except AL-9 (open question below). When AL-9 is
answered: implement it in `align_data`, then finish (PR body from these
notes, delete WP_NOTES.md, D6 review request).

## Decisions
- `n_runs`/`n_defined` names follow fixed-budget.md; I11 adds the rest of the
  FB summary (`n_active`, quantiles).
- Metric default grids stay over the whole frame, as before; per-pair grids in
  the metrics are I11/I12 (the generators are per pair already).
- Plot-level label templates (a `label=` argument on each plot) belong to the
  plotting WP (iohinspector.md, BBOB-style ECDF item 3); I10 gives the shared
  `group_label`.
- No fixture has algorithm parameters as descriptors; the parameter test uses
  a typed tag (`sigma0:float`), which takes the same path.
- Golden target tolerances: linear and log-linear rtol 1e-15 (IOHanalyzer reads
  values back from text row names, 1 ulp off), bbob 4e-15 (built from exact
  decades, IOHanalyzer goes through log10).

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
