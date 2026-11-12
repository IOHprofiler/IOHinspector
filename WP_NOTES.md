# WP I14: Parameters, positions and constraint penalties
Branch: wp/I14-parameters
Status: in progress

## Done
- step 1: `metrics/parameters.py`: `get_parameter_sample`, `get_parameter_summary`,
  `get_logged_rows` (correlation pairs and positions), `apply_penalty`; unit tests.

- step 2: golden tests against `extdata` (PAR_sample and PAR_summary by RT and FV,
  exact), `ioh040_cons` PAR_summary_by_RT, the JSON fixtures' PAR_sample_by_RT
  (equal where the run logged b; elsewhere IOHanalyzer's value is the run's next
  logged row), positions, Parameter_correlation and penalty_*_FV_matrix.

## Next step
Step 3: definitions page "what was built", CHANGELOG entry.

## Decisions
- `QUANTILES` is defined in `parameters.py` because I11 (which adds the same tuple
  to `metrics/utils.py`) is not merged; import it from utils once I11 is in.
- Exports appended at the end of `metrics/__all__` and the import block, to keep
  the merges with I11-I13 free of conflicts.
- Parameters are looked up on the logged rows with `align_data`; by target, the
  rows are first cut to those where the best-so-far changes, so each target has
  exactly one hitting row (the first that reaches it).

## Open questions / Blocked
- none
