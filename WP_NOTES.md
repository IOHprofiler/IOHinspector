# WP I14: Parameters, positions and constraint penalties
Branch: wp/I14-parameters
Status: in progress

## Done
- step 1: `metrics/parameters.py`: `get_parameter_sample`, `get_parameter_summary`,
  `get_logged_rows` (correlation pairs and positions), `apply_penalty`; unit tests.

## Next step
Step 2: golden tests in `tests/test_metrics/test_parameters.py` against
`extdata` (PAR_sample/summary by RT and FV), the JSON fixtures (PAR_sample_by_RT
at logged evaluations), `ioh040_pos/positions`, `ioh040_cons/Parameter_correlation`
and `ioh040_cons/penalty_*_FV_matrix`. Then step 3: definitions page "what was
built", CHANGELOG.

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
