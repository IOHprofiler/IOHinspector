# WP I17: EAF and EAF differences
Branch: wp/I17-eaf
Status: in progress

## Done
- step 1: `metrics.get_eaf` and `metrics.get_eaf_difference`, precise and discretised (EA-1, EA-2, EA-5), q across pairs (EA-3); unit tests; `get_eaf` in `test_by`. (commit 9bb32e6)
- step 2: golden tests (EAF, EAF_logx, EAF_Difference, EAF_diff_Approximate) and the implementation section of `docs/definitions/eaf.md`.

## Next step
Step 3: CHANGELOG, FUNCTION_REFERENCE.md, final checks; then remove WP_NOTES.md.

## Decisions
- moocore checked against `eaf` 2.5 (EA-1): on IOHanalyzer's inputs `moocore.eaf` gives the golden surfaces exactly, so the level convention is the same (p -> at least ceil(p n / 100) runs, and at least one).
- moocore reads a set from consecutive rows: unsorted rows give a wrong EAF. Traces are sorted by run; the 0.0.8 `get_eaf_pareto_data` / `get_eaf_diff_data` had the bug and are fixed (`_points`).
- `moocore.eafdiff` assumes both sides have equally many sets (it splits at nsets/2). The precise difference repeats each group's runs up to the lcm of the two run counts.
- `moocore.eafdiff` fails when there is no rectangle at all; that is now an empty frame (`_eafdiff`).
- Precise difference: exactly two groups (as IOHanalyzer's `EAF_Difference`); with more, `mode="discretised"` (max of the others, EA-2).
- Several pairs per group of `by`: the value is the normalised quality q of the continuum ECDF (`ecdf._quality`, EC-5), with the same default bounds, so I28's suite bounds (TG-8) apply here once merged. Runs are pooled (each run one attainment set).

## Open questions / Blocked
- none
