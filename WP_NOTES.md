# WP I17: EAF and EAF differences
Branch: wp/I17-eaf
Status: ready for review

## Done
- step 1: `metrics.get_eaf` and `metrics.get_eaf_difference`, precise and discretised (EA-1, EA-2, EA-5), q across pairs (EA-3); unit tests; `get_eaf` in `test_by`. (commit 9bb32e6)
- step 2: golden tests (EAF, EAF_logx, EAF_Difference, EAF_diff_Approximate) and the implementation section of `docs/definitions/eaf.md` (commit 1822179)
- step 3: CHANGELOG, FUNCTION_REFERENCE.md, lint (pylint clean on the new code).

## Next step
Done. Controller: push, open the PR, request dvermetten (metric code, D6).

## Decisions
- moocore checked against `eaf` 2.5 (EA-1): on IOHanalyzer's inputs `moocore.eaf` gives the golden surfaces exactly, so the level convention is the same (p -> at least ceil(p n / 100) runs, and at least one).
- moocore reads a set from consecutive rows: unsorted rows give a wrong EAF. Traces are sorted by run; the 0.0.8 `get_eaf_pareto_data` / `get_eaf_diff_data` had the bug and are fixed (`_points`).
- `moocore.eafdiff` assumes both sides have equally many sets (it splits at nsets/2). The precise difference repeats each group's runs up to the lcm of the two run counts.
- `moocore.eafdiff` fails when there is no rectangle at all; that is now an empty frame (`_eafdiff`).
- Precise difference: exactly two groups (as IOHanalyzer's `EAF_Difference`); with more, `mode="discretised"` (max of the others, EA-2).
- Several pairs per group of `by`: the value is the normalised quality q of the continuum ECDF (`ecdf._quality`, EC-5), with the same default bounds, so I28's suite bounds (TG-8) apply here once merged. Runs are pooled (each run one attainment set).

- IOHanalyzer defects found by the golden tests (recorded in `docs/definitions/eaf.md`): `EAF_Difference` ignores maximisation; `EAF_diff_Approximate` labels the value axis in reverse under maximisation; its grid ends are 1 ulp off the data.
- moocore works for 0.1.8 (the floor in pyproject) and 0.3.2; checked locally, polars 1.44 and 2.1.

## Open questions / Blocked
- For dvermetten: the precise difference compares exactly two groups. With more, Delta_a = alpha_a - max of the others is only offered discretised; an exact version would need the EAF of every group on the union of all coordinates.
- For dvermetten: the multi-function EAF pools runs (each run one attainment set), so pairs weigh by their run counts; the AOCC weighs pairs equally (EC-2). Equal when every pair has as many runs.
- `get_discritized_eaf_single_objective` (0.0.8, quality levels) is kept unchanged; a candidate for deprecation in favour of `get_eaf(mode="discretised")`.
