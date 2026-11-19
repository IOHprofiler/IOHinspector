# WP I17: EAF and EAF differences (review round 2)
Branch: wp/I17-eaf
Status: in progress

Steps 1 to 6 are in IOHprofiler/IOHinspector#45. This round does steps 7 to 9,
the user's answers (2026-11-19) to the round-1 questions.

## Done
- step 7: the name `get_eaf_margin` stays; nothing to change.
- step 8: `weighting="pairs"` is the default of `get_eaf`, `get_eaf_difference`
  and `get_eaf_margin`; EA-6, the CHANGELOG and FUNCTION_REFERENCE say so. Only
  `test_precise_eaf_weighs_pairs_by_repeating_runs` relied on the default
  ("runs"); it now names `weighting="runs"`. No golden result changed: the
  IOHanalyzer golden tests group by the pair, where the weightings agree.

## Next step
Step 9: cap the precise mode's repetition under `weighting="pairs"` with a
module constant (about 100,000 attainment sets); `ValueError` suggesting
`mode="discretised"`; test both sides of the limit.

## Decisions
- ...

## Open questions / Blocked
- none
