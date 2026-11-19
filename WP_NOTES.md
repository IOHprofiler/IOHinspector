# WP I17: EAF and EAF differences (review round 2)
Branch: wp/I17-eaf
Status: ready for review

Steps 1 to 6 are in IOHprofiler/IOHinspector#45. This round does steps 7 to 9,
the user's answers (2026-11-19) to the round-1 questions.

## Done
- step 7: the name `get_eaf_margin` stays; nothing to change (commit 9e7082b).
- step 8: `weighting="pairs"` is the default of `get_eaf`, `get_eaf_difference`
  and `get_eaf_margin`, so the volume under the EAF is the AOCC without an
  option; `"runs"` stays. EA-6 (decision and implementation), the CHANGELOG
  and FUNCTION_REFERENCE say so. Only
  `test_precise_eaf_weighs_pairs_by_repeating_runs` relied on the old default;
  it now names `weighting="runs"` for the comparison. No golden result
  changed: the IOHanalyzer golden tests group by the pair, where the two
  weightings agree (commit 4bf8e1f).
- step 9: `eaf.MAX_SETS = 100_000`. With `weighting="pairs"` in the precise
  mode (`get_eaf`, `get_eaf_difference`), when repeating each pair's runs up to
  the least common multiple would give more attainment sets than that in
  total, `ValueError` names the count and suggests `mode="discretised"`.
  `test_precise_pair_weighting_is_capped` sets the limit to 10 and 9 around
  the 10 sets of its data; discretised and `"runs"` repeat nothing and pass
  (commit df1609b).

## Tests
`.venv/bin/python -m pytest -q`: 1217 passed, 3 skipped on polars 2.1.0 and on
polars 1.44.2.

## Next step
None: push, and update #45's body or a comment from this file.

## Decisions
- The cap counts the sets after repetition over all groups, and only when
  something is repeated (pairs of a group with different run counts).

## Open questions / Blocked
- The precise difference also repeats each side's runs up to the lcm of the
  two groups' run counts (moocore needs equal sides; this predates the
  round). The cap does not cover that repetition. Say if it should.
