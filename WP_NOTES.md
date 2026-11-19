# WP I17: EAF and EAF differences, review round on #45
Branch: wp/I17-eaf
Status: in progress

Steps 1 to 3 are in #45. This round does steps 4 to 6 of "Review on #45" in
plan/iohinspector.md (dvermetten, 2026-11-19).

## Done
- step 4: `get_eaf_difference` for exactly two groups; `get_eaf_margin` for two or more (discretised); EA-2, CHANGELOG, function reference.

## Next step
Step 5: `weighting` option ("runs" default, "pairs") in `get_eaf`, a warning
when a group's pairs have different run counts, and a test that the AOCC
equals the area under the EAF with weighting="pairs".

## Decisions
- The multi-group comparison is `get_eaf_margin`, column `eaf_margin`: each
  group's EAF minus the best of the others (its lead over the next best, or how
  far it is behind the best). Discretised only. For two groups it equals the
  discretised `get_eaf_difference`. The golden test of IOHanalyzer's
  `EAF_diff_Approximate` (three algorithms) now runs against it.
- `get_eaf_difference` raises `ValueError` for any number of groups other than
  two, in both modes, and names `get_eaf_margin`.

## Open questions / Blocked
- For dvermetten: is `get_eaf_margin` a clear enough name?
