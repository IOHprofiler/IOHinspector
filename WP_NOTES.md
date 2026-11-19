# WP I17: EAF and EAF differences, review round on #45
Branch: wp/I17-eaf
Status: in progress

Steps 1 to 3 are in #45. This round does steps 4 to 6 of "Review on #45" in
plan/iohinspector.md (dvermetten, 2026-11-19).

## Done
- step 4: `get_eaf_difference` for exactly two groups; `get_eaf_margin` for two or more (discretised); EA-2, CHANGELOG, function reference. (commit 5a8b398)
- step 5: `weighting="runs"|"pairs"` in the EAF functions, warning on uneven run counts, AOCC = volume under the pair-weighted EAF (test), EA-6.

## Next step
Step 6: the 0.0.8 names. `get_discritized_eaf_single_objective` differs from
`get_eaf(mode="discretised")` (checked): keep it, test and document why.

## Decisions
- The multi-group comparison is `get_eaf_margin`, column `eaf_margin`: each
  group's EAF minus the best of the others (its lead over the next best, or how
  far it is behind the best). Discretised only. For two groups it equals the
  discretised `get_eaf_difference`. The golden test of IOHanalyzer's
  `EAF_diff_Approximate` (three algorithms) now runs against it.
- AOCC (step 5): the AOCC is by definition the volume under the pair-weighted
  EAF: for each budget the area under alpha_w(b, .) over q in [0, 1] is
  sum_i w_i q_i(b), the continuum ECDF. Both use `ecdf._quality` and the same
  default bounds, so `get_ecdf_auc` is left unchanged (its results do not
  change) and a test shows the identity to 1e-12. Recomputing the AOCC through
  moocore would give the same number at more cost.
- `weighting="runs"` stays the default (the reviewer asked for an option);
  `"pairs"` weighs each pair equally. Discretised: a weighted mean (1/n_p).
  Precise: moocore has no weights, so each pair's runs are repeated up to the
  lcm of the group's run counts. Both modes warn when a group's pairs have
  different run counts, naming the counts.
- `get_eaf_difference` raises `ValueError` for any number of groups other than
  two, in both modes, and names `get_eaf_margin`.

## Open questions / Blocked
- For dvermetten: is `get_eaf_margin` a clear enough name?
- For dvermetten: should `weighting="pairs"` be the default, so that the EAF
  and the AOCC agree without an option?
