# WP I17: EAF and EAF differences, review round on #45
Branch: wp/I17-eaf
Status: ready for review

Steps 1 to 3 are in #45. This round does steps 4 to 6 of "Review on #45" in
plan/iohinspector.md (dvermetten, 2026-11-19).

## Done
- step 4: `get_eaf_difference` for exactly two groups; `get_eaf_margin` for two or more (discretised); EA-2, CHANGELOG, function reference. (commit 5a8b398)
- step 5: `weighting="runs"|"pairs"` in the EAF functions, warning on uneven run counts, AOCC = volume under the pair-weighted EAF (test), EA-6. (commit 3b372be)
- step 6: no 0.0.8 name gives a new call's output, so none becomes an alias; test and definitions say why.

## Next step
Review done. Controller: push, update the body of #45 from this file, and
re-request dvermetten's review.

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
- Old names (step 6): `get_discritized_eaf_single_objective` is not
  `get_eaf(mode="discretised")`: same on its levels except q = 0, where 0.0.8
  counts every run (transform_fval clips values above 1e2 and fills nulls
  with 0) and `get_eaf` only the runs that reached it; it also uses fixed
  quality bounds on one pair (get_eaf uses f) and a wide layout. Kept, with
  a test pinning the difference. `get_eaf_data` (aligned values),
  `get_eaf_pareto_data` and `get_eaf_diff_data` (any two objectives, point
  sets) compute other things; kept. No deprecation warnings added.
- `get_eaf_difference` raises `ValueError` for any number of groups other than
  two, in both modes, and names `get_eaf_margin`.

## Open questions / Blocked
- For dvermetten: is `get_eaf_margin` a clear enough name?
- For dvermetten: should `weighting="pairs"` be the default, so that the EAF
  and the AOCC agree without an option?
