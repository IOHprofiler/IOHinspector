# WP I16: Shapley values and cumulative difference plot
Branch: wp/I16-shapley-cdp
Status: in progress

## Done
- step 1: Shapley values (`metrics/portfolio.py`): generic `shapley_values(n, value, marginal=...)`
  (SH-2, SH-3, SH-4) and `get_shapley_values(data, targets, by)` with the ECDF-envelope value
  (SH-1); unit tests and IOHanalyzer golden tests in `tests/test_metrics/test_portfolio.py`.

- step 2: cumulative difference plot (`metrics/cdp.py`): `cumulative_difference(x_a, x_b)`, a
  vectorised port of RVCompare 0.1.8's `get_Y_AB_bounds_bootstrap` (CC0), and
  `get_cumulative_difference(data, target=|budget=, by)` with seed, failures as +inf (CD-2),
  subsampling with a warning and `result.runs` (CD-3), eps 1e-80 (CD-4). Tests in
  `tests/test_metrics/test_cdp.py`; new RVCompare reference `tests/golden/rvcompare/` from
  `tests/golden/r/rvcompare.R` (wired into `make golden`).

## Next step
Step 3: write the "Implementation (I16)" sections and the resolved points into
`docs/definitions/portfolio.md` and `cdp.md` (Y_A/Y_B checked against Arza 2023; band tolerance),
CHANGELOG entry, then finish (delete WP_NOTES.md).

## Decisions
- Value functions are batched: v takes a boolean matrix (one row per coalition) and returns one
  value per row; the marginal is m(v, S, a) with v the cached value function. Exact: v is called
  once on all 2^n coalitions; sampled: once per distinct coalition (cache).
- Sampling draws, for every player and position j, `samples` random (j-1)-subsets of the other
  players (the same distribution as a permutation with the player at position j) and reports a
  standard error per player (`se`).
- `_prepare` was split out of `ecdf._increments` so the portfolio uses the same targets, pair
  selection (EC-2, `pairs=`) and hit rule (EC-1) as `get_ecdf`.
- Golden tests: IOHanalyzer's value (R `ecdf` of successful runtimes, summed over 50 grid points)
  plugged into `shapley_values` as a custom value reproduces `ECDF_raw` and `marg_contrib_ecdf`
  exactly; its unseeded sampled Shapley values are checked against the range its estimator can
  produce (exact for two players).

- CDP estimates match IOHanalyzer's goldens to 1e-13 on all 22 (fixture, pair, view) cases,
  including `CDP_by_FV_with_failures` on ioh040_max. Bands: the fixtures have 5 runs per
  algorithm, so their bootstrap bands are too discrete to compare across random streams (one
  IOHanalyzer band lies outside the envelope of our 50 seeds, inside that of 400). They are
  compared on the n=100 RVCompare reference instead: Y bands within 0.02, difference bands within
  0.04 (our own seed-to-seed spread of the difference band is up to 0.037).
- Fixed budget: a run with no evaluation yet at the budget (null, AL-5) counts as the worst value,
  like a failed run (CD-2).

## Open questions / Blocked
- For the default value (a sum of maximum games) the Shapley value has a closed form, exact for any
  number of algorithms (the tests use it as an oracle). SH-2 asks for sampling above 15; the
  closed form could replace it for the default value. For the reviewer.
