# WP I16: Shapley values and cumulative difference plot
Branch: wp/I16-shapley-cdp
Status: in progress

## Done
- step 1: Shapley values (`metrics/portfolio.py`): generic `shapley_values(n, value, marginal=...)`
  (SH-2, SH-3, SH-4) and `get_shapley_values(data, targets, by)` with the ECDF-envelope value
  (SH-1); unit tests and IOHanalyzer golden tests in `tests/test_metrics/test_portfolio.py`.

## Next step
Step 2: port RVCompare 0.1.8's `get_Y_AB_bounds_bootstrap` (CC0) as `get_cdp` in
`metrics/cdp.py`, with seed, failed runs as +inf (CD-2), subsampling (CD-3), eps 1e-80 (CD-4).

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

## Open questions / Blocked
- For the default value (a sum of maximum games) the Shapley value has a closed form, exact for any
  number of algorithms (the tests use it as an oracle). SH-2 asks for sampling above 15; the
  closed form could replace it for the default value. For the reviewer.
