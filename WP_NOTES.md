# WP I16: Shapley values and cumulative difference plot (review round on #44)
Branch: wp/I16-shapley-cdp
Status: ready for review

Steps 1 to 3 are in #44. This round does steps 4 to 6 of "Review on #44"
(dvermetten, 2026-11-19) in `plan/iohinspector.md`.

## Done
- step 4: `scripts/bench_shapley.py` times exact against sampled Shapley values (table below)
- step 5: `get_portfolio_ecdf` and `envelope_value` are public; tests define another value and another marginal through them; how-to in `docs/definitions/portfolio.md`
- step 6: `FUNCTION_REFERENCE.md` has separate subsections for Shapley values and the cumulative difference plot

## Step 4: exact against sampled

`python scripts/bench_shapley.py --groups 5 8 10 11 12 13 14 15`: `get_shapley_values`
end to end on a synthetic full-bbob set (the data of `bench_ert.py`: 24 functions,
6 dimensions, 15 runs per algorithm and pair, 40 improvements per run), default
targets, log budget axis, about 142 000 merged ECDF steps at 15 groups. Sampled
with the default 5 coalitions per player and position, seed 1. MacBook, 10 threads.

| groups | exact (s) | sampled (s) | exact / sampled | max diff | max se |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.33 | 0.33 | 1.0 | 3.8e-04 | 3.3e-04 |
| 8 | 0.57 | 0.54 | 1.1 | 3.1e-04 | 2.7e-04 |
| 10 | 1.31 | 0.89 | 1.5 | 4.2e-04 | 2.1e-04 |
| 11 | 2.48 | 1.19 | 2.1 | 2.5e-04 | 2.3e-04 |
| 12 | 5.17 | 1.61 | 3.2 | 3.9e-04 | 2.3e-04 |
| 13 | 11.50 | 2.14 | 5.4 | 3.1e-04 | 2.2e-04 |
| 14 | 26.18 | 2.83 | 9.3 | 2.2e-04 | 2.2e-04 |
| 15 | 69.27 | 5.19 | 13.4 | 3.3e-04 | 2.0e-04 |

Both include about 0.3 to 0.8 s to build the ECDF steps. Exact is much slower near
15 (69 s against 5 s), so the limit is **not** changed silently. Proposal, for the
reviewer to choose (Q-SH-2 in `portfolio.md`):

1. lower the default `exact_limit` to 12 (exact within a few seconds on a full-bbob set); or
2. use the closed form of the default value, a sum of maximum games: exact for any
   number of groups, 0.04 s at 15 groups, equal to the enumeration within 1e-15.
   Enumeration and sampling stay for other definitions.

## Step 5: another value or marginal

The test showed one gap: `shapley_values` itself took any value and marginal, but
the ECDF data of `get_shapley_values` were hidden in a closure, so a marginal not
built on the upper envelope could not be defined on them. Now
`get_portfolio_ecdf(data, targets, by, ...)` returns `(players, ecdf, width)` and
`envelope_value(ecdf, width)` is the default value; `get_shapley_values` is these two
plus `shapley_values`. Tests against a direct computation over all orders (the marginal also sampled):
- another value: the parallel portfolio, 1 − Π(1 − ECDF) per step;
- another marginal, not a difference of values: the share of the budget axis on
  which a is strictly ahead of every member of S.

## Decisions
- `get_portfolio_ecdf` no longer drops steps where no player has reached the target
  (they merge into one column anyway); a value that is not 0 there now sees them.
- Merged steps lose their pair, target and budget. A definition that weights steps
  by those needs the unmerged steps; not exposed (said in `portfolio.md`).

## Next step
None; review.

## Open questions / Blocked
- Q-SH-2: exact limit 12, or the closed form for the default value (step 4).
