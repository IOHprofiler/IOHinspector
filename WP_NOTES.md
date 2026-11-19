# WP I33: fixed-budget ECDF with simulated restarts, review round (#48)
Branch: wp/I33-fixed-budget-restarts
Status: in progress

Steps 1 to 4 are in #48. This round does steps 5 to 8 of `plan/iohinspector.md`
("Review on #48", dvermetten).

## Done
- definitions: EC-7 revised (restarts in every ECDF, expected value computed from
  the runtimes by the renewal equation E_t, no seed), FB-6 (seed 0, chains stop
  at a best run), README rows.

## Next step
`src/iohinspector/metrics/restarts.py`: the renewal computation; then wire
`get_ecdf`/`get_ecdf_auc` (targets, continuum), the fixed-budget ECDF (pieces,
replaces the alignment path) and the portfolio; then `aggregate_convergence`.

## Decisions
- Before timings: `scripts/bench_restarts.py --repeat 1` on the code of #48 (HEAD
  1668939), exported to the scratchpad.

## Open questions / Blocked
- (none)
