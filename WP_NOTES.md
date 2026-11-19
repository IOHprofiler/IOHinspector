# WP I33: fixed-budget ECDF with simulated restarts, review round (#48)
Branch: wp/I33-fixed-budget-restarts
Status: in progress

Steps 1 to 4 are in #48. This round does steps 5 to 8 of `plan/iohinspector.md`
("Review on #48", dvermetten).

## Done
- definitions: EC-7 revised (restarts in every ECDF, expected value computed from
  the runtimes by the renewal equation E_t, no seed), FB-6 (seed 0, chains stop
  at a best run), README rows.

- steps 5 to 7 code: `metrics/restarts.py` (E_t by the renewal recursion);
  restarts in `get_ecdf`/`get_ecdf_auc` (targets and continuum), the portfolio
  ECDFs, and the fixed-budget ECDF/AUC (now from pieces of the runs' traces, one
  path for both options, no seed); `aggregate_convergence` seed 0 by default and
  chains stop at a best run; the BBOB-style figure runs every curve to the common
  end; snapshots regenerated. Tests: exact reference E_t (recursion) for target,
  fixed-budget and portfolio ECDFs; the summary's chains average to E_t; cocopp
  within 5 of its standard errors for both ECDFs.

## Next step
Update `docs/definitions/ecdf.md` "Implementation (I33)" with the cocopp numbers,
write `docs/notes/simulated-restarts.md` (step 8) with the timings, test on
polars<2.

## Decisions
- The deprecated `get_aocc`, `get_data_ecdf` (and `plot_ecdf`) stay without
  restarts, as in 0.0.8.
- No `n_restarted` for the target ECDFs; the fixed-budget results keep it.
- Before timings: `scripts/bench_restarts.py --repeat 1` on the code of #48 (HEAD
  1668939), exported to the scratchpad.

## Open questions / Blocked
- (none)
