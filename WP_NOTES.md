# WP I22: Performance pass
Branch: wp/I22-performance
Status: in progress

## Done
- step 1: scripts/bench.py (was bench_ert.py): synthetic full bbob written as
  format-v1 datasets (one per algorithm) or --datasets DIR of converted
  datasets; cases through DataManager and the website's metrics.
- step 2a: restarts.continued vectorised (renewal in blocks of the smallest
  budget, chains trimmed by length): identical rows to main on synthetic and
  COCO f24 d2 / f1 d20, 0.98 s to 0.13 s on synthetic. Committed by the
  controller after the 20 Nov crash (the worker had verified it, not committed).
- step 2b: the restarts' renewal pass in windows of CHUNK evaluations (memory
  no longer grows with the horizon), and get_ecdf with x_values sums the
  restarts' increments at the samples instead of building the step function.
  Same rows as main on COCO f24 d2 (scratch/I22/scripts/dump.py, six cases:
  sampled, sampled/dim, exact, auc, continuous, fixed budget): no unmatched
  rows, max abs diff 1.4e-13. Full horizon (2e7) on f24 d2, slice.py:
  sampled 2.6 s / 0.7 GB, continuous 4.5 s / 0.8 GB (main: both killed above
  8 GB); fixed-budget ECDF 7.4 s / 0.8 GB (main 9.8 s / 3.6 GB).

- step 2c: get_ecdf_auc with restarts sums each chunk's restart increments
  times the part of [b_lo, b_hi] they cover (continued(weighted=...)), placed
  at x = 0, instead of building the step function. Same as main on COCO f24
  d2 (auc, log axis, divided by dimension, AOCC): max abs diff 1.4e-14. Full
  horizon (2e7) f24 d2 AUC: 5.6 s / 0.9 GB (main killed above 8 GB). Shapley
  (portfolio.py) still builds the full restarted steps: it needs the max over
  players at each x.

## Next step
Step 3: DataManager predicate pushdown (check that select() filters reach the
parquet scan, and POLARS_MAX_THREADS is honoured) and a cache of the manifest
validation; then step 4, docs/performance.md from scripts/bench.py runs on
synthetic data and scratch/I22/coco-ds (results above, under step 2b and 2c).

## Decisions
- bench_ert.py became bench.py (one harness); bench_shapley imports synthetic().
- Real data: 13 bbob archives of the COCO archive (every 20th of bbob/, not
  incomplete/), converted with `iohinspector convert --format coco`; kept in the
  /Users/jacob/code/ioh-work/scratch/I22, not committed.
- Every benchmark, profile and full-data run goes through plan/tools/memguard.py
  (AGENTS.md, Memory): on 20 Nov three concurrent runs reached 82 GB and froze
  the laptop. The step-function dump of f24 d2 alone was 1 GB of parquet. Baseline = src of main via PYTHONPATH.

## Open questions / Blocked
- none
