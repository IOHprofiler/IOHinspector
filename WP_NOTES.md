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

- step 3: DataManager. Opening the 13 COCO datasets spent 1.0 s of 1.1 s in the
  JSON-schema check of their 4 MB manifests (one entry per source file):
  read_manifest now remembers the blake2b digests of manifest bytes it found
  valid, so a process opens them again in 0.11 s (open + select f24 d2 + load).
  select() also filters the trace and positions by the selected runs'
  function_id and dimension, which row-group statistics can skip on (run ids
  are hashes): one COCO trace file 1.4 ms to 0.7 ms; select + load of a pair
  was already 0.08 s. POLARS_MAX_THREADS is honoured (pl.thread_pool_size()
  after import); iohinspector starts no thread pools of its own.

- step 4: docs/performance.md: how to run scripts/bench.py, results on
  synthetic (20 and 80 algorithms, before/after) and COCO f24 d2, what changed,
  and the limits (restarts over long horizons, Shapley, per-curve loop).

## Next step
Step 5: full test suite through memguard, PR body from these notes, delete
WP_NOTES.md, CI, D6 review request (dvermetten).

## Decisions
- bench_ert.py became bench.py (one harness); bench_shapley imports synthetic().
- Real data: 13 bbob archives of the COCO archive (every 20th of bbob/, not
  incomplete/), converted with `iohinspector convert --format coco`; kept in the
  /Users/jacob/code/ioh-work/scratch/I22, not committed.
- Every benchmark, profile and full-data run goes through plan/tools/memguard.py
  (AGENTS.md, Memory): on 20 Nov three concurrent runs reached 82 GB and froze
  the laptop. The step-function dump of f24 d2 alone was 1 GB of parquet. Baseline = src of main via PYTHONPATH.

## Open questions / Blocked
- The exact restarts cost time linear in the horizon: all COCO d20 pairs up to
  2e8 take 160 s, and every COCO pair at once up to 4e8 exceeds 8 GB (numpy
  holds only ~300 MB of it; the rest is not yet explained). A faster renewal or
  a capped horizon for such views needs a decision (EC-7 is exact by design).
