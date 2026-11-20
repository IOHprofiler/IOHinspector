# WP I22: Performance pass
Branch: wp/I22-performance
Status: in progress

## Done
- step 1: scripts/bench.py (was bench_ert.py): synthetic full bbob written as
  format-v1 datasets (one per algorithm) or --datasets DIR of converted
  datasets; cases through DataManager and the website's metrics.

## Next step
Step 2: vectorise restarts.continued (per (run set, k) polars loop) and the
ECDF step function; then DataManager pushdown and manifest validation cache.

## Decisions
- bench_ert.py became bench.py (one harness); bench_shapley imports synthetic().
- Real data: 13 bbob archives of the COCO archive (every 20th of bbob/, not
  incomplete/), converted with `iohinspector convert --format coco`; kept in the
  session scratchpad, not committed. Baseline = src of main via PYTHONPATH.

## Open questions / Blocked
- none
