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

## Next step
Step 2b: the ECDF step function (sort, cum_sum over the group, last per x);
then DataManager pushdown and manifest validation cache. Rebuild the COCO data
in /Users/jacob/code/ioh-work/scratch/I22 first (the /tmp copy was lost).

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
