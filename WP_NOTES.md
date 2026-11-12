# WP I11: Runtime and fixed-budget summaries
Branch: wp/I11-summaries (stacked on wp/I10-alignment, IOHinspector#32)
Status: ready for review

## Done
- step 1: runtime summary (RT-1..RT-6), fixed-budget summary with carry-forward
  and seeded simulated restarts (FB-1..FB-6), overview table (FB-4); golden tests
  against IOHanalyzer (`RT_summary_par1/10`, `FV_summary_geom`, `FV_overview`,
  `RT_overview_par1`, `overview_par1` on ioh040_min/max) and cocopp (aRT on the
  COCO fixture).
- step 2: full-bbob ERT benchmark (`scripts/bench_ert.py`), CHANGELOG, pointers
  from the definition pages.

## Next step
None: hand to the controller.

## Decisions
- API: the existing `aggregate_running_time` (metrics/fixed_target.py) and
  `aggregate_convergence` (metrics/fixed_budget.py) are the summaries; new
  `get_overview` in fixed_budget.py. The plan's module name `summary` was not
  introduced, to keep one function per view. Columns are the definitions' names:
  `n_runs`, `n_success`, `success_ratio`, `ERT`, `par_{k}`, `mean`, `median`, `sd`,
  `q2`..`q98`; fixed budget `best`, `worst`, `geometric_mean`, `n_defined`,
  `n_active`, `n_restarted`. Renamed columns are listed in the CHANGELOG; plots fail
  on an unknown measure name.
- B_i is `evals_used` when the frame has it (`load(include_columns=["evals_used"])`),
  else the run's last logged evaluation (documented in the docstrings).
- `eval_max` of `aggregate_running_time` is the budget cap C (RT-1, RT-3).
- Simulated restarts draw from the `by` group (the run set); one chain per stopped
  run, uniformly with replacement including the run itself, `numpy` default_rng(seed).
- Shared helpers, in `metrics/utils.py` (where I10 put `group_label`): `QUANTILES`,
  `runs_of` (one row per run with `by`, `evals_used` and an integer `run_set` key)
  and `name_run_sets`. Metrics align on `run_id` alone and group on `run_set`:
  aligning on `run_id` plus `by` took 11 s of the benchmark, now 2.2 s.
- Golden comparisons with IOHanalyzer set `evals_used` to the largest logged
  evaluation of the run set (its budget rule, as the runtime page's test tolerance
  says). cocopp rounds 3 of the 51 targets to its next stricter grid level
  (alignment.md); the test compares those with our next target.

## Benchmark
`python scripts/bench_ert.py`: 557,280 runs (258 algorithms x 24 functions x 6
dimensions x 15 runs), 22.3M trace rows, ERT table at the 51 bbob targets
(1.9M rows). On a 10-core laptop: read 0.1 s, table 4.9 to 5.3 s; 2.2 s of it is
`align_data`. Not run on the 64-core server (workers do not run commands there).

## Open questions / Blocked
- The 5 s target is met only borderline on the laptop; the controller should run
  `POLARS_MAX_THREADS=16 python scripts/bench_ert.py` on the server.
