# Performance

How fast iohinspector computes what the website shows, how to measure it, and
where the limits are. The targets of the website plan, on its 64-core server
with `POLARS_MAX_THREADS=16`, are the full ERT table in under 5 s and a
single-function view in under 0.3 s.

## Measuring

`scripts/bench.py` times the cases a website request runs: opening the datasets
with `DataManager`, selecting and loading, and the metrics on the loaded data.

```text
python scripts/bench.py [--algorithms 258] [--rows 40] [--out DIR]
                        [--datasets DIR] [--pair F D] [--repeat 3]
                        [--only CASE ...] [--json FILE]
```

Without `--datasets` it writes a synthetic set the size of full bbob (258
algorithms, 24 functions, 6 dimensions, 15 runs each, 40 improvements per
run), one format-v1 dataset per algorithm as on the website. With
`--datasets DIR` it uses every converted dataset in `DIR`, such as the COCO
archive converted with `iohinspector convert --format coco`. `all:` cases use
every (function, dimension) pair, `pair:` cases the one pair `--pair`. The
script prints a Markdown table with the best of `--repeat` runs.

The benchmark holds all the data in memory. Start with a few algorithms
(`--algorithms 20`) and watch the memory before you run the full size: the
synthetic set with 80 algorithms already peaks at 4.5 GB.

## Results

Measured on an Apple M5 laptop (10 threads, polars 2.1.0), in seconds.

Synthetic data, 20 algorithms (`syn20`), before and after the performance
pass (I22):

| case | before | after |
|---|---:|---:|
| all: load | 0.074 | 0.074 |
| all: ERT table | 0.216 | 0.235 |
| all: fixed-budget table | 0.191 | 0.208 |
| all: ECDF per pair, sampled | 4.810 | 1.081 |
| all: ECDF per pair, sampled, carry forward | 0.555 | 0.567 |
| pair: select and load | 0.035 | 0.032 |
| pair: ECDF | 0.046 | 0.019 |
| pair: ECDF, sampled | 0.046 | 0.018 |
| pair: ECDF AUC | 0.046 | 0.018 |
| pair: fixed-budget ECDF | 0.030 | 0.034 |

With 80 algorithms (after): load 0.28, ERT table 1.21, fixed-budget table 1.05,
sampled ECDF of all pairs 6.3 (peak 4.5 GB), each `pair:` case under 0.17.

Converted COCO data: 13 bbob algorithms of the COCO archive, f24 in dimension
2, where runs stop after up to 2·10^7 evaluations. The ECDFs run up to the
largest evaluation (the default):

| case | before | after |
|---|---|---|
| open 13 datasets, select f24 d2, load | 1.11 s | 0.11 s (manifests known valid) |
| ECDF, 50 samples | over 8 GB | 2.6 s, 0.7 GB |
| continuum ECDF, 50 samples | over 8 GB | 4.5 s, 0.8 GB |
| ECDF AUC | over 8 GB | 5.6 s, 0.9 GB |
| fixed-budget ECDF | 9.8 s, 3.6 GB | 7.4 s, 0.8 GB |

The results are the same as before up to rounding (largest absolute
difference 1.4·10^-13 over the ECDF, AUC, AOCC and fixed-budget ECDF cases
of f24 d2).

## What the pass changed

- **Simulated restarts** (EC-7, [definitions](definitions/ecdf.md)). The
  renewal pass that computes the restarts' increments runs in windows of
  `restarts.CHUNK` evaluations and keeps only as much history as the largest
  failing budget, so memory no longer grows with the horizon. A sampled ECDF
  (`x_values`) sums the restarts' increments at the samples, and the AUC sums
  them weighted by the part of [b_lo, b_hi] each one covers. Neither builds
  the step function, which has one row per evaluation once restarts begin.
- **Opening datasets.** `read_manifest` remembers the digests of the manifests
  it has found valid. The schema check of a manifest with one entry per source
  file (4 MB for a COCO algorithm) took about 80 ms, far more than reading it.
- **Selecting.** `DataManager.select` also filters the trace and positions by
  the selected runs' functions and dimensions. Row-group statistics can skip
  on these columns, but not on run ids, which are hashes.
- **Threads.** polars uses `POLARS_MAX_THREADS` (it must be set before polars
  is imported); iohinspector starts no thread pools of its own. With 2
  threads the synthetic ERT table takes 0.42 s instead of 0.24 s. The
  restarts' renewal pass runs in numpy on one thread and does not change.

## Limits

- **Restarts over long horizons.** The ECDF with restarts is exact, so its
  renewal pass costs time in proportion to the horizon (in evaluations) times
  the number of distinct budgets of the failing runs, for each curve. The
  memory stays bounded, but the time does not: on the COCO data, all 24
  functions in dimension 20 up to 2·10^8 evaluations take 160 s (3.4 GB), and
  f24 in dimension 40 up to 4·10^8 takes 45 s. The sampled ECDF of every pair
  of the COCO data at once, up to 4·10^8, exceeds 8 GB. Python and numpy hold
  only about 300 MB of that, so where the rest goes is still open. Such views
  need either a shorter horizon or a faster algorithm for the renewal.
- **Shapley values** (`metrics/portfolio.py`) still build the restarted step
  function, because the portfolio's ECDF is the maximum over its players at
  each x.
- **Per-curve loop.** The renewal pass loops in Python over curves (a group,
  a pair and a number of failing runs), not over runs: 29,603 curves on the
  synthetic 20-algorithm set, about half of the 1.1 s of its sampled ECDF.
