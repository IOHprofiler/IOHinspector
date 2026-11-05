# Runtime metrics (fixed target)

Status: **proposed**, for sign-off at checkpoint H1.
Parity-matrix rows: "ERT, PAR-k, runtime summary with quantiles"; the runtime
half of "Runtime and value samples".
Implemented in: I10 (hitting times), I11 (summaries).
Sources: [IOHanalyzer], [cocopp], [iohinspector]; see [references](references.md).

## Notation

- A **run set** S is every run of one algorithm on one (function, dimension)
  pair, pooled over instances. n = |S|.
- B_i is the number of evaluations run i used. It comes from the run's
  metadata (`evals` in IOHexperimenter JSON, `evals_used` in storage format
  v1), not from the last logged row, because a logger that records
  improvements only may stop logging long before the run ends.
- T_i(t) is the **first hitting time** of target t by run i: the evaluation
  count at which the run first reaches t, or ∞ if it never does. The hit rule
  (f ≤ t when minimising, f ≥ t when maximising) and the handling of ties are
  defined in [alignment](alignment.md).
- C is an optional **budget cap** chosen by the user (IOHanalyzer's `budget`
  argument). Without a cap, C = ∞.
- Run i is a **success** for t when T_i(t) ≤ C. s_i(t) ∈ {0, 1}.
  n_s(t) = Σ_i s_i(t) is the number of successes.
- A failed run is **charged** u_i = min(B_i, C) evaluations.

## Definitions

### Success count and success rate

n_s(t) as above, and the success rate p_s(t) = n_s(t) / n. IOHanalyzer calls
the rate `ps`; iohinspector calls it `success_ratio`. They are the same
quantity.

### ERT (aRT)

    ERT(t) = ( Σ_{i: s_i = 1} T_i(t)  +  Σ_{i: s_i = 0} u_i ) / n_s(t)

Equivalently ERT = mean(T | success) + (1 − p_s)/p_s · mean(u | failure). It
estimates the expected number of evaluations to reach t when failed runs are
restarted independently with the same budget. The estimator goes back to
[Price1997] and [Auger2005]. It is the ERT of the BBOB workshops
[Hansen2010], renamed aRT ("average runtime") in COCO [Hansen2016pa,
Hansen2021, Hansen2022]. cocopp's docstring calls it "the so-called success
performance which estimates the expected runtime ERT" (`cocopp/toolsstats.py`
119–123). Each failed run is charged **its own** evaluations: cocopp adds
`maxevals` of the unsuccessful runs only (`cocopp/pproc.py` 1172–1181).

- **No success** (n_s = 0, n > 0): ERT = +∞. All three implementations agree
  (R `x / 0`; `cocopp/toolsstats.py` 169–173; polars float division).
- **No runs** (n = 0): undefined; return null.

### PAR-k

    PAR-k(t) = (1/n) Σ_i ( T_i(t) if s_i = 1, else k · κ )

Penalised average runtime from algorithm configuration [Hutter2009]: runs that
miss the cutoff κ count as k times the cutoff, k = 10 being the common choice.
The cutoff is one number for the whole run set: κ = C when a cap is set, and
otherwise κ = max_i B_i. With k = 1 and equal budgets, PAR-1 is cocopp's
`detAverageEvals` (= ERT · p_s, `cocopp/pproc.py` 1792–1824).

### Mean, median and standard deviation

These describe the **conditional** distribution of the runtime given success.
They are computed over the successful runs only, {T_i(t) : s_i = 1}:

- mean: arithmetic mean; null when n_s = 0;
- median: the conventional sample median (mean of the two middle values when
  n_s is even, which equals Hyndman–Fan type 7 at p = 0.5); null when n_s = 0;
- sd: sample standard deviation with denominator n_s − 1; null when n_s < 2.

The table always shows n and n_s next to them, so a reader can see how many
runs the statistics rest on.

### Quantiles

Quantiles of the successful runtimes use Hyndman–Fan **type 3** [Hyndman1996]
(nearest even order statistic, SAS definition 2) at the probabilities
2, 5, 10, 25, 50, 75, 90, 95 and 98 %. Type 3 always returns an observed run
length. numpy's `method="closest_observation"` is the same estimator. This was
checked against R's `quantile(type = 3)` on random integer samples of size 1 to
26 at those nine probabilities; numpy `inverted_cdf`, `linear` and
`median_unbiased` matched R types 1, 7 and 8 in the same check. polars has no
type 3, so I11 must use numpy or an explicit formula. The `q50` column (type 3)
and the `median` column (type 7) can differ by half the gap between the two
middle values; both are kept.

## How IOHanalyzer implements it

- Hitting times come from the RT matrix. Its rows are every distinct observed
  f value of the run set; its entries are the first evaluation whose
  best-so-far value satisfies the hit rule (`src/align.cc` 76–107,
  `c_impute_running_time`). For IOHexperimenter JSON data the matrix is built
  in `R/readFiles.R` 1594–1600; for old `.dat` data in `align_running_time`,
  `R/readFiles.R` 754–809. A target t is looked up as the first row whose f
  value is ≤ t (≥ when maximising): `R/DataSet.R` 888–900 (`get_ERT.DataSet`)
  and 921–936 (`get_RT_summary.DataSet`).
- `SP()` (`R/stats.R` 12–25) computes ERT, successes and success rate. A run
  fails when its entry is NA or larger than `max_runtime`. A vector of budgets
  is collapsed to its maximum (line 14–16), and every failed run is charged
  that maximum (line 20).
- `get_RT_summary.DataSet` (`R/DataSet.R` 910–966): `maxRT` is the largest
  per-run budget, or the user's `budget` (913–917). mean, median and sd use
  `na.rm = TRUE`, so they cover the runs that reached t (947–949). Quantiles
  use `IOHanalyzer_env$D_quantile` (940), which is `quantile(type = 3,
  na.rm = TRUE)` (`R/IOHanalyzer.R` 117–133) at the probabilities set in
  `R/IOHanalyzer.R:71`. PAR-k replaces NA and values above `maxRT` by
  `maxRT * IOHanalyzer.PAR_penalty` and averages over all runs (937–939, 950).
  The penalty option defaults to 1 and is set from the Shiny input
  `ERTPlot.ParX` (`inst/shiny-server/server/ERTPlot.R:16`, default value 1 in
  `inst/shiny-server/ui/ERT_box.R` 56–57).
- The output column `runs` is the number of **successes** (`SP()$runs`), not
  the number of runs (`R/DataSet.R` 943, 962).
- Budgets of IOHexperimenter JSON data: the JSON reader stores the per-run
  `evals` (`R/readFiles.R` 114–115), but `read_IOH_v1plus` then overrides
  `maxRT` with `max(runtimes)`, the largest **logged** evaluation over all
  runs (`R/readFiles.R:1638`).

## How cocopp implements it

- ERT: `DataSet.computeERTfromEvals` (`cocopp/pproc.py` 1164–1192) and
  `toolsstats.sp` (`cocopp/toolsstats.py` 119–177). Unsuccessful runs add their
  own `maxevals`. `maxevals` is the evaluation count of a run's last line, the
  larger of the `.dat` and `.tdat` values (`cocopp/pproc.py` 980–985).
- Success rate: `detSuccesses` counts finite entries, `detSuccessRates`
  divides by `nbRuns()` (`cocopp/pproc.py` 1826–1858).
- Instances: runs of all instances are pooled. By default cocopp **balances
  instances** by duplicating runs of under-represented instances so that each
  instance has equal weight (`cocopp/genericsettings.py` 49–52,
  `instance_multipliers` in `cocopp/pproc.py` 2145–2195).
- Targets are expressed as precision Δf = f − f_opt, and a target between
  two points of cocopp's internal grid (5 per decade for bbob) is rounded down
  to the next stricter grid level (`cocopp/pproc.py` 1912–1916). See
  [alignment](alignment.md).
- Dispersion in tables: the 10th and 90th percentiles of runtimes from
  simulated restarts (`cocopp/pptable.py` 248–253), with cocopp's own
  percentile function, Hyndman–Fan type 5 (`cocopp/toolsstats.py` 520–544:
  position −0.5 + pN, linear interpolation). These are percentiles of a
  bootstrapped runtime distribution, not a confidence interval for ERT, and
  are not part of the IOHanalyzer parity list.

## How iohinspector 0.0.8 implements it

`aggregate_running_time` (`metrics/fixed_target.py` 8–95):

- targets: 50 log-spaced values between the data minimum and maximum (45);
- ERT: successes' runtimes plus `eval_max` per failure, divided by successes
  (68–74), with `eval_max = data["evaluations"].max()` over the **whole frame
  passed in**, all algorithms and functions together (57–58);
- mean, median, std: over all runs including the `inf` of failed runs (61–65),
  so the mean is `inf` and the std NaN whenever one run fails;
- PAR-10 only, with penalty `10 * eval_max` (75–81);
- no quantiles.

Hitting times come from `align_data` (`align.py` 8–67); see
[alignment](alignment.md) for its tie behaviour.

## Differences

| Item | IOHanalyzer | cocopp | iohinspector 0.0.8 | Proposed |
|---|---|---|---|---|
| Charge of a failed run in ERT | max budget in the run set (JSON data: max logged evaluation) | own `maxevals` | max evaluation in the whole frame | own B_i, capped at C |
| ERT with no success | Inf | inf | inf | +∞ |
| Instance weighting | runs pooled | instances balanced | runs pooled | runs pooled |
| PAR-k cutoff | max budget in the run set, k = 1 default | – | 10 × max evaluation in the frame | κ = C or max_i B_i; k a parameter |
| mean, median, sd | successful runs | – | all runs (inf) | successful runs |
| Quantiles | type 3, successful runs | type 5 (other use) | none | type 3, successful runs |
| `runs` column | number of successes | – | – | separate `n_runs` and `n_success` |

## Proposed decisions

- **RT-1.** A failed run is charged its own evaluations, min(B_i, C), with B_i
  from the run metadata. This is the COCO definition and differs from
  IOHanalyzer whenever budgets differ between runs, and for all
  IOHexperimenter JSON data that log improvements only.
- **RT-2.** ERT with no success is +∞, shown as "∞"; with no runs it is null.
- **RT-3.** PAR-k uses one cutoff κ per run set: C when set, otherwise
  max_i B_i. k is a parameter. Python default k = 10 (the usual PAR-10); the
  website shows k with default 1, as IOHanalyzer does. The column is named
  `par_{k}`.
- **RT-4.** mean, median and sd are conditional on success and computed over
  successful runs only, next to `n_runs` and `n_success`.
- **RT-5.** Runtime quantiles are Hyndman–Fan type 3 over successful runs, at
  2, 5, 10, 25, 50, 75, 90, 95 and 98 %. The median stays the conventional
  sample median.
- **RT-6.** Runs are pooled over instances without balancing. With the usual
  design (equal runs per instance) this equals cocopp's balanced value.

## Open questions for the user

- Q-RT-1: Should the Python and website defaults for k agree (RT-3)?
- Q-RT-2: Should the summary also offer unconditional quantiles, which treat
  failed runs as ∞ and so match the ECDF (the p-quantile is ∞ when p > p_s)?

## Test tolerance

Counts (n, n_s) and quantiles (observed run lengths) must match exactly.
ERT, PAR-k, mean, median and sd must agree to a relative error of 1e-12.
Golden comparisons against IOHanalyzer use the IOHanalyzer budget rule
(largest budget in the set) for ERT and PAR-k, and the test names RT-1 where
the agreed value differs.
