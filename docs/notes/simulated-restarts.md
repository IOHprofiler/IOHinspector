# Simulated restarts: the cost problem and how it was solved

A note for the review of IOHinspector#48 (WP I33). The definitions are in
[`docs/definitions/ecdf.md`](../definitions/ecdf.md), section "Simulated
restarts" and EC-7. This note covers the cost problem reported in #48, what
caused it, and what changed.

## The problem

The first version of #48 restarted the runs of the fixed-budget ECDF by
simulating them. It reused FB-6's procedure from `aggregate_convergence`:
every run that stopped before the largest budget got one chain. The chain
appended runs drawn at random from the run set until the largest budget, and
then the chain's trace was aligned at the budgets. The pull request reported
that budgets far beyond the runs were slow, for example 10⁶ on the `coco`
fixture (BFGS on f1 in 2-D, 15 runs that all reach 10⁻⁸ within 13 to 25
evaluations).

Timings on the code of #48 (commit 1668939, Apple M5, one run each, polars 2.1;
`scripts/bench_restarts.py`, ten budgets from 10 to 10⁶):

| Case | Runs (chains) | Seconds |
|---|---|---|
| `get_fixed_budget_ecdf`, `coco` | 15 | 0.40 |
| the same, each run copied 10 times | 150 | 4.1 |
| the same, each run copied 30 times | 450 | 21.7 |
| the same, each run copied 100 times | 1500 | 702.6 |
| `aggregate_convergence(stopped="restart")`, `coco` | 15 | 0.37 |

One call on 15 runs is fast. The cost becomes a problem when the curve has to
be accurate. A simulated ECDF has a Monte Carlo error of about
1/√(chains), so the validation against cocopp copied every run 1000 times.
On `coco` that would have taken hours. The case with "over 3 minutes" was of
this kind.

## The cause

The cost is chains × restarts per chain × rows per run. A chain on `coco`
needs about 10⁶ / 20 = 50 000 restarts. Two things made this needlessly
expensive.

1. **Runs that had already succeeded were restarted.** Every run on `coco`
   reaches every bbob target, so no restart can change the target ECDF. Yet
   each chain ran on to 10⁶, because the procedure restarted every run that
   stopped before the largest budget, whatever it had reached. (In the
   fixed-budget ECDF, restarts matter only for values below a run's own final
   value that some other run reaches.)
2. **The restarts were simulated, although the ECDF only needs their
   distribution.** A restarted run reaches t within b with a probability that
   follows directly from the runs' runtimes (below). Sampling it costs
   accuracy, needs a seed, and its cost grows with the number of chains.

## The solution

**Method (step 7).** For a run set of n runs and a target t, let F be the
runs that do not reach t. A restarted run reaches t within b with probability

    E_t(b) = (1/n) Σ_{i ∉ F} 1[T_i(t) ≤ b]  +  (1/n) Σ_{j ∈ F} E_t(b − B_j).

The first term is the ordinary single-target ECDF of the run set. The second
continues each failed run's share 1/n after its B_j evaluations. A run that
reaches t is never restarted for it, and a target that no run reaches stays
at 0. This is the expected value of cocopp's `drawSP` and of FB-6's chains,
so the ECDF calculation is adapted directly and nothing is simulated:

- Over the integer evaluations the increments r of E_t solve
  r = h + (1/n) Σ_{j ∈ F} r(· − B_j), with h the increments without restarts.
  One pass over the evaluations computes this, in blocks of the smallest B_j.
  The pass is linear in h, so all targets with the same F are continued in one
  pass.
- A run fails t exactly when its final value is worse than t. So F is always
  the k runs with the worst final values, and a run set has at most n
  different F.
- The pass stops at the horizon, or earlier where the chance that a chain
  still has not succeeded, at most (|F|/n)^k after k restarts, is below 10⁻¹⁷.
  On `coco` that is at most about 14 000 evaluations, not 10⁶.
- The fixed-budget ECDF uses the duality V(b) ≤ v ⇔ T(v) ≤ b: at a value v
  it is (1/m) Σ_k E_v(b_k). It is computed from pieces of the runs' traces,
  each piece a range of values that a run reaches at one evaluation with a
  fixed F.

**Scope (step 5).** The same computation is now used by every ECDF:
`get_ecdf` and `get_ecdf_auc` (targets and continuum, so also the AOCC), the
portfolio game behind `get_shapley_values`, and `get_fixed_budget_ecdf` and
its AUC. All of them take `stopped="restart"` (the default) or
`"carry_forward"`. The deprecated `get_aocc` and `get_data_ecdf` keep their
0.0.8 behaviour without restarts.

**Seed (step 6).** The ECDFs no longer need a seed: they are the same on every
call. Simulation is still needed in one place: `aggregate_convergence` with
`stopped="restart"`, because its statistics (mean, quantiles, best, worst)
need one value per run. Its seed is now 0 by default. A run whose final value
is the best of its run set is no longer restarted, and a chain stops once it
has drawn such a run, because no later restart can improve it.

## Timings after

Same machine and script (`python scripts/bench_restarts.py`, best of three):

| Case | Before (s) | After (s) |
|---|---|---|
| `get_fixed_budget_ecdf`, `coco`, budgets to 10⁶ | 0.40 | 0.015 |
| the same on 1500 copied runs | 702.6 | 0.08 |
| `aggregate_convergence(stopped="restart")`, `coco` | 0.37 | 0.004 |
| `get_fixed_budget_ecdf`, `coco_old` (2-D and 3-D) | 0.007 | 0.031 |
| `get_ecdf` at the 51 bbob targets, `coco_old`, sampled to 10⁶ | – (no restarts) | 0.020 |

The copies are no longer needed, because the result is exact. The cocopp
comparison now compares each cell's expected value with cocopp's 20000
samples, within 5 of cocopp's standard errors. The largest difference is
0.0054 (2-D) and 0.0056 (3-D); in the first version of #48 it was 0.0074 and
0.0106. On `coco_old` the exact computation is slower than one simulated draw
(0.03 s against 0.007 s), because it covers every restart up to 10⁶ instead
of one sample of them.

## What is left

The pass makes one block step per smallest failed budget B_j, so its cost
grows with (horizon) / (smallest B_j). It stays small unless a run set mixes
failed runs that stop almost at once with failed runs that use the whole
budget, and the horizon lies far beyond that budget. A constructed worst
case, with one successful run, 13 failed runs of 10⁵ evaluations and one
failed run of a single evaluation, takes 1.8 s for the fixed-budget ECDF at
budgets up to 10⁵ and 26 s at budgets up to 10⁶ (the target ECDF: 0.14 s and
2.0 s). If this matters in practice, the passes for the different F of a run
set can be run as one array operation.
