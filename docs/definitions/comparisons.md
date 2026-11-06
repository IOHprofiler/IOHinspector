# Statistical comparisons: KS tests, Holm correction, win fractions

Status: **proposed**, for sign-off at checkpoint H1.
Parity-matrix rows: "KS tests with Holm correction", "Win-fraction heatmap"
(and the dominance graph drawn from the same tests).
Implemented in: I15.
Sources: [IOHanalyzer], [R], [iohinspector]; see [references](references.md).
Glicko-2 has its own page, [glicko2](glicko2.md).

## Notation

- One (function, dimension) pair, algorithms a = 1..N, and one target t
  (fixed target) or budget b (fixed budget).
- X_a: the sample of algorithm a. Fixed target: the runtimes T_i(t) of its
  runs, a failed run counting as +∞. Fixed budget: the values V_i(b).
- "a is better than c": a's runtimes are stochastically smaller; for values,
  smaller when minimising and larger when maximising.

## Definitions

### One-sided two-sample KS test

For an ordered pair (a, c), with X_a, X_c oriented so that smaller is better:

- H1: a is better than c, i.e. F_a(x) > F_c(x) for some x (F the empirical
  CDFs);
- statistic D⁺ = sup_x (F_a(x) − F_c(x)), with sample sizes n and m;
- p-value, asymptotic [Smirnov1948]:
  p = exp(−2 · (n m / (n + m)) · (D⁺)²).

This is R's `ks.test(x, y, alternative = "greater", exact = FALSE)`
(`stats:::psmirnov_asymp`, R 4.3.3: `1 − exp(−2 n q²)` with n = nm/(n+m),
returned as an upper tail). For the fixed-target case, failed runs count as
+∞, so F_a(x) is the single-target ECDF of [ecdf](ecdf.md) and D⁺ is the
largest vertical gap between two such curves. Under maximisation of values the
samples are negated first.

### Holm correction

Per (pair, target): all ordered pairs (a, c), a ≠ c, that have a p-value form
one family, N(N − 1) tests at most; p-values are adjusted with Holm's step-down
method [Holm1979], as R's `p.adjust(method = "holm")`. Missing p-values do not
count towards the family size (checked: `p.adjust(c(0.01, NA, 0.04, 0.03),
"holm")` gives 0.03, NA, 0.06, 0.06).

### Decisions, heat map and dominance graph

With significance level α (default 0.01): S_ac = 1 if the adjusted p_ac ≤ α.
The heat map shows S_ac − S_ca ∈ {−1, 0, 1}; the dominance graph has an edge
a → c when S_ac = 1. If X_a has no defined values and X_c does, p_ac = 1 and
p_ca = 0 before adjustment; if both are empty, both are missing.

### Win-fraction matrix

For each pair and each of its targets (or budgets), algorithm a wins against c
when its summary is better: ERT for fixed target ([runtime](runtime.md)),
the mean value for fixed budget ([fixed-budget](fixed-budget.md)). A win scores
1, a tie (including ∞ against ∞) 0.5, a loss 0. W_ac is the mean score over
the targets of a pair, then over the pairs with equal weight (as EC-2).

## How IOHanalyzer implements it

- `pairwise.test.list` (`R/stats.R` 89–136) loops over unordered pairs and
  runs `ks.test(x1, x2, alternative = 'greater', exact = F)` for p[i, j] and
  `alternative = 'less'` for p[j, i] (123–126). The documented meaning of
  p[i, j] is "the running time of algorithm i is smaller" (`R/stats.R` 71–72).
  All N(N − 1) p-values are then Holm-adjusted together (132–133). An
  algorithm whose sample is all NA gets p = 1 against, 0 for the other
  (108–120).
- `pairwise.test.DataSetList` (`R/stats.R` 143–168): fixed target uses
  `get_RT_sample`, where failed runs are NA. With `bootstrap.size = 0`, the
  Shiny default (`inst/shiny-server/ui/stats_box.R` 13–15), `ks.test` drops
  NA values, so **failed runs are ignored**: an algorithm with one fast success
  in 15 runs can be declared better than one that always succeeds. With
  `bootstrap.size > 0`, `bootstrap_RT` (`R/stats.R` 41–62) draws simulated
  restarts: a geometric number of failed budgets plus one successful runtime,
  without a seed. Fixed budget uses `get_FV_sample` with no bootstrap, and the
  same `'greater'` direction whatever the optimisation direction, so under
  **maximisation the result is reversed** (`inst/shiny-server/server/fv_stat_tests.R:22`
  calls it without negating).
- Heat map and graph (`R/plotDataSetList.R` 1259–1293 and 1343–1380):
  `p_matrix <= alpha` (1272, 1362), then `y - t(y)` for the heat map and an igraph
  adjacency matrix for the graph; α defaults to 0.01 in Shiny
  (`inst/shiny-server/ui/stats_box.R` 10–12, `fv_stats_box.R` 10–12).
- Win fractions, `generate_data.Heatmaps` (`R/DataSetList.R` 1855–1941): per
  (function, dimension, target) it compares the `mean` column of
  `get_RT_summary`, which is the mean of **successful** runs only (see
  [runtime](runtime.md)), or of `get_FV_summary`; NA becomes ±∞; wins score
  1, ties 0.5. Two defects: the fixed-target comparison ignores failures, and
  the final `array(unlist(res), dim = c(n_ids, n_ids, length(res)))` (1914–1915) sizes the
  array by the number of functions while `res` holds one matrix per
  (function, dimension), so with more than one dimension only the first
  matrices are averaged.

## Why one tool gave 0.146 and another 0.226

The plan reports one-sided KS p-values of 0.146 and 0.226 "on the same input"
without recording the input. iohinspector 0.0.8 has no KS test, so the second
value came from another implementation, presumably scipy. The p-value method
alone moves the result by that much. For a = (10, 12, 15, 30, 40) and
c = (14, 35, 45, 50, 60), D⁺ = 0.6 and:

| Implementation | p |
|---|---|
| R 4.3.3 `ks.test(..., "greater", exact = FALSE)` (IOHanalyzer) | 0.165299 = exp(−2 · 2.5 · 0.36) |
| R 4.3.3 `ks.test(..., "greater", exact = TRUE)` | 0.178571 = C(10, 2)/C(10, 5) |
| scipy 1.18.1 `ks_2samp(..., "greater", method="exact")` | 0.178571 |
| scipy 1.18.1 `ks_2samp(..., "greater", method="asymp")` | 0.090718 |

scipy's "asymp" uses the one-sample one-sided distribution with
n = nm/(n+m), not the limit exp(−2nD²), and scipy's default for small samples
is "exact". So the same data give three p-values; I15 must hand-code R's
formula to match IOHanalyzer.

## How iohinspector 0.0.8 implements it

No KS test, no Holm correction, no win-fraction matrix. `metrics/ranking.py`
offers robust-ranking bootstrap comparisons (`get_robustrank_over_time`,
`get_robustrank_changes`), which are outside the parity list.

## Differences

| Item | IOHanalyzer | Proposed |
|---|---|---|
| KS p-value | asymptotic one-sided | same; exact as an option |
| Failed runs (fixed target) | dropped (default) or simulated restarts | +∞ |
| Maximisation (fixed budget) | direction reversed | samples negated |
| Holm family | all ordered pairs | same |
| Win fractions, fixed target | mean of successful runs | ERT |
| Win fractions, several dimensions | partly dropped | every pair, equal weight |
| Seed for bootstrap | none | explicit seed (only if the bootstrap option is kept) |

## Proposed decisions

- **CM-1.** One-sided two-sample KS with R's asymptotic p-value
  exp(−2 nm/(n+m) D⁺²), hand-coded; an `exact=True` option gives the exact
  distribution.
- **CM-2.** Fixed target: failed runs count as +∞. IOHanalyzer's
  drop-failures behaviour is available only as an explicit option for golden
  tests.
- **CM-3.** Fixed budget: the orientation follows the maximisation flag.
- **CM-4.** Holm correction over all ordered pairs with a p-value, per (pair,
  target); α default 0.01; heat map S − Sᵀ and dominance graph as above.
- **CM-5.** Win fractions compare ERT (fixed target) or mean value (fixed
  budget), ties 0.5, averaged over targets then pairs with equal weight.
- **CM-6.** Simulated-restart bootstrapping is not ported unless the user asks
  for it (see Q-CM-2); if kept, it takes a seed.

## Open questions for the user

- Q-CM-1: Keep the asymptotic p-value as default (parity), or use the exact
  p-value for small samples as R's own `ks.test` would (n·m < 10000)?
- Q-CM-2: Is the bootstrap option (simulated restarts) still wanted?
- Q-CM-3: CM-5 changes IOHanalyzer's win fractions from "mean of successes" to
  ERT. Accept?

## Test tolerance

D⁺ and the decisions must match exactly. p-values must agree to a relative
error of 1e-12 (closed-form formula). Golden tests with IOHanalyzer use the
drop-failures option and name CM-2 where the agreed value differs.
