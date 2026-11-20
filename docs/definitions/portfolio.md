# Shapley values of algorithm portfolios

Status: **proposed**, for sign-off at checkpoint H1; the open questions were
answered by the D6 review (dvermetten, 2026-11-06).
Parity-matrix row: "Shapley portfolio values".
Implemented in: I16.
Sources: [IOHanalyzer], [Shapley1953], [Frechette2016]; see
[references](references.md).

## Definitions

### Portfolio value

For a set S of algorithms, the portfolio is the **virtual best** of S: on
every (function, dimension) pair p, target t_k of p and budget b, it reaches
what the best member reaches. With ECDF_{a,p,k}(b) the single-target ECDF of
algorithm a ([ecdf](ecdf.md), EC-1: fraction of all runs, failures included),

    v(S) = Σ_p Σ_k AUC_b[ max_{a ∈ S} ECDF_{a,p,k}(b) ],   v(∅) = 0,

with the AUC on the chosen budget axis (EC-4). v(S)/v(N) ∈ [0, 1] for the full
set N.

### Shapley value

The Shapley value [Shapley1953] of algorithm a is its marginal contribution
averaged over all orders in which the portfolio can be built [Frechette2016]:

    φ_a = Σ_{S ⊆ N∖{a}} |S|! (|N| − |S| − 1)! / |N|! · ( v(S ∪ {a}) − v(S) ).

Values are reported normalised, φ_a / v(N), so they sum to 1 (efficiency).

The computation is generic in the definition of the portfolio value and of the
marginal contribution (SH-4). The Shapley routine takes a *value function*
v: subset → number, and optionally a *marginal function* m(S, a), which
defaults to v(S ∪ {a}) − v(S). The ECDF-envelope value above is the default
value function; other definitions, for example different anytime
contributions to portfolio performance, plug in without changing the
enumeration, the sampling, the normalisation or the caching of v(S).

### Computation

- The default value (SH-1) with the default marginal contribution is computed
  exactly, in closed form, for any number of algorithms (Q-SH-2). v is a sum
  of *maximum games*, one per (pair, target, budget step) c with weight w_c:
  v(S) = Σ_c w_c max_{a ∈ S} e_{a,c}. The Shapley value is linear in the game,
  and in a maximum game a gains the increase of the maximum whenever it joins
  a coalition of lower ECDFs. With the ECDFs of a step sorted ascending,
  0 = e_(0) ≤ e_(1) ≤ … ≤ e_(n), the algorithm in place i gets

      φ_(i),c = Σ_{j ≤ i} (e_(j) − e_(j−1)) / (n − j + 1),

  because the slice (e_(j−1), e_(j)] goes to whichever of the n − j + 1
  algorithms in places j..n comes first in the order, each equally likely.
  Tied algorithms get equal values. The cost is a sort per step, n log n.
- Any other value or marginal contribution (SH-4): for up to 15 algorithms
  (2^15 subsets) φ is computed exactly by enumerating subsets.
- Above that it is estimated by stratified sampling: for each position
  j = 1..|N|, g random permutations with a placed at position j; φ_a is the
  mean of the marginal contributions over positions. The caller gives g and a
  seed.
- The caller can choose the computation (`method`): the closed form,
  enumeration or sampling. Enumeration and sampling work for every value,
  the default one included, so the closed form can be checked against them.
  `auto`, the default, picks as above.

## How IOHanalyzer implements it

- `get_shapley_values` (`R/stats.R` 1184–1226) with `get_marg_contrib_ecdf`
  (1151–1164):
  - the ECDF data come from `generate_data.ECDF_raw` (`R/DataSetList.R`
    1736–1792): per (algorithm, pair, target), R's `ecdf()` of the runtime
    sample on a 50-point `seq_RT` grid (1742). `ecdf()` drops NA, so the ECDF
    is normalised by the number of **successful** runs: an algorithm with one
    success out of 15 reaches 1.0 at that runtime. Its comparison is `≤`;
  - v(S) is the sum over (pair, target, grid point) of the maximum `hit` over
    S (1154–1155, 1204–1205);
  - `nr_players * group_size` random permutations are drawn without a seed
    (1199–1202). For each position j = 1..`max_perm_size`, group j of
    `group_size` permutations is used with a swapped to position j
    (1208–1217); the estimate is the mean over j, divided by v(N) (1218–1220).
    With `max_perm_size` smaller than the number of algorithms, positions
    beyond it, and so the larger coalitions, are never sampled.
- Shiny: `group_size` default 5 (`inst/shiny-server/ui/rt_portfolio_box.R:13`)
  and `max_perm_size` set to min(10, number of algorithms) on upload
  (`inst/shiny-server/server/upload.R` 487–488). With more than 10
  algorithms the value is therefore truncated.
- Targets: `get_ECDF_targets` with the page's type and number
  (`inst/shiny-server/server/Shapley_computations.R:56`).

## How iohinspector 0.0.8 implements it

No Shapley or portfolio function.

## Differences

| Item | IOHanalyzer | Proposed |
|---|---|---|
| ECDF normalisation | successful runs only | all runs (EC-1) |
| Budget axis | sum over 50 grid points | exact AUC (EC-4) |
| Estimator | stratified permutations, truncated above 10 algorithms, no seed | exact for any number (closed form); other definitions exact up to 15, else stratified over all positions, seeded |

## Decisions

- **SH-1.** v(S) is the summed AUC of the virtual-best ECDF over pairs and
  targets, with ECDFs over all runs.
- **SH-2.** The default value (SH-1) is computed exactly for any number of
  algorithms, in closed form. Other values or marginal contributions (SH-4):
  exact Shapley values up to 15 algorithms (2^15 subsets); seeded stratified
  sampling over **all** positions above that. Decided by the D6 review
  (dvermetten, 2026-11-06), and on #44 for the closed form (Q-SH-2,
  2026-11-19).
- **SH-3.** Values are normalised by v(N) and sum to 1.
- **SH-4.** The Shapley functionality is generic: it takes the portfolio value
  function and the marginal contribution as arguments, with the ECDF-envelope
  value (SH-1) and v(S ∪ {a}) − v(S) as defaults, so other definitions of
  marginal contribution can be added later without rewriting it. Decided by the D6 review (dvermetten, 2026-11-06).

## Questions answered

- Q-SH-1 (D6 review, limit for exact Shapley values): 15 is a good limit. See
  SH-2.
- Added requirement (D6 review): generic marginal contributions. See SH-4.
- Q-SH-2 (review of #44, answered 2026-11-19; enumeration was 13 times slower
  than sampling at 15 algorithms): use the closed form for the default value,
  exact for any number of algorithms. Enumeration up to `exact_limit` and
  sampling above it remain for other values and marginals. See SH-2 and
  "Speed" below.

Remaining points for H1: none.

## Implementation (I16)

`iohinspector.metrics`, in `metrics/portfolio.py`:

- `shapley_values(n_players, value, *, marginal=None, method="auto",
  exact_limit=15, samples=5, seed=None)`: the generic routine (SH-4). `value`
  takes a boolean matrix, one row per coalition and one column per player,
  and returns v of each row; `marginal(v, S, a)` gets the cached value
  function and the coalitions S without a, and defaults to v(S ∪ {a}) − v(S).
  `method` chooses the computation:
  - `"closed"`: the closed form (Q-SH-2). Only for `envelope_value` with the
    default marginal; anything else raises a `ValueError`.
  - `"exact"`: enumeration (SH-2), v called once on all 2^n coalitions. More
    than `exact_limit` players raises a `ValueError`; raise `exact_limit` to
    enumerate more.
  - `"sampled"`: for each player and position j, `samples` random coalitions
    of j − 1 other players (the predecessors of a in a random permutation that
    puts a at position j), seeded by `seed`; v is called once per distinct
    coalition.
  - `"auto"` (default): `"closed"` where it applies, else `"exact"` up to
    `exact_limit` players and `"sampled"` above.

  Returns the values divided by v(N) (SH-3) and a standard error per value (0
  unless sampled); NaN when v(N) = 0.
- `get_portfolio_ecdf(data, targets, by, *, b_lo, b_hi, scale_eval_log, ...)`:
  the ECDF data of the game. The players are the groups of `by` (D17). With the
  targets, the pair selection (`pairs`, EC-2), the hit rule (EC-1) and the
  budget range and axis (EC-4) of `get_ecdf_auc`, each player's ECDF on a
  (pair, target) is a step function on the steps all players share. Returns
  the players, the matrix `ecdf` (players × steps) and the step widths `width`
  on the budget axis, divided by its length. Identical steps are merged and
  their widths summed, so a step no longer knows its pair, target or budget.
- `envelope_value(ecdf, width)`: SH-1 on these data,
  v(S) = Σ width · max_{a ∈ S} ecdf[a]. It also carries the closed form, which
  `shapley_values` uses for it. Any other function, even one that only calls
  it, is a custom value.
- `get_shapley_values(data, targets, by, *, b_lo, b_hi, scale_eval_log, ...)`:
  `shapley_values` of `envelope_value` on `get_portfolio_ecdf`. Returns `by`,
  `shapley`, `se`.

### Another value or marginal contribution (SH-4)

`shapley_values` does not depend on the envelope. The enumeration, the
sampling, the caching of v and the normalisation only call `value` and
`marginal`, so another definition is a new function, not a change to the
routine:

```python
import numpy as np
from iohinspector.metrics import envelope_value, get_portfolio_ecdf, shapley_values

players, ecdf, width = get_portfolio_ecdf(data, targets)

# another portfolio value: the members run in parallel, each once and
# independently; per step, the probability that at least one reaches the target
def parallel(coalitions):  # boolean, coalitions × players
    return (1 - np.prod(1 - coalitions[:, :, None] * ecdf, axis=1)) @ width

phi, se = shapley_values(players.height, parallel)

# another marginal contribution, not a difference of values: the share of the
# budget axis on which a is strictly ahead of every member of S
def ahead(value, coalitions, a):
    best = (coalitions[:, :, None] * ecdf).max(axis=1)
    return (ecdf[a] > best) @ width

phi, se = shapley_values(players.height, envelope_value(ecdf, width), marginal=ahead)
```

Rules for a definition:

- `value(S)` takes a boolean matrix, one row per coalition and one column per
  player, and returns one number per row. It is called on batches: once on all
  2^n coalitions when exact, once per new coalition when sampling.
- `marginal(v, S, a)` gets the cached value function, the coalitions S (rows
  without a) and the player a, and returns one number per row. It may use v,
  or data of its own such as `ecdf`, or both.
- A custom value or marginal is enumerated up to `exact_limit` players and
  sampled above (`method="auto"`), or as `method="exact"` or `"sampled"` says;
  `method="closed"` is for the default value only.
- The values are always divided by v(N). They sum to 1 only with the default
  marginal; with a marginal that does not use v, v only sets that scale.
- The merged steps of `get_portfolio_ecdf` suit definitions that add up over
  steps, as SH-1 does. A definition that weights a step by its pair, target or
  budget needs the steps before merging, which is not exposed yet.

The tests (`tests/test_metrics/test_portfolio.py`) run both examples above,
exact and sampled, against a direct computation over all orders.

### Speed (review of #44, Q-SH-2)

`scripts/bench_shapley.py` builds the game of `get_shapley_values` on a
synthetic full-bbob set (24 functions, 6 dimensions, 15 runs per algorithm and
pair, the default targets, a log budget axis) and times the default value
with each `method`: `"closed"`, which `get_shapley_values` uses by default,
`"exact"` (enumeration) and `"sampled"` (5 coalitions per player and position,
the default). On a 10-core laptop:

| algorithms | steps | ECDF data (s) | closed form (s) | enumeration (s) | sampled (s) | enumeration − closed | sampled − closed | max se |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 78 374 | 1.18 | 0.007 | 0.01 | 0.01 | 1·10⁻¹⁶ | 4·10⁻⁴ | 3·10⁻⁴ |
| 10 | 198 404 | 2.36 | 0.028 | 1.93 | 0.91 | 6·10⁻¹⁶ | 4·10⁻⁴ | 2·10⁻⁴ |
| 12 | 235 350 | 2.71 | 0.041 | 13.6 | 2.80 | 6·10⁻¹⁶ | 4·10⁻⁴ | 2·10⁻⁴ |
| 14 | 271 367 | 3.06 | 0.057 | 85.7 | 7.00 | 4·10⁻¹⁶ | 2·10⁻⁴ | 2·10⁻⁴ |
| 15 | 288 802 | 3.39 | 0.072 | 224 | 11.4 | 4·10⁻¹⁶ | 3·10⁻⁴ | 2·10⁻⁴ |

The ECDF data use simulated restarts (EC-7, the `get_ecdf_auc` default), which
about doubles the steps. The differences are the largest absolute difference
of a normalised value. Enumeration grows as 2^n · n per step, sampling as n^3
and the closed form as n log n, so `get_shapley_values` now spends its time on
the ECDF data: 3.5 s at 15 algorithms, against 228 s with enumeration (69 s on
#44, before simulated restarts). Custom values and marginals still pay the
enumeration column up to `exact_limit` with `method="auto"`.

Golden tests: IOHanalyzer's value (R's `ecdf` of the successful runtimes,
summed over its 50-point grid) is a custom value function for
`shapley_values`. With it the tests reproduce `ECDF_raw` and
`marg_contrib_ecdf` exactly, and IOHanalyzer's unseeded Shapley estimates lie in
the range its estimator can produce (equal to the exact values for two
algorithms). The agreed value (SH-1) gives different numbers, which the test
names.

## Test tolerance

Exact Shapley values: relative error 1e-12, and the sum must be 1 within
1e-12. The closed form equals enumeration within a relative 1e-12 for 1 to 15
algorithms, with ties and steps no algorithm reaches. Sampled values: compared
with the exact values on a 6-algorithm case, within 3 standard errors (4 on a
40-algorithm case, where 3 is too tight for 40 values at once).
