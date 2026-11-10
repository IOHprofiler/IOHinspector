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

- For up to 15 algorithms (2^15 subsets) φ is computed exactly by enumerating
  subsets.
- Above that it is estimated by stratified sampling: for each position
  j = 1..|N|, g random permutations with a placed at position j; φ_a is the
  mean of the marginal contributions over positions. The caller gives g and a
  seed.

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
| Estimator | stratified permutations, truncated above 10 algorithms, no seed | exact up to 15, else stratified over all positions, seeded |

## Decisions

- **SH-1.** v(S) is the summed AUC of the virtual-best ECDF over pairs and
  targets, with ECDFs over all runs.
- **SH-2.** Exact Shapley values up to 15 algorithms (2^15 subsets); seeded
  stratified sampling over **all** positions above that. Decided by the D6 review (dvermetten, 2026-11-06).
- **SH-3.** Values are normalised by v(N) and sum to 1.
- **SH-4.** The Shapley functionality is generic: it takes the portfolio value
  function and the marginal contribution as arguments, with the ECDF-envelope
  value (SH-1) and v(S ∪ {a}) − v(S) as defaults, so other definitions of
  marginal contribution can be added later without rewriting it. Decided by the D6 review (dvermetten, 2026-11-06).

## Questions answered by the D6 review

- Q-SH-1 (limit for exact Shapley values): 15 is a good limit. See SH-2.
- Added requirement: generic marginal contributions. See SH-4.

Remaining points for H1: none.

## Test tolerance

Exact Shapley values: relative error 1e-12, and the sum must be 1 within
1e-12. Sampled values: compared with the exact values on a 6-algorithm case,
within 3 standard errors.
