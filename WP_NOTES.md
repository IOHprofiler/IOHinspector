# WP I15: Statistical comparisons and Glicko-2
Branch: wp/I15-stat-comparisons
Status: in progress

Definitions: `docs/definitions/comparisons.md` (CM-1 to CM-6), `docs/definitions/glicko2.md` (GL-1 to GL-4).

## Done
- step 1: `metrics/stats.py`: `ks_test` (one-sided two-sample KS, exact by R's psmirnov port or asymptotic), `holm`, `get_ks_tests` (pairwise table per view, Holm per view, `better` and heat-map `decision`); unit tests against R 4.3.3 values and golden tests against `KS_by_FV`, `KS_by_RT` on ioh040_min/max/cons (rtol 1e-12).

- step 2: `get_win_fractions` with the `MEASURES` registry (CM-5: fixed target ert/mean/par_k/median, fixed budget mean/median/best); golden tests against all `Heatmaps_*` files (measure `mean`, IOHanalyzer's first-n-functions defect reproduced by restricting the pairs).

- step 3: `ranking.py`: `fmin` (R's `optimize`, Brent_fmin port), `glicko2` (PlayerRatings 1.1-0 port), `elo` (K = 20), `get_tournament_ranking` (GL-1 tournament, seeded, `method` glicko2/elo, `tau` default 0.5); `get_tournament_ratings` deprecated (the plot still uses it, warning silenced, until I20). Golden: the six recorded R game lists give PlayerRatings' ratings bit for bit on five, 4e-12 relative on ioh040_min by_RT (macOS libm); test rtol 1e-11. Glickman's example matches PlayerRatings' values in glicko2.md.

## Next step
Step 4: docs (definitions "implemented" notes, CHANGELOG), final review.

## Decisions
- `get_ks_tests` takes a sample table (the output of `get_runtime_sample` / `get_value_sample`), like `get_histogram`; `view` columns form one Holm family. The second group's `by` columns get the suffix `_vs`.
- `higher_is_better` (as in `get_rank_table`) orients the values; runtimes are always smaller-is-better.
- `drop_failures=True` is IOHanalyzer's behaviour (CM-2 option for golden tests).
- Win fractions and the tournament take `targets` per pair (`function_id`, `dimension`, `target`, as `get_target_dt`), default `get_target_dt(perspective)`; several targets per pair are allowed (each is one more match).
- Tournament: groups are players in sorted order; pairs, targets and groups are visited in sorted order so a seed is reproducible. A missing value (fixed budget before the first evaluation) counts as a failure.
- Elo deviation: std (ddof 1) of a player's ratings after its games in the last 5 % of rounds, at least the last round (0.0.8 had none when rounds < 20).
- Exact p-values: equal `inf` values (failed runs) are ties. R 4.3.3's `psmirnov` treats them as distinct, because `Inf - Inf` is NaN in its tie pattern (R gives 0.8333 for the comparisons.md example; 0.7183 with ties, as R gives for 1e9 in place of Inf). Recorded in comparisons.md; question for the D6 reviewer.

## Open questions / Blocked
- (for dvermetten) Exact KS with failed runs: ties (implemented) or R's distinct-Inf behaviour?
