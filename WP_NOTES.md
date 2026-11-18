# WP I15: Statistical comparisons and Glicko-2
Branch: wp/I15-stat-comparisons
Status: in progress

Definitions: `docs/definitions/comparisons.md` (CM-1 to CM-6), `docs/definitions/glicko2.md` (GL-1 to GL-4).

## Done
- step 1: `metrics/stats.py`: `ks_test` (one-sided two-sample KS, exact by R's psmirnov port or asymptotic), `holm`, `get_ks_tests` (pairwise table per view, Holm per view, `better` and heat-map `decision`); unit tests against R 4.3.3 values and golden tests against `KS_by_FV`, `KS_by_RT` on ioh040_min/max/cons (rtol 1e-12).

## Next step
Step 2: win fractions (`get_win_fractions`, CM-5 measure registry) and golden tests against `Heatmaps_*`.

Then step 3: Glicko-2 / Elo (`ranking.py`: `glicko2`, `elo`, `get_tournament_ranking`; deprecate `get_tournament_ratings`), golden tests feeding the R game lists `glicko2_games_by_*`.
Step 4: docs (definitions "implemented" notes, CHANGELOG), final review.

## Decisions
- `get_ks_tests` takes a sample table (the output of `get_runtime_sample` / `get_value_sample`), like `get_histogram`; `view` columns form one Holm family. The second group's `by` columns get the suffix `_vs`.
- `higher_is_better` (as in `get_rank_table`) orients the values; runtimes are always smaller-is-better.
- `drop_failures=True` is IOHanalyzer's behaviour (CM-2 option for golden tests).
- Exact p-values: equal `inf` values (failed runs) are ties. R 4.3.3's `psmirnov` treats them as distinct, because `Inf - Inf` is NaN in its tie pattern (R gives 0.8333 for the comparisons.md example; 0.7183 with ties, as R gives for 1e9 in place of Inf). Recorded in comparisons.md; question for the D6 reviewer.

## Open questions / Blocked
- (for dvermetten) Exact KS with failed runs: ties (implemented) or R's distinct-Inf behaviour?
