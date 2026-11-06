# Glicko-2 tournament ranking

Status: **proposed**, for sign-off at checkpoint H1.
Parity-matrix row: "Glicko-2 ranking" (iohinspector has Elo instead).
Implemented in: I15.
Sources: [IOHanalyzer], [PlayerRatings], [Glickman2022], [iohinspector]; see
[references](references.md).

The ranking turns many random head-to-head "games" between algorithms into
Glicko-2 ratings [Glickman1999, Glickman2022]. It needs two parts: the
tournament that produces the games, and the rating update.

## Definitions

### Tournament

Inputs: a selection of (function, dimension) pairs, one target (fixed
target) or one budget (fixed budget) per pair (default from TG-4), R rounds
(default 25 on the website, as IOHanalyzer's Shiny input), and a seed.

- Each round is one Glicko-2 rating period.
- In each round, for every pair (function, dimension) and every unordered pair
  of algorithms {a, c}, one game is played: one run of a and one run of c are
  drawn uniformly at random, with replacement, from their runs on that pair.
- Outcome for a: fixed target, the smaller runtime T(t) wins; a failed run
  loses to a successful one; two failed runs draw. Fixed budget, the better
  value V(b) wins (direction from the maximisation flag). Equal values draw.
  Win 1, draw 0.5, loss 0.
- Random numbers come from a numpy `Generator` seeded by the caller.

### Rating update

PlayerRatings' Glicko-2 [PlayerRatings], which follows [Glickman2022] with
these settings:

- initial rating 1500, deviation 350, volatility 0.06 (IOHanalyzer's `init`);
- system constant τ = 1.2 (PlayerRatings' default, which IOHanalyzer does not
  override);
- scale q = ln(10)/400, i.e. 173.7178 per Glicko-2 unit;
- deviation capped at rdmax = 350 and volatility at q · 350;
- no first-move advantage (γ = 0).

The new volatility σ' is the stationary point of Glickman's step-5 function
f(x) [Glickman2022]. PlayerRatings finds it by minimising the equivalent
objective (x − a)²/τ² + log(φ² + eˣ + v) + Δ²/(φ² + eˣ + v) with R's
`optimize` (Brent's method) on [a − 4τ, a + 4τ], a = ln σ², at R's default
tolerance (`.Machine$double.eps^0.25` ≈ 1.2e-4 in x). Setting its derivative
to zero gives exactly Glickman's f(x) = 0, so the two agree up to the
minimiser's tolerance.

Output per algorithm: rating, deviation, volatility, and games, wins, draws
and losses.

## Checked against Glickman's worked example

Player 1500/200/0.06 against 1400/30, 1550/100 and 1700/300, winning the first
game and losing two, τ = 0.5 [Glickman2022]:

| Source | rating | deviation | volatility |
|---|---|---|---|
| Glickman's note (rounded steps) | 1464.06 | 151.52 | 0.05999 |
| Glickman's algorithm, full precision, ε = 1e-6 (Python, scratch) | 1464.0506708 | 151.5165219 | 0.0599959843 |
| PlayerRatings 1.1-0 `glicko2(..., tau = 0.5)` | 1464.050671 | 151.5165213 | 0.0599958339 |

The volatility differs in the sixth significant digit because of
`optimize`'s tolerance; rating and deviation agree to better than 1e-8
relative.

## How IOHanalyzer implements it

`glicko2_ranking` (`R/stats.R` 595–717):

- default target per pair from `get_target_dt(dsl, which)` (614–615); here
  `which = 'by_FV'` means fixed target, which matches `get_target_dt`.
- `nr_rounds` rounds (default 100 in the function, 25 in Shiny:
  `inst/shiny-server/ui/stats_box.R` 94–96); in each round the dimensions and
  functions are shuffled (623–636).
- Per (dimension, function): each algorithm's sample is first resampled with
  replacement to a common length (`vals[i, ] = sample(y[1:n_valid], ...)`,
  659–669), and each game then draws one value from those resamples (676–677).
  Marginally this is a uniform draw of one run; games within a round share the
  resample, so they are not independent.
- Outcomes (678–696): NA (failed run, or missing value) loses to a value, two
  NA draw, equal values draw, otherwise `win_operator`: `<` for runtimes, and
  for values `>` when maximising, `<` otherwise.
- Each unordered pair plays once per (round, function, dimension) (671–675).
- `PlayerRatings::glicko2(games, init = c(1500, 350, 0.06))` (714) with
  `Weeks` = round, so each round is a rating period.
- No seed is set, so results change between runs.
- The algorithm order is taken from the first subset (`alg_names <- x_arr[, 3]`,
  658) and rows are then addressed by position, which assumes every subset
  lists the algorithms in the same order.

## How iohinspector 0.0.8 implements it

`get_tournament_ratings` (`metrics/ranking.py` 9–86) uses **Elo**
(`skelo.EloEstimator`), not Glicko-2. Every ordered pair plays, so each pair
plays twice per (round, function); values are drawn from all rows of the given
column per (algorithm, function); the generator and the shuffle are unseeded
(`np.random.default_rng()`, `sample(frac=1)`); the reported "deviation" is the
standard deviation of the ratings over the last 5 % of rounds.

## Differences

| Item | IOHanalyzer | iohinspector 0.0.8 | Proposed |
|---|---|---|---|
| Rating system | Glicko-2 (PlayerRatings) | Elo (skelo) | Glicko-2 as PlayerRatings |
| Games per round | each unordered pair once per pair of (f, d) | each ordered pair once | as IOHanalyzer |
| Draw of runs | shared resample per round | uniform per game | uniform per game |
| Failed runs (fixed target) | lose | – (values only) | lose; two failures draw |
| Seed | none | none | required |
| Rounds | 100 (function), 25 (Shiny) | 25 | parameter, default 25 |

## Proposed decisions

- **GL-1.** Tournament as defined above: one game per round, (function,
  dimension) pair and unordered algorithm pair, runs drawn independently and
  uniformly, failures lose, ties draw.
- **GL-2.** Rating update ported from PlayerRatings 1.1-0 with init
  (1500, 350, 0.06), τ = 1.2, rdmax 350, γ = 0, each round a rating period.
  The volatility step ports R's `optimize` (Brent's `fmin`) with its default
  tolerance, so that given the same game list the ratings match PlayerRatings.
- **GL-3.** The caller supplies a seed; the same seed gives the same ratings.
  Results are not expected to match IOHanalyzer's random games; golden tests
  feed the recorded R game list into the rating update instead.
- **GL-4.** The Elo-based `get_tournament_ratings` stays available under its
  own name, marked as not Glicko-2.

## Open questions for the user

- Q-GL-1: Keep τ = 1.2 (PlayerRatings' default, and IOHanalyzer's behaviour)
  or use a value in Glickman's suggested range 0.3–1.2 such as 0.5?

## Test tolerance

Given the same game list: rating and deviation agree with PlayerRatings to a
relative error of 1e-8, volatility to 1e-5 (the minimiser tolerance). If
GL-2's port of `optimize` is exact, the tolerance tightens to 1e-12. Game
outcomes for a fixed seed must be identical between runs.
