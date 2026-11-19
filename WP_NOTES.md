# WP I16b: Shapley closed form for the default value (I16 step 7)
Branch: wp/I16b-shapley-closed-form
Status: in progress

## Done
- step 7: closed form for the default value, tests, docs, bench (this commit)

## Next step
Run the full suite on polars 2.1.0 and polars<2, the portfolio tests on 1.33.1,
restore polars 2.1.0, then remove WP_NOTES.md in a final commit.

## Decisions
- The closed form lives on `envelope_value`'s object (`_Envelope.shapley`);
  `shapley_values` uses it when the value is that object and the marginal is
  the default. A wrapper around it counts as a custom value (tests use this
  to force enumeration).
- The test-only `_max_game` is gone; the closed form is tested against enumeration.
- Venv: `.venv` in the worktree (Python 3.14, polars 2.1.0 + polars-runtime-32 2.1.0).

## Open questions / Blocked
- none
