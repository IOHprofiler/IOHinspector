# WP I06: Reader: legacy IOHprofiler .info/.dat
Branch: wp/I06-legacy-reader
Status: in progress

## Done
- step 1: `.info` parser (`io/_legacy_info.py`, `io/_legacy_meta.py`) with tests
- step 2: data files (`io/_legacy_dat.py`), merge of `.dat/.tdat/.cdat/.idat` under the
  identical-row rule (`io/_legacy_merge.py`), reader `read_legacy` / `detect_legacy`
  (`io/legacy.py`), `READERS` line `ioh-legacy`; tests `tests/io/test_legacy.py` with a
  writer that mimics the IOHexperimenter 0.2 Default logger (`tests/io/legacy_writer.py`)

## Next step
Step 3: list the old variants from `readFiles.R` in `docs/readers/legacy.md`, one test per variant.

## Decisions
- Branched from main, not stacked on I05 (#21). `io/_compat.py` holds stand-ins for
  I05's `io/base.py` (`ReaderError`, `ReadResult`, `Reader`) with the same signatures.
  After I05 merges: delete `_compat.py`, import from `.base`, keep one line per reader
  in `io/__init__.py`'s `READERS`.

- `raw_y` is the second column (`current f(x)`); `best-so-far f(x)` and the transformed
  columns stay as extra trace columns. IOHanalyzer analyses the logged best-so-far column;
  for `.dat`/`.cdat` data the two agree (the reader warns `best_so_far_differs` when not).
- Merge: rows of different files with equal content at the same evaluation are stored
  once (`logged_in` = kinds, e.g. `dat+cdat`); differing rows are kept, `merge_conflict`.
  Both columns exist only for run sets with more than one data file.
- `evals_used` is the `.info` evaluation count as written (IOHanalyzer takes the max with
  the last logged evaluation at analysis time).
- Non-integer `funcId` (extdata `bla`, `blubb`): ids assigned from max integer id + 1 in
  byte order of names; the name goes to `function_name`.

## Open questions / Blocked
- none
