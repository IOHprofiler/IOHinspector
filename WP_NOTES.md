# WP I06: Reader: legacy IOHprofiler .info/.dat
Branch: wp/I06-legacy-reader
Status: in progress

## Done
- step 1: `.info` parser (`io/_legacy_info.py`, `io/_legacy_meta.py`) with tests

## Next step
Step 2: data files and merge (`io/_legacy_dat.py`, `io/_legacy_merge.py`, `io/legacy.py`), tests in `tests/io/test_legacy.py`.

## Decisions
- Branched from main, not stacked on I05 (#21). `io/_compat.py` holds stand-ins for
  I05's `io/base.py` (`ReaderError`, `ReadResult`, `Reader`) with the same signatures.
  After I05 merges: delete `_compat.py`, import from `.base`, keep one line per reader
  in `io/__init__.py`'s `READERS`.

## Open questions / Blocked
- none
