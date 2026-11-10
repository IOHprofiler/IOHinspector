# WP I07: Reader: COCO single-objective
Branch: wp/I07-coco-reader (stacked on wp/I05-ioh-reader, IOHinspector#21)
Status: in progress

## Done
- step 1: `scripts/coco_archive.py` downloads a sample of the COCO data archive
  (smallest archive per suite and year folder, 24 archives, ~190 MB) into
  `.cache/coco-archive` (git-ignored), checking sha256.

- step 2: reader `io/coco.py` (+ `_coco_info.py`, `_coco_dat.py`), generic
  `io/archive.py` (folders, archives, nested archives) and `io/merge.py`
  (identical-row rule, same `logged_in`/`merge_conflict` columns as I06's
  `_legacy_merge.py`); `READERS["coco"]`; `tests/io/test_coco.py`.

- step 3: `docs/readers-coco.md` (inventory, mapping, merge, archives,
  differences); `tests/io/test_coco_cocopp.py` (runtimes + aRT vs cocopp on
  `coco` and the new `coco_old` fixture); `tests/io/test_coco_archive.py`
  (opt-in, `COCO_ARCHIVE_TESTS=1`: all 24 sampled archives lossless, 87 s);
  `tests/golden/cocopp/generate.py` takes a fixture name, `make golden-cocopp`
  also writes `cocopp/coco_old/` (the `coco` outputs are byte-identical).

## Next step
Step 4: back `DataManager`'s COCO path with the reader; defect tests.

## Decisions
- `data-archive.json` does not exist at the archive's URL; the archive's index is
  `coco_archive_definition.txt` (a Python list of `(path, sha256, kB)`), which
  cocopp reads too. The script uses that.
- Sample: the smallest archive of every year folder of `bbob`, `bbob-noisy`,
  `bbob-largescale` and `bbob-boxed` (incl. `bbob/incomplete`). A wider scan of all
  92 archives of bbob/2009, bbob/2010, bbob-noisy/2009, bbob-noisy/2010 (info
  files and data headers only, not stored) found no further variants beyond the
  ones in the sample, plus: bare instance entries (unfinalized runs), one `.info`
  line with garbage after a complete data line (NELDER 2009 f13), blank lines,
  CRLF line endings, backup files `*.info~`.
- `raw_y` is column 3, "best noise-free fitness - Fopt" (Δf), as cocopp and
  IOHanalyzer use; `optimum` is the Fopt from each run's header line, so raw f is
  `raw_y + optimum`. Other columns are kept as extra trace columns.

- New golden fixture `tests/golden/fixtures/coco_old` (DIRECT f6, 2-D and 3-D,
  bbob 2009: old header, Windows paths, CRLF, unsuccessful runs), cocopp only;
  not in `load.FIXTURES` (IOHanalyzer's list). `versions.json` records Python
  3.11.16 (this machine; `coco` was made with 3.11.17).
- Finding for I10/I12: cocopp's `detEvals` looks 17 of the 51 standard targets
  up one grid level stricter (its level `10**(i/5)` is one ulp above the
  target). The cocopp goldens carry that; the comparison test reproduces it.

## Open questions / Blocked
- none
