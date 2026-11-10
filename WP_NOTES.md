# WP I07: Reader: COCO single-objective
Branch: wp/I07-coco-reader (stacked on wp/I05-ioh-reader, IOHinspector#21)
Status: in progress

## Done
- step 1: `scripts/coco_archive.py` downloads a sample of the COCO data archive
  (smallest archive per suite and year folder, 24 archives, ~190 MB) into
  `.cache/coco-archive` (git-ignored), checking sha256.

## Next step
Step 2: write `src/iohinspector/io/coco.py` (+ `archive.py`, `merge.py`).

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

## Open questions / Blocked
- none
