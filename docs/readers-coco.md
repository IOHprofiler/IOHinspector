# Reader: COCO single-objective data

`iohinspector.io.read_coco(source, dataset_id)` reads the data that COCO's single-objective loggers write, as it appears in the [COCO data archive](https://numbbo.github.io/data-archive): the suites `bbob`, `bbob-noisy`, `bbob-largescale` and `bbob-boxed`, from the 2009 logger to today's. Its name in `READERS` is `coco`. `source` is a folder (searched recursively) or an `.info` file; `iohinspector convert` also opens archives (below). It returns the tables of [format v1](format.md), which `ReadResult.write` stores.

COCO's bi-objective, constrained and mixed-integer suites are out of scope (list C of the plan).

## Files

An `.info` file holds one block of three lines per function, dimension and experiment session:

```text
suite = 'bbob', funcId = 1, DIM = 2, Precision = 1.000e-08, algId = 'BFGS', coco_version = '2.3.6', logger = 'bbob', data_format = 'bbob-new2'
% BFGS: a description of the algorithm
data_f1/bbobexp_f1_DIM2.dat, 1:13|2.7e-13, 2:25|2.3e-13, 3:17|9.2e-12
```

1. A header of `key = value` pairs; strings are in single quotes. Loggers before COCO 2 write only `funcId`, `DIM`, `Precision` and `algId`.
2. A comment line starting with `%`.
3. The data file, relative to the `.info` file (and to `folder` when the header has one), then one entry per run: `instance:evaluations|value`, where `value` is the run's final best Δf written with `%.1e`. A bare `instance` marks a run the logger did not finalise.

A data file holds the runs of one function and dimension one after the other. Each run starts with a header line, then one line per logged evaluation: five columns, followed by the position `x1 … xD` when the logger wrote it.

```text
% f evaluations | g evaluations | best noise-free fitness - Fopt (7.948000000000e+01) + sum g_i+ | measured fitness | best measured fitness or single-digit g-values | x1 | x2...
1 0 +5.826987706e-01 +8.006269877e+01 +8.006269877e+01 -3.2755e-01 -6.6092e-01
```

Up to three data files share a stem, each with its own trigger:

| File | Written when | Manifest `triggers` |
|---|---|---|
| `.dat` | Δf reaches the next target level | `fixed_target` |
| `.tdat` | at fixed evaluation counts | `fixed_budget` |
| `.rdat` | at restarts (the old logger adds `% restart: …` comment lines) | `other`, `params: {"event": "restart"}` |

## Inventory of the archive

`scripts/coco_archive.py` downloads the smallest archive of every year folder of the four suites (24 archives, about 190 MB) from the archive's index, `coco_archive_definition.txt`, and checks their sha256. The plan named `data-archive.json`; the archive publishes no such file, and its index (also what cocopp reads) is that definition file. A second scan read the `.info` files and data headers of all 92 archives of `bbob` 2009 and 2010 and `bbob-noisy` 2009 and 2010. The variants found:

| Logger, years | `.info` header | Data header | Data files | Seen in |
|---|---|---|---|---|
| `bbob` 2009–2018 (C, Python, MATLAB) | `funcId`, `DIM`, `Precision`, `algId` | old: `% function evaluation \| noise-free fitness - Fopt (F) \| best noise-free fitness - Fopt \| measured fitness \| best measured fitness \| x1 \| x2...` | `.dat`, `.tdat`; `.rdat` from 2013, with `% restart` lines from 2015 | bbob 2009–2018, bbob-noisy 2009–2016, bbob/incomplete 2014 |
| COCO 2 (`data_format = 'bbob-new2'`) 2019– | adds `suite`, `coco_version` (may be empty), `logger`, `data_format` | new: `% f evaluations \| g evaluations \| best noise-free fitness - Fopt (F) + sum g_i+ \| measured fitness \| best measured fitness or single-digit g-values \| x1 \| x2...` | `.dat`, `.tdat`, `.rdat` (often headers only); `.mdat` (bbob-largescale 2024), `.cdat` (bbob-boxed 2023) | bbob 2019–2023, bbob-largescale 2019–2024, bbob-boxed 2023–2024, bbob/incomplete 2019 |

Quirks in the sample and the scan, each covered by a test:

- Old `.info` values are Δf to the final target (Fopt + Precision), so solved runs show small negative values; new ones are Δf to Fopt.
- MATLAB data (bbob 2009, DIRECT): Windows paths `data_f6\bbobexp_f6_DIM2.dat`, CRLF line endings, three-digit exponents (`-8.6e-010`). The fixture `tests/golden/fixtures/coco_old` is taken from it.
- Bare instance entries (unfinalised runs): bbob 2009 CMA-ESPLUSSEL and NELDER, bbob-noisy 2010 CMAEGS (45 runs).
- A data line followed by a line of stray text (`:2001991|1.4e-03, 5`, bbob 2009 NELDER f13), blank lines, backup files (`*.info~`).
- Several sessions of one function: separate blocks and files `bbobexp-01_f4_DIM2.dat`; batches in folders `…-001`, `…-002` (bbob-largescale 2019).
- No positions above 20-D (rows of five fields; COCO 2) and position-less rows next to rows with positions in one file.
- `.dat` and `.tdat` rows of the same evaluation with different values: other positions at the final evaluation (DIRECT), several evaluations all counted as 1 (bbob 2017, EvoSpace-PSO-GA).
- Files the reader does not use: `.mdat`, `.cdat`, cocopp output in a nested `TEMP.zip` (bbob 2022, EXPLO2), algorithm sources (`.m`), `.json`, `.pydict`.
- Every `.info` data line names one data file. cocopp also accepts several; this reader refuses them with file:line, as no archive in the scan has one.

## Mapping to format v1

| Source | Stored as |
|---|---|
| `suite` | `runs.suite`, as written. Without it: `bbob-noisy` for `funcId` above 100, else `bbob` (cocopp's rule), with a `suite_inferred` warning |
| `funcId` (or `function`) | `runs.function_id` |
| `DIM` (or `dim`) | `runs.dimension` |
| — | `runs.maximization` is false |
| `algId` (or `algorithm`) | descriptor `algorithm_name` (origin `algorithm`); `""` with a `missing_algorithm` warning when absent |
| comment line | descriptor `algorithm_info`, without the `%` |
| other header keys (`Precision`, `coco_version`, `logger`, `data_format`, …) | descriptors with origin `experiment`, under their own names; quoted values are strings, others `int` or `float` |
| run entry | `runs.instance`; `runs.evals_used` (null for an unfinalised run); descriptor `final_delta_f` (origin `run`) |
| Fopt in the run's header line | `runs.optimum` |
| data column 1 | `trace.evaluations` |
| data column 2 | `g_evaluations` (new header, int64) or `current_delta_f` (old header: the evaluation's noise-free Δf) |
| data column 3 | `trace.raw_y`: the best noise-free Δf so far (plus the constraint sum, zero on these suites) |
| data columns 4 and 5 | `measured_f`, `best_measured_f` (the measured value includes the noise on `bbob-noisy`) |
| `x1 … xD` | `positions.parquet`, `x0 … x(D-1)`, for the rows that have them |

`raw_y` holds Δf = f − Fopt, the value COCO's targets apply to (`docs/definitions/alignment.md`, AL-7) and the column cocopp and IOHanalyzer read. Raw f is `raw_y + optimum`; the logger wrote Δf with ten significant digits, which the measured value (`%+10.9e` of f itself) cannot give for small Δf. `best_y` is null: the `.info` value is rounded to two digits and, for old loggers, relative to the final target, so it is kept as the descriptor `final_delta_f` instead.

Header keys that clash with reserved names are renamed (`instance_experiment`) and listed in `manifest.renamed_columns`.

The runs of a data file start at its header lines (`% f evaluations …` or `% function evaluation …`); other `%` lines are comments and are not stored. The runs are matched in order with the entries of the blocks that name the file. `source_file` is the `.dat` file, or the first that exists of `.dat`, `.tdat`, `.rdat`; `source_rows_sha256` hashes that file's data lines of the run.

## Merging the data files

All data files of a stem are read and must hold the same number of runs, otherwise the reader fails. Rows are merged under the identical-row rule (`iohinspector.io.merge`): a row that two files log for the same evaluation of a run is stored once when every column, position included, is identical (NaN equals NaN). Otherwise both rows are kept. Nothing is merged within one file. When a stem has more than one file, two trace columns describe the merge:

- `logged_in`: the kinds that hold the row, joined by `+` in the order `dat`, `tdat`, `rdat`, for example `dat+tdat`;
- `merge_conflict`: true for every row of an evaluation that the files log with different values. A `merge_conflict` warning counts them per stem.

Rows are ordered by evaluation, then by kind, then by position in the file. If a file's evaluations decrease within a run, its rows are not merged: the files' rows follow each other per run, and an `unsorted_evaluations` warning is added. So every file can be rebuilt from the trace: the rows whose `logged_in` names its kind, in `row` order.

## Archives

`iohinspector convert` opens archives before the reader runs (`iohinspector.io.archive`, the same for every reader): a folder may hold archives, and an archive may hold archives. zip, tar, tgz, tar.gz, tar.bz2 and tar.xz are opened. A file inside an archive has the source path `<archive path>/<member path>`, for example `BFGS.tgz/BFGS/data_f1/bbobexp_f1_DIM2.dat`; `manifest.archives` lists every archive with its sha256 (and the archive it came from), and each source names its innermost archive. One input may hold many algorithms: every `.info` block carries its own `algId`. Members that are links or devices, or whose names leave the archive, are skipped (`skipped_archive_member`); an archive that cannot be opened is skipped (`skipped_archive`).

## Differences from cocopp and IOHanalyzer

- cocopp aligns the `.dat` rows on its grid of target levels 10^(i/5) and looks a target up at the closest level not larger than it. It computes the level as `10 ** (i / 5)`, which for 17 of the 51 standard targets is one ulp above the target `10 ** (2 - k / 5)`, so it uses the next, stricter level there. iohinspector hits the target itself (AL-2, AL-8). `tests/io/test_coco_cocopp.py` reproduces cocopp's lookup to compare, and checks that the agreed rule differs only at those targets, and only by being earlier.
- cocopp skips runs without data rows when it splits a data file, which shifts the instances of later runs; this reader splits on header lines.
- cocopp names datasets read from archives after the archive, not `algId`; this reader keeps `algId`; the source paths name the archive.
- IOHanalyzer reads only one data file (`.dat`, else `.tdat`, else `.cdat`; `R/DataSet.R` 93–101) and the first five columns; this reader keeps every row of every file and every column.
- iohinspector 0.0.8 set every instance to 0 and merged `.dat` and `.tdat` rows on the evaluation count alone, keeping the first and dropping the rest. `DataManager` now reads COCO data with this reader.

## Errors and warnings

Malformed input raises `ReaderError`, whose message starts with `file:line`: a header without `funcId` or `DIM`, a run entry that is not `instance:evaluations|value` or `instance`, a data line naming several data files, a data line before the first header line, a row of other than 5 or 5 + D fields, a token that is not a number (or not an integer in the evaluation columns), a header style that changes within a file or between the files of a stem, files of one stem with different run counts, and one stem named with two dimensions.

Warnings in the manifest (the last two from `convert`): `suite_inferred`, `missing_algorithm`, `unfinalized_run`, `run_count_mismatch`, `missing_data_file` (the runs are kept without trace rows, the source is listed with a null sha256), `unreferenced_data_file` (also every `.mdat` and `.cdat`), `skipped_file` (an `.info` file that is not COCO's), `info_line_skipped`, `incomplete_info_record`, `not_utf8` (an `.info` file read as Latin-1), `merge_conflict`, `unsorted_evaluations`, `skipped_archive` and `skipped_archive_member`.

## Tests

- `tests/io/test_coco.py`: the fixture `tests/test_coco_data` against the files read independently (every row of every data file, hashes, instances, Fopt), the merge, every variant above, malformed input, and nested archives (through `convert`).
- `tests/io/test_coco_cocopp.py`: runtimes for the 51 targets and aRT for the 51 and the table targets against cocopp 2.9.1 on `coco` and `coco_old` (`tests/golden/cocopp`).
- `tests/io/test_coco_archive.py`: every sampled archive, with every data file counted from the archive independently; run with `python scripts/coco_archive.py` and then `COCO_ARCHIVE_TESTS=1 pytest tests/io/test_coco_archive.py` (about 90 s).
