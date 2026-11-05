# Distributions: histograms, PMF and violin data

Status: **proposed**, for sign-off at checkpoint H1.
Parity-matrix row: "Histograms, violin data".
Implemented in: I13.
Sources: [IOHanalyzer], [R], [plotly.js], [iohinspector]; see
[references](references.md).

These views show the distribution of one sample: the runtimes
{T_i(t)} at a target, or the values {V_i(b)} at a budget, per algorithm
([alignment](alignment.md)). Only runs with a defined value enter; for
runtimes that means successful runs, and the view states how many runs failed.

## Definitions

### Histogram breaks

For a sample x of size n ≥ 2 (fewer values: no histogram for that algorithm):

1. **Number of classes** k, Freedman–Diaconis [Freedman1981] as in R's
   `nclass.FD` (R 4.3.3):
   - h = 2 · IQR(signif(x, 5)), IQR with quantile type 7;
   - if h = 0: h = (q(1−a) − q(a)) / (1 − 2a) for a = 1/8, 1/16, …, down to
     1/512, stopping at the first h > 0;
   - if still h = 0: h = 3.5 · sd(x), Scott's rule [Scott1979];
   - k = ceiling(range(x) / h · n^(1/3)) if h > 0, else 1.
2. **Breaks**: R's `pretty(range(x), n = k, min.n = 1)`, so the bin edges are
   "round" numbers and all bins have the same width.
3. **Bins** are right-closed, (a, b], with the lowest bin closed on both
   sides, and R's relative fuzz of 1e-7 on the edges (`hist.default`).
4. Output per bin: left and right edge, midpoint, count, and the bin width.

By default breaks are computed per algorithm. With "equal bins" they are
computed once from the pooled sample of all algorithms in the view and used
for every algorithm.

### PMF and violin data

IOHanalyzer's "probability mass function" view is the raw sample drawn as a
violin with a box and, optionally, the points. Because the new website draws
with ECharts, iohinspector computes the violin itself:

- kernel: Gaussian;
- bandwidth: Silverman's rule of thumb [Silverman1986],
  bw = 1.059 · min(sd, IQR / 1.349) · n^(−1/5), sd with denominator n − 1;
- support: the data range only ("hard" span), evaluated on a grid whose step
  is at most bw / 3;
- density scaled by the sample size (violins of larger samples are wider);
- box: median and quartiles (type 7), as in [fixed-budget](fixed-budget.md);
- a sample with zero range gives a zero-width violin at that value.

## How IOHanalyzer implements it

- `generate_data.hist` (`R/DataSetList.R` 1250–1317): `hist(data, breaks =
  nclass.FD, plot = F)` per DataSet (1295), or with the breaks of the pooled
  sample when `use.equal.bins` (1260–1273, 1290–1293); fewer than two non-NA
  values gives no histogram (1287–1288). `hist` drops non-finite values, so
  failed runs (NA) are excluded. The output width is `breaks[2] - breaks[1]`
  (1310). R's `hist.default` turns a class count into breaks with
  `pretty(range(x), n = breaks, min.n = 1)` and counts with right-closed bins
  and `fuzz = 1e-7` (checked in R 4.3.3).
- `generate_data.PMF` (`R/DataSetList.R` 1232–1236) returns the runtime or
  value sample. The Shiny page draws it through `plot_general_data` with the
  plotly `violin` trace (`R/plotDataSetList.R` 1754–1790): `scalemode =
  'count'`, `spanmode = 'hard'`, box visible, `jitter = 0`. The density is
  computed by plotly.js in the browser: Gaussian kernel, Silverman's rule
  1.059 · min(ssd, IQR/1.349) · n^(−0.2) (`src/traces/violin/calc.js` 95–98
  and 126–128, plotly.js v2.25.2), step bw/3 (line 41), box quartiles by
  plotly's default `linear` method.
- The static ggplot backend (`R/plot.R` 180–225) uses `geom_violin` with
  `bw = 'SJ'` (Sheather–Jones [Sheather1991]), so the two IOHanalyzer
  backends draw different violins for the same data.

## How iohinspector 0.0.8 implements it

There are no histogram, PMF or violin functions in 0.0.8.

## Differences

| Item | IOHanalyzer | iohinspector 0.0.8 | Proposed |
|---|---|---|---|
| Histogram breaks | `nclass.FD` + `pretty`, right-closed | none | same as R, ported |
| Violin density | plotly.js, Silverman; ggplot: SJ | none | Silverman, as plotly.js |
| Failed runs | dropped silently | – | dropped, count shown |

## Proposed decisions

- **DI-1.** Histogram breaks follow R exactly: `nclass.FD` (R 4.3.3), then
  `pretty`, right-closed bins. I13 ports `pretty` (R's C routine) rather than
  using numpy's `"fd"` bins, whose edges differ. Golden tests compare edges and
  counts.
- **DI-2.** Violin data: Gaussian KDE with Silverman's rule, hard span, as
  plotly.js does; Sheather–Jones is not offered.
- **DI-3.** Runtime distributions use successful runs only, and the view
  reports the number of failed runs.

## Open questions for the user

- Q-DI-1: Should the website keep IOHanalyzer's per-algorithm breaks as the
  default, or use equal bins so that algorithms can be compared bin by bin?

## Test tolerance

Bin edges and counts must match R exactly. KDE values must agree to a
relative error of 1e-12 for the same evaluation grid.
