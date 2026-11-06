# References

The definition pages cite these works by key, for example [Hansen2021]. Every
entry was checked against a bibliographic database during the I02 audit
(November 2026): Crossref (`api.crossref.org`), arXiv, OpenAlex or the
publisher's own page. "Checked" means that the authors, title, venue, year and
DOI below were confirmed there. Paywalled papers were not re-read for this
audit. Where a page states what a paper defines, the statement is
cross-checked against the reference implementation by the same authors (cocopp
for COCO, the R packages for their methods) and the page says so.

Corrections to the starting list in `plan/iohinspector.md` are marked
**Correction**.

## Runtime, ERT and COCO

- **[Price1997]** K. V. Price. Differential evolution vs. the functions of the
  2nd ICEO. *Proc. 1997 IEEE International Conference on Evolutionary
  Computation (ICEC '97)*, pp. 153–157, 1997. doi:10.1109/ICEC.1997.592287.
  Checked: Crossref.
- **[Auger2005]** A. Auger, N. Hansen. Performance evaluation of an advanced
  local search evolutionary algorithm. *2005 IEEE Congress on Evolutionary
  Computation (CEC 2005)*, vol. 2, pp. 1777–1784, 2005.
  doi:10.1109/CEC.2005.1554903. Checked: Crossref.
- **[Hansen2010]** N. Hansen, A. Auger, R. Ros, S. Finck, P. Pošík. Comparing
  results of 31 algorithms from the black-box optimization benchmarking
  BBOB-2009. *GECCO '10 Companion (Proc. 12th Annual Conference Companion on
  Genetic and Evolutionary Computation)*, pp. 1689–1696, 2010.
  doi:10.1145/1830761.1830790. Checked: Crossref.
- **[Hansen2021]** N. Hansen, A. Auger, R. Ros, O. Mersmann, T. Tušar,
  D. Brockhoff. COCO: a platform for comparing continuous optimizers in a
  black-box setting. *Optimization Methods and Software* 36(1):114–144, 2021
  (online 2020). doi:10.1080/10556788.2020.1808977. Preprint arXiv:1603.08785.
  Checked: Crossref, arXiv.
- **[Hansen2022]** N. Hansen, A. Auger, D. Brockhoff, T. Tušar. Anytime
  performance assessment in blackbox optimization benchmarking. *IEEE
  Transactions on Evolutionary Computation* 26(6):1293–1305, 2022.
  doi:10.1109/TEVC.2022.3210897. Checked: Crossref.
- **[Hansen2016pa]** N. Hansen, A. Auger, D. Brockhoff, D. Tušar, T. Tušar.
  COCO: Performance assessment. arXiv:1605.03560, 2016. The open preprint
  that [Hansen2022] grew out of; note the extra author (Dejan Tušar).
  Checked: arXiv.
- **[cocopp]** COCO post-processing, Python package `cocopp` 2.9.1 (PyPI sdist,
  read only). Paths in the pages are relative to its `src/` folder.

## IOHprofiler

- **[Doerr2018]** C. Doerr, H. Wang, F. Ye, S. van Rijn, T. Bäck. IOHprofiler:
  a benchmarking and profiling tool for iterative optimization heuristics.
  arXiv:1810.05281, 2018. Checked: arXiv.
- **[Wang2022]** H. Wang, D. Vermetten, F. Ye, C. Doerr, T. Bäck. IOHanalyzer:
  detailed performance analyses for iterative optimization heuristics. *ACM
  Transactions on Evolutionary Learning and Optimization* 2(1):1–29, 2022.
  doi:10.1145/3510426. Checked: Crossref.
- **[Ye2022]** F. Ye, C. Doerr, H. Wang, T. Bäck. Automated configuration of
  genetic algorithms by tuning for anytime performance. *IEEE Transactions on
  Evolutionary Computation* 26(6):1526–1538, 2022.
  doi:10.1109/TEVC.2022.3159087. Checked: Crossref.
- **[IOHanalyzer]** IOHanalyzer R package, reference copy
  `source/IOHanalyzer` at commit 20e14e4. Paths in the pages are relative to
  that folder.
- **[iohinspector]** iohinspector 0.0.8, commit f0c25d3. Paths in the pages
  are relative to `src/iohinspector/`.

## Statistics

- **[Hyndman1996]** R. J. Hyndman, Y. Fan. Sample quantiles in statistical
  packages. *The American Statistician* 50(4):361–365, 1996.
  doi:10.1080/00031305.1996.10473566. Checked: Crossref.
- **[Holm1979]** S. Holm. A simple sequentially rejective multiple test
  procedure. *Scandinavian Journal of Statistics* 6(2):65–70, 1979. JSTOR
  4615733 (no Crossref DOI). Checked: OpenAlex.
- **[Smirnov1948]** N. Smirnov. Table for estimating the goodness of fit of
  empirical distributions. *The Annals of Mathematical Statistics*
  19(2):279–281, 1948. doi:10.1214/aoms/1177730256. Checked: Crossref.
- **[Marsaglia2003]** G. Marsaglia, W. W. Tsang, J. Wang. Evaluating
  Kolmogorov's distribution. *Journal of Statistical Software* 8(18), 2003.
  doi:10.18637/jss.v008.i18. Checked: Crossref.
- **[Scholz1987]** F. W. Scholz, M. A. Stephens. K-sample Anderson–Darling
  tests. *Journal of the American Statistical Association* 82(399):918–924,
  1987. doi:10.1080/01621459.1987.10478517. Checked: Crossref.
- **[Garcia2010]** S. García, A. Fernández, J. Luengo, F. Herrera. Advanced
  nonparametric tests for multiple comparisons in the design of experiments
  in computational intelligence and data mining: experimental analysis of
  power. *Information Sciences* 180(10):2044–2064, 2010.
  doi:10.1016/j.ins.2009.12.010. Checked: Crossref.
- **[Derrac2011]** J. Derrac, S. García, D. Molina, F. Herrera. A practical
  tutorial on the use of nonparametric statistical tests as a methodology for
  comparing evolutionary and swarm intelligence algorithms. *Swarm and
  Evolutionary Computation* 1(1):3–18, 2011. doi:10.1016/j.swevo.2011.02.002.
  Checked: Crossref.
- **[Sturges1926]** H. A. Sturges. The choice of a class interval. *Journal of
  the American Statistical Association* 21(153):65–66, 1926.
  doi:10.1080/01621459.1926.10502161. Checked: Crossref.
- **[Scott1979]** D. W. Scott. On optimal and data-based histograms.
  *Biometrika* 66(3):605–610, 1979. doi:10.1093/biomet/66.3.605.
  Checked: Crossref.
- **[Freedman1981]** D. Freedman, P. Diaconis. On the histogram as a density
  estimator: L2 theory. *Zeitschrift für Wahrscheinlichkeitstheorie und
  verwandte Gebiete* 57(4):453–476, 1981. doi:10.1007/BF01025868.
  Checked: Crossref.

- **[Silverman1986]** B. W. Silverman. *Density Estimation for Statistics
  and Data Analysis*. Chapman & Hall, London, 1986; reissued by Routledge,
  2018, doi:10.1201/9781315140919. Checked: Crossref (reissue). Source of the
  rule-of-thumb kernel bandwidth.
- **[Sheather1991]** S. J. Sheather, M. C. Jones. A reliable data-based
  bandwidth selection method for kernel density estimation. *Journal of the
  Royal Statistical Society, Series B* 53(3):683–690, 1991.
  doi:10.1111/j.2517-6161.1991.tb01857.x. Checked: Crossref.
- **[R]** R Core Team. R 4.3.3 (2024-02-29), base functions `quantile`,
  `hist.default`, `nclass.FD`, `pretty`, `round`. Read from the installed R
  in this environment (`Rscript -e 'body(nclass.FD)'`).
- **[plotly.js]** plotly.js, violin and box traces, read at tag v2.25.2, the
  version bundled with the plotly R package 4.12.1:
  `src/traces/violin/calc.js` (bandwidth) and `src/traces/box/attributes.js`
  (quartile method). <https://github.com/plotly/plotly.js>.

## Algorithm configuration

- **[Hutter2009]** F. Hutter, H. H. Hoos, K. Leyton-Brown, T. Stützle.
  ParamILS: an automatic algorithm configuration framework. *Journal of
  Artificial Intelligence Research* 36:267–306, 2009. doi:10.1613/jair.2861.
  Checked: Crossref. Cited for penalised average runtime (PAR-k).

## Attainment functions and AOCC

- **[LopezIbanez2025]** M. López-Ibáñez, D. Vermetten, J. Dreo, C. Doerr.
  Using the empirical attainment function for analyzing single-objective
  black-box optimization algorithms. *IEEE Transactions on Evolutionary
  Computation* 29(5):1774–1782, 2025 (online 2024).
  doi:10.1109/TEVC.2024.3462758. Checked: Crossref. **Correction:** the plan
  lists it as TEVC 2024; the issue is 2025. Open preprint arXiv:2404.02031
  (v3). Its abstract states that the target-based ECDF approximates the EAF
  and that the average area over the convergence curves is an equivalent
  measure of anytime performance. Checked: Crossref, arXiv abstract.
- **[LopezIbanez2010]** M. López-Ibáñez, L. Paquete, T. Stützle. Exploratory
  analysis of stochastic local search algorithms in biobjective optimization.
  In *Experimental Methods for the Analysis of Optimization Algorithms*,
  Springer, pp. 209–222, 2010.
  doi:10.1007/978-3-642-02538-9_9. Checked: Crossref.
- **[GrunertdaFonseca2001]** V. Grunert da Fonseca, C. M. Fonseca,
  A. O. Hall. Inferential performance assessment of stochastic optimisers and
  the attainment function. *Evolutionary Multi-Criterion Optimization
  (EMO 2001)*, Lecture Notes in Computer Science, Springer, pp. 213–225, 2001. doi:10.1007/3-540-44719-9_15.
  Checked: Crossref.

## Ranking and portfolios

- **[Glickman2022]** M. E. Glickman. Example of the Glicko-2 system.
  Technical note, Boston University, dated 22 March 2022.
  <https://www.glicko.net/glicko/glicko2.pdf>. Checked: read in full.
- **[Glickman1999]** M. E. Glickman. Parameter estimation in large dynamic
  paired comparison experiments. *Journal of the Royal Statistical Society,
  Series C (Applied Statistics)* 48(3):377–394, 1999.
  doi:10.1111/1467-9876.00159. Checked: Crossref.
- **[PlayerRatings]** A. Stephenson, J. Sonas. PlayerRatings: Dynamic updating
  methods for player ratings estimation. R package version 1.1-0 (CRAN,
  2020-03-01). doi:10.32614/CRAN.package.PlayerRatings. Checked: CRAN.
- **[RVCompare]** E. Arza. RVCompare: compare real valued random variables.
  R package version 0.1.8 (CRAN, 2023-08-21); read from the source tarball.
  Reference implementation of [Arza2023]. Checked: CRAN.
- **[eaf]**, **[moocore]**: the R package `eaf` (used by IOHanalyzer) and the
  Python package `moocore` (a dependency of iohinspector), both maintained by
  M. López-Ibáñez. Versions are pinned in I03 and I17.
- **[Shapley1953]** L. S. Shapley. A value for n-person games. In H. W. Kuhn,
  A. W. Tucker (eds.), *Contributions to the Theory of Games II*, Annals of
  Mathematics Studies 28, Princeton University Press, pp. 307–318, 1953.
  doi:10.1515/9781400881970-018. Checked: Crossref.
- **[Frechette2016]** A. Fréchette, L. Kotthoff, T. Michalak, T. Rahwan,
  H. H. Hoos, K. Leyton-Brown. Using the Shapley value to analyze algorithm
  portfolios. *Proceedings of the AAAI Conference on Artificial Intelligence*
  30(1), 2016. doi:10.1609/aaai.v30i1.10440. Checked: Crossref.
- **[Arza2023]** E. Arza, J. Ceberio, E. Irurozki, A. Pérez. Comparing two
  samples through stochastic dominance: a graphical approach. *Journal of
  Computational and Graphical Statistics* 32(2):551–566, 2023 (online 2022).
  doi:10.1080/10618600.2022.2084405. Checked: Crossref.

## Deep Statistical Comparison

The user supplied [Eftimov2020] and [DSCTool2022] as PDFs for the DSC
specification. Both are copyrighted and stay outside the repository. The
other DSC papers were not supplied and were not read.

- **[Eftimov2017]** T. Eftimov, P. Korošec, B. Koroušić Seljak. A novel
  approach to statistical comparison of meta-heuristic stochastic
  optimization algorithms using deep statistics. *Information Sciences*
  417:186–215, 2017. doi:10.1016/j.ins.2017.07.015. Checked: Crossref. Not
  read: it defines the DSC ranking scheme in full.
- **[Eftimov2017b]** T. Eftimov, P. Korošec, B. Koroušić Seljak. The behavior
  of deep statistical comparison approach for different criteria of comparing
  distributions. *Proceedings of the 9th International Joint Conference on
  Computational Intelligence (IJCCI 2017)*, pp. 73–82, 2017.
  doi:10.5220/0006499900730082. Checked: Crossref (pages from the reference
  list of [Eftimov2020], item 29). Not read. [Eftimov2020] cites it for the
  choice between the KS and AD tests.
- **[Eftimov2019]** T. Eftimov, P. Korošec. Identifying practical significance
  through statistical comparison of meta-heuristic stochastic optimization
  algorithms. *Applied Soft Computing* 85:105862, 2019.
  doi:10.1016/j.asoc.2019.105862. Checked: Crossref. Not read: it defines
  practical DSC in full.
- **[Eftimov2020]** T. Eftimov, G. Petelin, P. Korošec. DSCTool: a
  web-service-based framework for statistical comparison of stochastic
  optimization algorithms. *Applied Soft Computing* 87:105977, 2020 (online
  5 December 2019). doi:10.1016/j.asoc.2019.105977. Open access, CC BY-NC-ND
  4.0. Checked: Crossref, and read in full from the supplied copy
  (`ioh_web/dsc/paper.pdf` in the rebuild workspace, gitignored). Page
  numbers cited in [dsc](dsc.md) are the journal's pages 1–11.
- **[DSCTool2022]** T. Eftimov, G. Petelin, U. Škvorc, P. Korošec. DSCTool: a
  web-service-based framework for statistical comparison of stochastic
  optimization algorithms, version 1.5. Service documentation, 20 January
  2022, 14 pages. Note the extra author (Urban Škvorc). The v1.4 edition was
  served at `https://ws.ijs.si:8443/dsc-1.4/documentation.pdf`
  ([Eftimov2020] p. 5). Checked: read in full from the supplied copy
  (`ioh_web/dsc/documentation.pdf`, gitignored). Page numbers cited in
  [dsc](dsc.md) are PDF pages. From page 2 on, the printed page number is one
  lower. Its section 3 and [Eftimov2020] section 5 link example requests and
  responses at `https://cs.ijs.si/dl/dsctool/` (`dsc.json`, `dsc.result`,
  `omnibus.json`, `omnibus.result`, `posthoc.json`, `posthoc.result` and
  others). These answered without credentials on 6 November 2026. They were
  not downloaded.
- **[dsc-core]** P. Korošec (repository owner). dsc-core, the core library of
  DSCTool (Java, with an R script). <https://repo.ijs.si/korosec/dsc-core>,
  named in [Eftimov2020] pp. 3 and 5. Licence GPL-3.0-or-later. Last commit
  4b2f2d9 of 19 August 2019, which is older than service version 1.5. Checked:
  repository metadata and licence header only. The source was not read (see
  Q-DS-3 in [dsc](dsc.md)).
