# RVCompare 0.1.8 reference for the cumulative difference plot (WP I16, docs/definitions/cdp.md).
# Two samples of 100 values each, rounded so that some tie, and RVCompare's bootstrap bands with
# 1000 samples. Usage: Rscript rvcompare.R <output dir>
library(RVCompare)
out <- commandArgs(trailingOnly = TRUE)[[1]]
set.seed(20240423)
a <- round(rnorm(100, mean = 2, sd = 1), 1)
b <- round(rnorm(100, mean = 2.1, sd = 0.5), 1)
write.csv(data.frame(a = a, b = b), file.path(out, "samples.csv"), row.names = FALSE)
res <- get_Y_AB_bounds_bootstrap(a, b, alpha = 0.05, EPSILON = 1e-80, nOfBootstrapSamples = 1000,
                                 ignoreMinimumLengthCheck = TRUE)
res <- as.data.frame(res)
keep <- seq(1, nrow(res), by = 10)  # every 10th of the 2000 points keeps the file small
res <- res[keep, ]
res[] <- lapply(res, signif, digits = 15)
write.csv(res, file.path(out, "bootstrap.csv"), row.names = FALSE)
cat(as.character(packageVersion("RVCompare")), "\n", file = file.path(out, "version.txt"))
