# Reference container for the golden outputs (WP I03).
# CRAN packages come from the Posit snapshot of 2024-04-23 that rocker/r-ver:4.3.3 is configured with,
# so every version is fixed by the base image digest.
FROM rocker/r-ver:4.3.3@sha256:732d15020af326da9e919c07f70ca32bf5d3e409220af32e0a4b6d0a89437309

RUN apt-get update && apt-get install -y --no-install-recommends \
      git libgsl-dev libglpk-dev libcurl4-openssl-dev libssl-dev libxml2-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Dependencies of IOHanalyzer plus the three packages named in the plan, all from the snapshot.
RUN Rscript -e 'options(Ncpus = 4); install.packages(c( \
      "Rcpp", "magrittr", "dplyr", "data.table", "ggplot2", "plotly", "colorspace", \
      "RColorBrewer", "shiny", "reshape2", "stringi", "httr", "knitr", "rjson", "viridis", \
      "eaf", "PlayerRatings", "RVCompare", "arrow", "igraph", "jsonlite"))' \
 && Rscript -e 'for (p in c("eaf","PlayerRatings","RVCompare","arrow","Rcpp")) library(p, character.only = TRUE)'

# IOHanalyzer at caecada^ (tree exported with `git archive`, see tests/golden/README.md).
COPY ctx/IOHanalyzer /opt/IOHanalyzer
RUN R CMD INSTALL --no-docs --no-multiarch /opt/IOHanalyzer \
 && Rscript -e 'library(IOHanalyzer); cat(as.character(packageVersion("IOHanalyzer")), "\n")'

WORKDIR /work
