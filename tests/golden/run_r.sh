#!/bin/sh
# Run the R generator inside the reference container. Usage: run_r.sh <image> <golden dir>
set -e
img="$1"; dir="$2"
docker run --rm -u "$(id -u):$(id -g)" -v "$dir:/work" "$img" Rscript /work/r/generate.R /work/fixtures /work/iohanalyzer
