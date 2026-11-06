# Golden reference outputs (WP I03). See tests/golden/README.md.
#
#   make golden           build the R image if needed, regenerate the IOHanalyzer and cocopp references
#   make golden-check     regenerate and fail if any committed output changes (byte for byte)
#   make golden-fixtures  regenerate the ioh 0.4.0 fixture logs (needs the golden venv)
#
# Docker may need a wrapper: sg docker -c 'make golden'
DOCKER      ?= docker
IMAGE       ?= ioh-golden:1
GOLDEN      := $(CURDIR)/tests/golden
IOHANALYZER ?= /home/jacob/code/ioh_web/source/IOHanalyzer
GOLDEN_VENV ?= $(CURDIR)/.venv
GOLDEN_PY   := $(GOLDEN_VENV)/bin/python

.PHONY: golden golden-ctx golden-image golden-r golden-cocopp golden-venv golden-fixtures golden-check

golden: golden-r golden-cocopp

# IOHanalyzer at caecada^ (the parent of the commit that broke c.DataSet), exported without history.
golden-ctx:
	test -d $(GOLDEN)/r/ctx/IOHanalyzer || { mkdir -p $(GOLDEN)/r/ctx/IOHanalyzer && \
	  git -C $(IOHANALYZER) archive caecada^ | tar -x -C $(GOLDEN)/r/ctx/IOHanalyzer; }

golden-image: golden-ctx
	$(DOCKER) image inspect $(IMAGE) >/dev/null 2>&1 || $(DOCKER) build -t $(IMAGE) $(GOLDEN)/r

golden-r: golden-image
	$(DOCKER) run --rm -u $$(id -u):$$(id -g) -v $(GOLDEN):/work $(IMAGE) \
	  Rscript /work/r/generate.R /work/fixtures /work/iohanalyzer

golden-venv:
	test -x $(GOLDEN_PY) || uv venv --python 3.11 $(GOLDEN_VENV)
	uv pip install --python $(GOLDEN_PY) -r $(GOLDEN)/requirements-golden.txt

golden-cocopp: golden-venv
	$(GOLDEN_PY) $(GOLDEN)/cocopp/generate.py $(GOLDEN)/fixtures $(GOLDEN)/cocopp/coco

golden-fixtures: golden-venv
	$(GOLDEN_PY) $(GOLDEN)/make_fixtures.py

# Regenerate and fail if any committed output changes or an uncommitted one appears.
golden-check: golden
	git diff --exit-code --stat -- tests/golden/iohanalyzer tests/golden/cocopp tests/golden/fixtures
	test -z "$$(git ls-files --others --exclude-standard -- tests/golden)"
