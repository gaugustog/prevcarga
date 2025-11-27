# Makefile for Sphinx documentation
#
# You can set these variables from the command line:
# make html SPHINXOPTS="-W --keep-going"

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Colors for terminal output
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

.PHONY: help
help:
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@echo '  ${GREEN}help${RESET}       Show this help message'
	@echo '  ${GREEN}html${RESET}       Build HTML documentation'
	@echo '  ${GREEN}clean${RESET}      Remove build directory'
	@echo '  ${GREEN}livehtml${RESET}   Build and serve with auto-reload'
	@echo '  ${GREEN}linkcheck${RESET}  Check all external links'
	@echo '  ${GREEN}coverage${RESET}   Check documentation coverage'
	@echo '  ${GREEN}doctest${RESET}    Run doctests in documentation'
	@echo '  ${GREEN}pdf${RESET}        Build PDF documentation (requires LaTeX)'
	@echo ''

.PHONY: clean
clean:
	@echo "Cleaning build directory..."
	rm -rf $(BUILDDIR)/*
	@echo "Done."

.PHONY: html
html:
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

.PHONY: livehtml
livehtml:
	@echo "Starting live server with auto-reload..."
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O) \
		--port 8000 \
		--open-browser \
		--watch ../src

.PHONY: linkcheck
linkcheck:
	@echo "Checking external links..."
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Link check complete. Results in $(BUILDDIR)/linkcheck."

.PHONY: coverage
coverage:
	@echo "Checking documentation coverage..."
	@$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Coverage report in $(BUILDDIR)/coverage."

.PHONY: doctest
doctest:
	@echo "Running doctests..."
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Doctest results in $(BUILDDIR)/doctest."

.PHONY: pdf
pdf:
	@echo "Building PDF documentation..."
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "PDF available at $(BUILDDIR)/latex/*.pdf"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
