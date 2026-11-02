# Commands are run in the pixi dev environment by default. If the environment is
# already activated (e.g. conda), override with: make docs RUN=
RUN ?= pixi run -e dev

.PHONY: help test docs docs-serve

help:
	@echo "make test        Run the test suite (including docstring examples)"
	@echo "make docs        Build the documentation into site/"
	@echo "make docs-serve  Serve the documentation locally with live reload"

test:
	$(RUN) pytest

docs:
	$(RUN) mkdocs build

docs-serve:
	$(RUN) mkdocs serve
