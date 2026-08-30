PYTHON ?= python
.PHONY: install test demo evaluate
install:
	$(PYTHON) -m pip install -e .
test:
	$(PYTHON) -m unittest discover -s tests -v
demo:
	$(PYTHON) -m drift_sentinel demo --output results/demo.json
evaluate:
	$(PYTHON) -m drift_sentinel evaluate --output results/evaluation.json
