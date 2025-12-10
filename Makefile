# Prefer local virtualenv's Python if available, else fallback to system python3
PY ?= python3
ifneq (,$(wildcard .venv/bin/python))
	PY := .venv/bin/python
endif
OUTDIR ?= experiments/Figs
RESULTS ?= experiments/results.csv
ARGS ?=

.PHONY: install-exp exp quick-exp

install-exp:
	$(PY) -m pip install -r experiments/requirements.txt

exp:
	$(PY) experiments/run_centroid_similarity_experiments.py --outdir $(OUTDIR) $(ARGS)

quick-exp:
	$(PY) experiments/run_centroid_similarity_experiments.py --quick --seed 0 --outdir $(OUTDIR) $(ARGS)

.PHONY: exp-csv quick-csv clean

exp-csv:
	$(PY) experiments/run_centroid_similarity_experiments.py --outdir $(OUTDIR) --save-csv $(RESULTS) $(ARGS)

quick-csv:
	$(PY) experiments/run_centroid_similarity_experiments.py --quick --seed 0 --outdir $(OUTDIR) --save-csv $(RESULTS) $(ARGS)

clean:
	rm -rf $(OUTDIR)
	rm -f $(RESULTS)
