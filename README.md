# CentroidSimilarity -- simple classification rule of high-dimensional data

CentroidSimilarity includes two main classes:
- CentroidSimilarity
- CentroidSimilarityFeatureSelection

In the training phase, we evaluate the per-coordinate mean and standard deviation of each class.
CentroidSimilarityFeatureSelection also identifies coordinates distinguishing the classes and masks
the non-distinguishing ones. This identification is done using one of the methods 'one_vs_all' or
'diversity_pursuit'.

## Example:
```
from centroid_similarity import CentroidSimilarityFeatureSelection, CentroidSimilarity

c = 2
n = 100
d = 1000

mu = 5 / np.sqrt( n ) # signal strength

eps = .01 # non sparsity rate

sparsity_pattern = np.random.rand(d) < eps
contrast_vector = mu * sparsity_pattern

y = np.random.randint(c,size = n)
X = (np.expand_dims(2*(y - 1/2), 1) * contrast_vector) + np.random.randn(n, d) 

test_train_split_ratio = .2
train_indices = np.random.rand(n) > test_train_split_ratio

X_train, X_test = X[train_indices], X[~train_indices]
y_train, y_test = y[train_indices], y[~train_indices]

clf_naive = CentroidSimilarity()
clf_fs = CentroidSimilarityFeatureSelection()

clf_naive.fit(X_train, y_train)
clf_fs.fit(X_train, y_train, method='one_vs_all')
y_pred_naive = clf_naive.predict(X_test)
y_pred_fs = clf_fs.predict(X_test)
print("Accuracy (naive) = ", np.mean(y_pred_naive == y_test))
print("Accuracy (feature selection) = ", np.mean(y_pred_fs == y_test))



```

## Experiments

To reproduce the synthetic experiments and figures used during development, run:

```
python3 experiments/run_centroid_similarity_experiments.py
```

Figures are saved under `experiments/Figs/`.

Optional: install experiment dependencies in your environment:

```
pip install -r experiments/requirements.txt
```

Quick run for faster verification (smaller grids and iterations):

```
python3 experiments/run_centroid_similarity_experiments.py --quick --seed 0
```

Customizable arguments:
- `--p` number of features (default 10000; reduced if `--quick`)
- `--n` number of samples (default: `int(2*log(p)^2)`)
- `--c` number of classes (default 10; 5 with `--quick`)
- `--sig` noise standard deviation (default 1.0)
- `--test-frac` test split fraction (default 0.2)
- `--beta-start/--beta-stop/--beta-steps` grid for beta (default 0.5–0.9, 5 steps)
- `--r-start/--r-stop/--r-steps` grid for r (default 0.01–0.3, 7 steps)
- `--nMonte` Monte Carlo iterations per grid point (default 10; 1 with `--quick`)
- `--seed` random seed
- `--outdir` output directory for figures (default `experiments/Figs`)
- `--save-csv` path to save a CSV of the results dataframe

### Make targets

Alternatively, use the provided Makefile targets (pass extra CLI via `ARGS`):

```
make install-exp                  # pip install experiment deps
make quick-exp OUTDIR=tmp/figs    # fast sanity run (seeded); custom OUTDIR optional
make quick-exp ARGS="--p 2000 --beta-steps 4"  # override defaults
make exp ARGS="--seed 0"          # full run (can pass extra ARGS)
make exp-csv RESULTS=tmp/res.csv  # full run and save results CSV
make quick-csv RESULTS=tmp/res.csv  # quick run and save results CSV
make clean                        # remove OUTDIR and RESULTS (defaults)
```

### Notes

- The Makefile prefers the project's `.venv/bin/python` if it exists; otherwise it falls back to `python3`.
- If you see Matplotlib cache warnings, set a writable cache dir (optional but recommended):
  - `export MPLCONFIGDIR=.cache/matplotlib`
- For a one-off quick run without Makefile:
  - `.venv/bin/python experiments/run_centroid_similarity_experiments.py --quick --seed 0`
