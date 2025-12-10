#!/usr/bin/env python3
"""
Reproduce the experiments from the notebook:
  ../transfer-learning-with-feature-selection/Code/normally distributed clusters.ipynb

Adapted to run fully within this CentSim repository, using the local
`centroid_similarity` implementation and saving figures under experiments/Figs.

Two experiment groups are implemented:
  1) Feature dimension scan (vary p, fixed classes)
  2) Number of classes scan (vary c)

This script avoids external dependencies like sklearn/hyperopt by default.
If scikit-learn is installed, an SVM baseline will be computed and added
to the relevant plots; otherwise it is skipped.
"""

import os
import math
import argparse
import numpy as np
import matplotlib

# Non-interactive backend for saving figures
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm

# Local imports from this repository (robust to running from experiments/)
try:
    from centroid_similarity import CentroidSimilarity, CentroidSimilarityFeatureSelection
except Exception:
    import sys
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from centroid_similarity import CentroidSimilarity, CentroidSimilarityFeatureSelection


# ---------------------------------
# Optional scikit-learn dependency
# ---------------------------------
_HAVE_SKLEARN = False
try:
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


# ---------------------------
# Synthetic data generation
# ---------------------------
def sample_centroids(num_classes: int,
                     num_features: int,
                     eps: float,
                     power: float,
                     non_nulls_location: str = 'free') -> np.ndarray:
    """Randomly sample sparse class means (centroids).

    Args:
    - num_classes: number of classes
    - num_features: number of features
    - eps: fraction of non-null features per class
    - power: amplitude assigned to non-null features
    - non_nulls_location: 'free' (different per class) or 'fixed' (shared)
    """
    if non_nulls_location == 'fixed':
        idcs = np.random.rand(num_features) < eps

    centroids = np.zeros((num_classes, num_features))
    for i in range(num_classes):
        if non_nulls_location == 'free':
            idcs = np.random.rand(num_features) < eps
        # make some non-null features negative, some positive
        centroids[i][idcs] = power * (1 - 2 * (np.random.rand(np.sum(idcs)) > .5)) / 2
    return centroids


def sample_normal_clusters(centroids: np.ndarray, n: int, sigma: float):
    """Sample noisy data around the provided centroids.

    Args:
    - centroids: class means matrix (c x p)
    - n: number of samples to draw
    - sigma: noise standard deviation
    """
    c, p = centroids.shape
    Z = np.random.randn(n, p)
    y = np.random.randint(c, size=n)
    X = centroids[y] + sigma * Z
    return X, y


def train_test_split_mask(n: int, test_frac: float):
    """Return boolean mask to split n samples into train/test by fraction."""
    test_mask = np.random.rand(n) < test_frac
    train_mask = ~test_mask
    return train_mask, test_mask


def pairwise_distances_euclidean(A: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances for rows of A.

    Equivalent to sklearn.metrics.pairwise_distances(A), but avoids dependency.
    """
    sq_norms = np.sum(A * A, axis=1, keepdims=True)
    d2 = sq_norms + sq_norms.T - 2.0 * (A @ A.T)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _oracle_with_true_means(cs: CentroidSimilarity, centroids: np.ndarray):
    # Fit to set internal shapes, then inject true centroids and normalize rows
    # (mirrors logic used elsewhere in the repo)
    eps_norm = 1e-10
    cs._cls_mean = np.array(centroids, copy=True)
    norms = np.linalg.norm(cs._cls_mean, axis=1, keepdims=True)
    cs._mat = np.where(norms >= eps_norm, cs._cls_mean / norms, np.zeros_like(cs._cls_mean))
    cs._mat = np.nan_to_num(cs._mat, nan=0.0, posinf=0.0, neginf=0.0)


def _svm_baseline(X_train, y_train, X_test, y_test):
    if not _HAVE_SKLEARN:
        return None
    # Simple baseline without hyperopt to avoid extra dependency
    # Try a small grid of reasonable SVM params
    grid = [
        dict(C=1.0, kernel='linear'),
        dict(C=1.0, kernel='rbf', gamma='scale'),
        dict(C=10.0, kernel='rbf', gamma='scale'),
    ]
    best_acc, best_params = -1.0, None
    for params in grid:
        clf = SVC(**params)
        clf.fit(X_train, y_train)
        acc = float(accuracy_score(y_test, clf.predict(X_test)))
        if acc > best_acc:
            best_acc, best_params = acc, params
    return best_acc


def feature_dim_scan(figs_dir: str,
                     num_experiments: int = 50,
                     features_num_array = (100, 500, 1000, 5000, 10000),
                     num_classes: int = 3,
                     test_frac: float = 0.9,
                     sig: float = 1.0,
                     beta: float = 0.6):
    """Vary p (features) and evaluate accuracies, recall, precision, etc."""
    features_num_array = np.array(features_num_array)
    accuracies = np.empty((5, len(features_num_array), num_experiments))
    features_recall = np.empty((3, len(features_num_array), num_experiments))
    features_precision = np.empty((3, len(features_num_array), num_experiments))
    num_separating_features = np.empty((len(features_num_array), num_experiments))
    ref_model_accuracies = np.full((len(features_num_array),), np.nan)

    for i, p in enumerate(features_num_array):
        n_train = int(3 * (math.log(p) ** 2))
        n = int(n_train / (1 - test_frac))
        eps = p ** (-beta)
        r = 0.01 * num_classes
        mu = math.sqrt(r * math.log(p))

        # One reference SVM model (if available)
        if _HAVE_SKLEARN:
            centroids = sample_centroids(num_classes=num_classes, num_features=p, eps=eps, power=mu, non_nulls_location='fixed')
            X, y = sample_normal_clusters(centroids, n, sig)
            train_mask, test_mask = train_test_split_mask(len(X), test_frac)
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            ref_model_accuracies[i] = _svm_baseline(X_train, y_train, X_test, y_test) or np.nan

        for j in range(num_experiments):
            centroids = sample_centroids(num_classes=num_classes, num_features=p, eps=eps, power=mu, non_nulls_location='fixed')
            X, y = sample_normal_clusters(centroids, n, sig)
            true_mask = (centroids != 0)
            num_separating_features[i, j] = int(np.sum(true_mask))

            train_mask, test_mask = train_test_split_mask(len(X), test_frac)
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            # Naive
            cs = CentroidSimilarity()
            cs.fit(X_train, y_train)
            accuracies[0, i, j] = cs.eval_accuracy(X_test, y_test)
            rec, prec = cs.get_mask_prec_recall(true_mask)
            features_precision[0, i, j] = prec
            features_recall[0, i, j] = rec

            # One-vs-All
            cs_ova = CentroidSimilarityFeatureSelection()
            cs_ova.fit(X_train, y_train, method='one_vs_all')
            accuracies[1, i, j] = cs_ova.eval_accuracy(X_test, y_test)
            rec, prec = cs_ova.get_mask_prec_recall(true_mask)
            features_precision[1, i, j] = prec
            features_recall[1, i, j] = rec

            # Diversity pursuit
            cs_dp = CentroidSimilarityFeatureSelection()
            cs_dp.fit(X_train, y_train, method='diversity_pursuit')
            accuracies[2, i, j] = cs_dp.eval_accuracy(X_test, y_test)
            rec, prec = cs_dp.get_mask_prec_recall(true_mask)
            features_precision[2, i, j] = prec
            features_recall[2, i, j] = rec

            # Oracle: true feature mask
            cs_oracle = CentroidSimilarity()
            cs_oracle.fit(X_train, y_train)
            cs_oracle.set_mask(true_mask)
            accuracies[3, i, j] = cs_oracle.eval_accuracy(X_test, y_test)

            # Oracle: true centroids
            cs_oracle_mu = CentroidSimilarity()
            cs_oracle_mu.fit(X_train, y_train)
            _oracle_with_true_means(cs_oracle_mu, centroids)
            accuracies[4, i, j] = cs_oracle_mu.eval_accuracy(X_test, y_test)

    # ----- Figures -----
    ensure_dir(figs_dir)

    # Figure: sample size and separating features vs p
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(features_num_array, 3 * np.log(features_num_array) ** 2, 'o')
    ax[0].set_xlabel('number of features')
    ax[0].set_ylabel('num. of samples')
    ax[0].set_title('number of observations')
    ax[1].plot(features_num_array, np.mean(num_separating_features, axis=1), 'o')
    ax[1].set_xlabel('number of features')
    ax[1].set_ylabel('num. of features')
    ax[1].set_title('number of separating features')
    fig.tight_layout()
    out = os.path.join(figs_dir, 'feature_dim_scan_observations_and_separating_features.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Figure: accuracies vs p
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks(np.arange(len(features_num_array)))
    ax.bar(x=np.arange(len(features_num_array)) - 0.2, height=np.mean(accuracies[0], axis=1), width=0.1, label='naive')
    ax.bar(x=np.arange(len(features_num_array)) - 0.1, height=np.mean(accuracies[1], axis=1), width=0.1, label='one-vs-all')
    ax.bar(x=np.arange(len(features_num_array)) - 0.0, height=np.mean(accuracies[2], axis=1), width=0.1, label='diversity-pursuit')
    ax.bar(x=np.arange(len(features_num_array)) + 0.1, height=np.mean(accuracies[3], axis=1), width=0.1, label='feature-oracle')
    ax.bar(x=np.arange(len(features_num_array)) + 0.2, height=np.mean(accuracies[4], axis=1), width=0.1, label='centroids-oracle')
    if _HAVE_SKLEARN and not np.all(np.isnan(ref_model_accuracies)):
        ax.bar(x=np.arange(len(features_num_array)) + 0.3, height=ref_model_accuracies, width=0.1, label='SVM')
    ax.set_xlabel('features dimension')
    ax.set_xticklabels(features_num_array)
    ax.set_ylabel('accuracy')
    ax.set_title('feature selection accuracy')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(figs_dir, 'feature_dim_scan_accuracies.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Figure: recall vs p
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks(np.arange(len(features_num_array)))
    ax.bar(x=np.arange(len(features_num_array)) - 0.1, height=np.mean(features_recall[0], axis=1), width=0.1, label='naive')
    ax.bar(x=np.arange(len(features_num_array)) + 0.1, height=np.mean(features_recall[1], axis=1), width=0.1, label='one-vs-all')
    ax.bar(x=np.arange(len(features_num_array)) + 0.0, height=np.mean(features_recall[2], axis=1), width=0.1, label='diversity-pursuit')
    ax.set_xlabel('features dimension')
    ax.set_xticklabels(features_num_array)
    ax.set_ylabel('recall')
    ax.set_title('feature selection recall')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(figs_dir, 'feature_dim_scan_recall.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Figure: precision vs p
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks(np.arange(len(features_num_array)))
    ax.bar(x=np.arange(len(features_num_array)) - 0.1, height=np.mean(features_precision[0], axis=1), width=0.1, label='naive')
    ax.bar(x=np.arange(len(features_num_array)) + 0.1, height=np.mean(features_precision[1], axis=1), width=0.1, label='one-vs-all')
    ax.bar(x=np.arange(len(features_num_array)) + 0.0, height=np.mean(features_precision[2], axis=1), width=0.1, label='diversity-pursuit')
    ax.set_xlabel('features dimension')
    ax.set_xticklabels(features_num_array)
    ax.set_ylabel('precision')
    ax.set_title('feature selection precision')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(figs_dir, 'feature_dim_scan_precision.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)


def num_classes_scan(figs_dir: str,
                     num_experiments: int = 100,
                     num_classes_array=(2, 3, 5, 10),
                     num_features: int = 5000,
                     test_frac: float = 0.9,
                     sig: float = 1.0,
                     beta: float = 0.6):
    """Vary number of classes and evaluate metrics, including signal power plot.

    Runs two variants:
      (A) mu depends on classes (for signal-power vs classes plot)
      (B) fixed mu=0.5 for number of significant features plot
    """
    num_classes_array = np.array(num_classes_array)

    # Variant A (mu depends on c)
    accuracies_A = np.empty((5, len(num_classes_array), num_experiments))
    features_recall_A = np.empty((3, len(num_classes_array), num_experiments))
    features_precision_A = np.empty((3, len(num_classes_array), num_experiments))
    ref_model_accuracies_A = np.full((len(num_classes_array),), np.nan)
    sig_power = []

    for i, c in enumerate(num_classes_array):
        p = int(num_features)
        n_train = int(3 * (math.log(p) ** 2))
        n = int(n_train / (1 - test_frac))
        eps = p ** (-beta)
        r = 0.005 * c
        mu = math.sqrt(r * math.log(p))
        sig_power.append(mu)

        # SVM baseline (if available)
        if _HAVE_SKLEARN:
            centroids = sample_centroids(num_classes=c, num_features=p, eps=eps, power=mu, non_nulls_location='fixed')
            X, y = sample_normal_clusters(centroids, n, sig)
            train_mask, test_mask = train_test_split_mask(len(X), test_frac)
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            ref_model_accuracies_A[i] = _svm_baseline(X_train, y_train, X_test, y_test) or np.nan

        for j in range(num_experiments):
            centroids = sample_centroids(num_classes=c, num_features=p, eps=eps, power=mu, non_nulls_location='fixed')
            X, y = sample_normal_clusters(centroids, n, sig)
            true_mask = (centroids != 0)
            train_mask, test_mask = train_test_split_mask(len(X), test_frac)
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            cs = CentroidSimilarity(); cs.fit(X_train, y_train)
            accuracies_A[0, i, j] = cs.eval_accuracy(X_test, y_test)
            rec, prec = cs.get_mask_prec_recall(true_mask)
            features_precision_A[0, i, j] = prec
            features_recall_A[0, i, j] = rec

            cs_ova = CentroidSimilarityFeatureSelection(); cs_ova.fit(X_train, y_train, method='one_vs_all')
            accuracies_A[1, i, j] = cs_ova.eval_accuracy(X_test, y_test)
            rec, prec = cs_ova.get_mask_prec_recall(true_mask)
            features_precision_A[1, i, j] = prec
            features_recall_A[1, i, j] = rec

            cs_dp = CentroidSimilarityFeatureSelection(); cs_dp.fit(X_train, y_train, method='diversity_pursuit')
            accuracies_A[2, i, j] = cs_dp.eval_accuracy(X_test, y_test)
            rec, prec = cs_dp.get_mask_prec_recall(true_mask)
            features_precision_A[2, i, j] = prec
            features_recall_A[2, i, j] = rec

            cs_oracle = CentroidSimilarity(); cs_oracle.fit(X_train, y_train)
            cs_oracle.set_mask(true_mask)
            accuracies_A[3, i, j] = cs_oracle.eval_accuracy(X_test, y_test)

            cs_oracle_mu = CentroidSimilarity(); cs_oracle_mu.fit(X_train, y_train)
            _oracle_with_true_means(cs_oracle_mu, centroids)
            accuracies_A[4, i, j] = cs_oracle_mu.eval_accuracy(X_test, y_test)

    # Figures for Variant A
    ensure_dir(figs_dir)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(num_classes_array, sig_power, 'o')
    ax.set_xlabel('number of classes')
    ax.set_ylabel('signal mean')
    ax.set_title('signal mean vs. number of classes')
    fig.tight_layout()
    out = os.path.join(figs_dir, 'num_classes_scan_signal_mean.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks(np.arange(len(num_classes_array)))
    ax.bar(x=np.arange(len(num_classes_array)) - 0.2, height=np.mean(accuracies_A[0], axis=1), width=0.1, label='naive')
    ax.bar(x=np.arange(len(num_classes_array)) - 0.1, height=np.mean(accuracies_A[1], axis=1), width=0.1, label='one-vs-all')
    ax.bar(x=np.arange(len(num_classes_array)) - 0.0, height=np.mean(accuracies_A[2], axis=1), width=0.1, label='diversity-pursuit')
    ax.bar(x=np.arange(len(num_classes_array)) + 0.1, height=np.mean(accuracies_A[3], axis=1), width=0.1, label='feature-oracle')
    ax.bar(x=np.arange(len(num_classes_array)) + 0.2, height=np.mean(accuracies_A[4], axis=1), width=0.1, label='centroids-oracle')
    if _HAVE_SKLEARN and not np.all(np.isnan(ref_model_accuracies_A)):
        ax.bar(x=np.arange(len(num_classes_array)) + 0.3, height=ref_model_accuracies_A, width=0.1, label='SVM')
    ax.set_xlabel('number of classes')
    ax.set_xticklabels(num_classes_array)
    ax.set_ylabel('accuracy')
    ax.set_title('feature selection accuracy')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(figs_dir, 'num_classes_scan_accuracies.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks(np.arange(len(num_classes_array)))
    ax.bar(x=np.arange(len(num_classes_array)) - 0.1, height=np.mean(features_recall_A[0], axis=1), width=0.1, label='naive')
    ax.bar(x=np.arange(len(num_classes_array)) + 0.1, height=np.mean(features_recall_A[1], axis=1), width=0.1, label='one-vs-all')
    ax.bar(x=np.arange(len(num_classes_array)) + 0.0, height=np.mean(features_recall_A[2], axis=1), width=0.1, label='diversity-pursuit')
    ax.set_xlabel('number of classes')
    ax.set_xticklabels(num_classes_array)
    ax.set_ylabel('recall')
    ax.set_title('feature selection recall')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(figs_dir, 'num_classes_scan_recall.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks(np.arange(len(num_classes_array)))
    ax.bar(x=np.arange(len(num_classes_array)) - 0.1, height=np.mean(features_precision_A[0], axis=1), width=0.1, label='naive')
    ax.bar(x=np.arange(len(num_classes_array)) + 0.1, height=np.mean(features_precision_A[1], axis=1), width=0.1, label='one-vs-all')
    ax.bar(x=np.arange(len(num_classes_array)) + 0.0, height=np.mean(features_precision_A[2], axis=1), width=0.1, label='diversity-pursuit')
    ax.set_xlabel('number of classes')
    ax.set_xticklabels(num_classes_array)
    ax.set_ylabel('precision')
    ax.set_title('feature selection precision')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(figs_dir, 'num_classes_scan_precision.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Variant B (fixed mu=0.5) for number of significant features figure
    mu_fixed = 0.5
    number_features_found = np.empty((3, len(num_classes_array), num_experiments))
    for i, c in enumerate(num_classes_array):
        p = int(num_features)
        n_train = int(3 * (math.log(p) ** 2))
        n = int(n_train / (1 - test_frac))
        eps = p ** (-beta)
        mu = mu_fixed
        for j in range(num_experiments):
            centroids = sample_centroids(num_classes=c, num_features=p, eps=eps, power=mu, non_nulls_location='fixed')
            X, y = sample_normal_clusters(centroids, n, sig)
            true_mask = (centroids != 0)
            number_features_found[0, i, j] = np.sum(true_mask)

            train_mask, test_mask = train_test_split_mask(len(X), test_frac)
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            cs_ova = CentroidSimilarityFeatureSelection(); cs_ova.fit(X_train, y_train, method='one_vs_all')
            number_features_found[1, i, j] = np.sum(cs_ova.get_mask())
            cs_dp = CentroidSimilarityFeatureSelection(); cs_dp.fit(X_train, y_train, method='diversity_pursuit')
            number_features_found[2, i, j] = np.sum(cs_dp.get_mask())

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks(np.arange(len(num_classes_array)))
    ax.bar(x=np.arange(len(num_classes_array)) - 0.1, height=np.mean(number_features_found[0], axis=1), width=0.1, label='true')
    ax.bar(x=np.arange(len(num_classes_array)) + 0.1, height=np.mean(number_features_found[1], axis=1), width=0.1, label='one-vs-all')
    ax.bar(x=np.arange(len(num_classes_array)) + 0.0, height=np.mean(number_features_found[2], axis=1), width=0.1, label='diversity-pursuit')
    ax.set_xlabel('number of classes')
    ax.set_xticklabels(num_classes_array)
    ax.set_ylabel('number of sign. features')
    ax.set_title('number of significant features')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(figs_dir, 'num_classes_scan_number_of_significant_features.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description='Run normally distributed clusters experiments (scriptified notebook)')
    ap.add_argument('--outdir', type=str, default=None, help='Directory to save figures (default: experiments/Figs)')
    ap.add_argument('--seed', type=int, default=None, help='Random seed')
    ap.add_argument('--quick', action='store_true', help='Quick run (smaller arrays, fewer iterations)')
    return ap.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    # Default figures directory next to this script
    script_dir = os.path.dirname(__file__)
    default_figs = os.path.join(script_dir, 'Figs')
    figs_dir = args.outdir or default_figs
    ensure_dir(figs_dir)

    if args.quick:
        feature_dim_scan(figs_dir,
                         num_experiments=5,
                         features_num_array=(100, 500, 1000),
                         num_classes=3,
                         test_frac=0.8,
                         sig=1.0,
                         beta=0.6)
        num_classes_scan(figs_dir,
                         num_experiments=5,
                         num_classes_array=(2, 3),
                         num_features=1000,
                         test_frac=0.8,
                         sig=1.0,
                         beta=0.6)
    else:
        feature_dim_scan(figs_dir)
        num_classes_scan(figs_dir)

    print('All figures saved to:', figs_dir)


if __name__ == '__main__':
    main()
