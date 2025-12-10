#!/usr/bin/env python3
"""
Run centroid similarity feature-selection experiments within the CentSim repo.

This script is adapted from the original
transfer-learning-with-feature-selection project to work directly with the
`centroid_similarity` package in this repository. Figures are saved next to
this script under `Figs/`.

Note: This script expects numpy, scipy, pandas, matplotlib, seaborn, and tqdm
to be available in your environment. The core classifier depends on SciPy for
the F distribution.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Ensure repository root is on sys.path when running from experiments/
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Higher-criticism threshold util: prefer package-local version
try:
    from centroid_similarity.multitest import MultiTest
except Exception:  # fallback to repo-local copy if needed
    from multitest import MultiTest

# Import the library from this repo
from centroid_similarity import CentroidSimilarity, CentroidSimilarityFeatureSelection


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


def pairwise_distances_euclidean(A: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances for rows of A.

    Equivalent to sklearn.metrics.pairwise_distances(A), but avoids dependency.
    """
    # d(i,j) = sqrt(||ai||^2 + ||aj||^2 - 2 ai·aj)
    sq_norms = np.sum(A * A, axis=1, keepdims=True)
    d2 = sq_norms + sq_norms.T - 2.0 * (A @ A.T)
    # Numerical safety: clamp very small negatives to zero
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)


# Figures directory next to this script
SCRIPT_DIR = os.path.dirname(__file__)
# Default figures directory; may be overridden by --outdir
DEFAULT_FIGS_DIR = os.path.join(SCRIPT_DIR, 'Figs')
os.makedirs(DEFAULT_FIGS_DIR, exist_ok=True)


def print_method_descriptions():
    print("\n" + "=" * 80)
    print("CLASSIFICATION METHODS DESCRIPTION")
    print("=" * 80)
    print("\n1. 'naive':")
    print("   - Basic CentroidSimilarity classifier without feature selection")
    print("   - Uses all features to compute class centroids")
    print("   - Classifies based on cosine similarity to class centroids")
    print("   - No statistical testing or feature filtering applied")

    print("\n2. 'OneVsAll' (OvA):")
    print("   - CentroidSimilarityFeatureSelection with 'one_vs_all' method")
    print("   - Uses two-sample ANOVA test (one class vs. all others) for each feature")
    print("   - Applies Higher Criticism (HC) test to select significant features")
    print("   - Only features that significantly discriminate one class from all others are used")

    print("\n3. 'DivPursuit' (Diversity Pursuit):")
    print("   - CentroidSimilarityFeatureSelection with 'diversity_pursuit' method")
    print("   - Uses full one-way ANOVA test across all classes for each feature")
    print("   - Applies Higher Criticism (HC) test to select significant features")
    print("   - Selects features that show significant variation across all classes")
    print("   - More powerful for multi-class problems than one-vs-all approach")

    print("\n4. 'CS_oracle_mu' (Centroid Similarity Oracle with True Means):")
    print("   - Oracle classifier that uses the true class centroids (ground truth)")
    print("   - Represents the theoretical upper bound on performance")
    print("   - Uses the actual generating centroids instead of estimated ones")
    print("   - Helps evaluate how close other methods are to optimal performance")

    print("\n" + "=" * 80 + "\n")


def eval_accuracy(clf, X, y) -> float:
    y_pred = clf.predict(X)
    return float(np.mean(y_pred == y))


def get_FDR(csf: CentroidSimilarityFeatureSelection, true_mask: np.ndarray) -> float:
    FP = np.sum(csf._mask.any(0)[~true_mask.any(0)])
    TP = np.sum(csf._mask.any(0)[true_mask.any(0)])
    return float(FP / (FP + TP)) if (FP + TP) > 0 else 0.0


def atomic_exp(p: int, n: int, c: int, sig: float, mu: float, eps: float, test_frac: float):
    """Run a single experiment with given parameters.

    Returns a dict of metrics and parameters for aggregation.
    """
    # Generate synthetic data
    centroids = sample_centroids(
        num_classes=c,
        num_features=p,
        eps=eps,
        power=mu,
        non_nulls_location='fixed',
    )
    true_mask = centroids != 0

    X, y = sample_normal_clusters(centroids, n, sig)

    # Split into train and test
    train_split_mask = np.random.rand(len(X)) > test_frac
    X_train = X[train_split_mask]
    y_train = y[train_split_mask]
    X_test = X[~train_split_mask]
    y_test = y[~train_split_mask]

    # Distance statistics of true centroids
    dist_mat = pairwise_distances_euclidean(centroids)
    # mean/std over upper triangle (i<j)
    iu, ju = np.triu_indices(len(dist_mat), k=1)
    upper_vals = dist_mat[iu, ju]
    delta_mean = float(np.mean(upper_vals)) if upper_vals.size else 0.0
    delta_min = float(np.min(dist_mat + 1e9 * np.eye(len(dist_mat))))
    delta_std = float(np.std(upper_vals)) if upper_vals.size else 0.0
    delta_th = float(mu * np.sqrt(eps * p / 2))

    # Method 1: Naive (no feature selection)
    cs = CentroidSimilarity()
    cs.fit(X_train, y_train)
    acc_naive = eval_accuracy(cs, X_test, y_test)

    # Method 2: One-vs-All feature selection
    csf_OvA = CentroidSimilarityFeatureSelection()
    csf_OvA.fit(X_train, y_train, method='one_vs_all')
    pvals_OvA = csf_OvA.get_pvals(cls_id=0, method='one_vs_all')
    acc_OvA = eval_accuracy(csf_OvA, X_test, y_test)

    # Method 3: Diversity Pursuit feature selection
    csf_DP = CentroidSimilarityFeatureSelection()
    csf_DP.fit(X_train, y_train, method='diversity_pursuit')
    pvals_DP = csf_DP.get_pvals(cls_id=0, method='diversity_pursuit')
    acc_DP = eval_accuracy(csf_DP, X_test, y_test)

    # Method 4: Oracle with true feature mask
    cs_oracle_t = CentroidSimilarityFeatureSelection()
    cs_oracle_t.fit(X_train, y_train)
    cs_oracle_t.set_mask(true_mask)
    acc_oracle_t = eval_accuracy(cs_oracle_t, X_test, y_test)

    # Method 5: Oracle with true centroids
    cs_oracle_mu = CentroidSimilarity()
    cs_oracle_mu.fit(X_train, y_train)  # benign, sets structure
    cs_oracle_mu._cls_mean = centroids
    norms = np.linalg.norm(cs_oracle_mu._cls_mean, axis=1, keepdims=True)
    eps_norm = 1e-10
    cs_oracle_mu._mat = np.where(
        norms >= eps_norm,
        cs_oracle_mu._cls_mean / norms,
        np.zeros_like(cs_oracle_mu._cls_mean),
    )
    cs_oracle_mu._mat = np.nan_to_num(cs_oracle_mu._mat, nan=0.0, posinf=0.0, neginf=0.0)
    acc_oracle_mu = eval_accuracy(cs_oracle_mu, X_test, y_test)

    # Higher Criticism statistics
    hc_OvA = MultiTest(pvals_OvA).hc()[0]
    hc_DP = MultiTest(pvals_DP).hc()[0]

    return dict({
        'naive': acc_naive,
        'OvA': acc_OvA,
        'DivPursuit': acc_DP,
        'acc_oracle_thr': acc_oracle_t,
        'cs_oracle_mu': acc_oracle_mu,
        'hc_OvA': float(hc_OvA),
        'hc_DP': float(hc_DP),
        'fdr_OvA': get_FDR(csf_OvA, true_mask),
        'fdr_DP': get_FDR(csf_DP, true_mask),
        'eps': eps,
        'n': n, 'p': p, 'mu': mu, 'c': c,
        'delta_mean': delta_mean,
        'delta_min': delta_min,
        'delta_std': delta_std,
        'delta_th': delta_th,
        'mask_sum': int(np.sum(true_mask)),
    })


def rho(beta: float) -> float:
    return float(((1 - np.sqrt(1 - beta)) ** 2) * (beta >= .75) + (beta - 1/2) * (beta < .75))


def parse_args():
    ap = argparse.ArgumentParser(description='Run Centroid Similarity FS experiments')
    ap.add_argument('--p', type=int, default=10000, help='Number of features')
    ap.add_argument('--n', type=int, default=None, help='Number of samples (default: 2*log(p)^2)')
    ap.add_argument('--c', type=int, default=10, help='Number of classes')
    ap.add_argument('--sig', type=float, default=1.0, help='Noise standard deviation')
    ap.add_argument('--test-frac', type=float, default=0.2, help='Test split fraction')
    ap.add_argument('--beta-start', type=float, default=0.5, help='Beta grid start')
    ap.add_argument('--beta-stop', type=float, default=0.9, help='Beta grid stop')
    ap.add_argument('--beta-steps', type=int, default=5, help='Number of beta grid points')
    ap.add_argument('--r-start', type=float, default=0.01, help='r grid start')
    ap.add_argument('--r-stop', type=float, default=0.3, help='r grid stop')
    ap.add_argument('--r-steps', type=int, default=7, help='Number of r grid points')
    ap.add_argument('--nMonte', type=int, default=10, help='Monte Carlo iterations per grid point')
    ap.add_argument('--seed', type=int, default=None, help='Random seed')
    ap.add_argument('--outdir', type=str, default=None, help='Directory to save figures (default: experiments/Figs)')
    ap.add_argument('--save-csv', type=str, default=None, help='Path to save a CSV of results (optional)')
    ap.add_argument('--quick', action='store_true', help='Quick run with small grids and iterations')
    return ap.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    print_method_descriptions()

    # Experiment parameters
    p = args.p
    if args.quick:
        p = min(p, 1000)
    n = int(2 * np.log(p) ** 2) if args.n is None else args.n
    c = 5 if args.quick else args.c
    sig = float(args.sig)
    test_frac = float(args.test_frac)

    # Parameter ranges
    beta_steps = 2 if args.quick else args.beta_steps
    r_steps = 2 if args.quick else args.r_steps
    bb = np.linspace(args.beta_start, args.beta_stop, beta_steps)
    rr = np.linspace(args.r_start, args.r_stop, r_steps)
    nMonte = 1 if args.quick else args.nMonte

    # Output directory
    figs_dir = args.outdir or DEFAULT_FIGS_DIR
    os.makedirs(figs_dir, exist_ok=True)

    print("\nRunning experiments with:")
    print(f"  - Features (p): {p}")
    print(f"  - Samples (n): {n}")
    print(f"  - Classes (c): {c}")
    print(f"  - Monte Carlo iterations: {nMonte}")
    print(f"  - Beta range: {bb[0]:.2f} to {bb[-1]:.2f}")
    print(f"  - r range: {rr[0]:.3f} to {rr[-1]:.3f}")
    print(f"\nTotal experiments: {nMonte * len(bb) * len(rr)}\n")

    # Run experiments
    res = []
    for itr in tqdm(range(nMonte), desc="Monte Carlo iterations"):
        for r_val in rr:
            for beta_val in bb:
                mu = np.sqrt(r_val * np.log(p))
                eps = p ** (-beta_val)
                res_atom = atomic_exp(p, n, c, sig, mu, eps, test_frac)
                res_atom['itr'] = itr
                res.append(res_atom)

    df = pd.DataFrame(res)

    print(f"\nExperiments completed! Results shape: {df.shape}")
    print("\nMean accuracies:")
    methods = ['naive', 'OvA', 'DivPursuit', 'cs_oracle_mu']
    for method in methods:
        print(f"  {method}: {df[method].mean():.4f}")

    # Ensure figures directory exists
    os.makedirs(figs_dir, exist_ok=True)

    # Figure 1: Bar plot of mean accuracies
    print("\nGenerating Figure 1: Mean accuracy comparison...")
    mean_accuracies = [df[val].mean() for val in methods]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=methods, y=mean_accuracies)
    plt.ylabel("Mean Accuracy", fontsize=12)
    plt.xlabel("Method", fontsize=12)
    plt.title("Mean Accuracy of Each Classification Method", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'mean_accuracy_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to: {fig_path}")

    # Figure 2: Heatmaps for each method
    print("\nGenerating heatmaps for each method...")
    mat = np.zeros((len(bb), len(rr), len(methods)))
    for k, val in enumerate(methods):
        print(f"  Processing {val}...")
        for i, beta_val in enumerate(bb):
            for j, r_val in enumerate(rr):
                mu = np.sqrt(r_val * np.log(p))
                eps = p ** (-beta_val)
                dfc = df[np.abs(df.eps - eps) + np.abs(df.mu - mu) < 1e-10]
                if len(dfc) > 0:
                    mat[i, j, k] = np.mean(dfc[val])

        plt.figure(figsize=(10, 8))
        g = sns.heatmap(mat[:, ::-1, k].T, annot=False, fmt='.3f', cmap='viridis')
        plt.title(f"{val} (accuracy)", fontsize=14)
        g.set_xticklabels([f'{b:.2f}' for b in bb])
        g.set_xlabel(r'$\beta$', fontsize=12)
        g.set_ylabel(r'$r$', fontsize=12)
        g.set_yticklabels([f'{r:.3f}' for r in rr[::-1]])
        plt.tight_layout()
        fig_path = os.path.join(figs_dir, f'{val}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved to: {fig_path}")

    # Figure 3: Phase transition curve
    print("\nGenerating phase transition curve...")
    bb_plot = np.linspace(0.5, 1, 57)
    rho_b = [rho(beta_val) for beta_val in bb_plot]
    plt.figure(figsize=(10, 6))
    plt.plot(bb_plot, rho_b, linewidth=2)
    plt.xlabel(r'$\beta$', fontsize=12)
    plt.ylabel(r'$\rho(\beta)$', fontsize=12)
    plt.title('Phase Transition Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'phase_transition_curve.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to: {fig_path}")

    # Optionally save results CSV
    if args.save_csv:
        save_dir = os.path.dirname(args.save_csv)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        df.to_csv(args.save_csv, index=False)
        print(f"\nSaved results CSV to: {args.save_csv}")

    print("\n" + "=" * 80)
    print("All experiments completed and figures saved!")
    print(f"Figures saved to: {figs_dir}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
