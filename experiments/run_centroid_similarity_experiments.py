#!/usr/bin/env python3
"""
Script to run centroid similarity feature selection experiments.
Converts the notebook CentroidSimilarityFeatureSelection_old.ipynb into a runnable script.
All figures are saved to the Figs/ directory.

Requirements:
    Install dependencies with: pip install -r requirements.txt
    See requirements.txt in the parent directory for all required packages.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from multitest import MultiTest

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from CentroidSimilarity import CentroidSimilarity, CentroidSimilarityFeatureSelection
from utils import synthetic_data_gen

# Create Figs directory if it doesn't exist
FIGS_DIR = os.path.join(os.path.dirname(__file__), 'Figs')
os.makedirs(FIGS_DIR, exist_ok=True)


def print_method_descriptions():
    """Print descriptions of all classification methods used."""
    print("\n" + "="*80)
    print("CLASSIFICATION METHODS DESCRIPTION")
    print("="*80)
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
    
    print("\n" + "="*80 + "\n")


def eval_accuracy(clf, X, y):
    """Evaluate classifier accuracy."""
    y_pred = clf.predict(X)
    return np.mean(y_pred == y)


def get_FDR(csf, true_mask):
    """Calculate False Discovery Rate for feature selection."""
    FP = np.sum(csf._mask.any(0)[~true_mask.any(0)])
    TP = np.sum(csf._mask.any(0)[true_mask.any(0)])
    return FP / (FP + TP) if (FP + TP) > 0 else 0.0


def atomic_exp(p, n, c, sig, mu, eps, test_frac):
    """
    Run a single experiment with given parameters.
    
    Args:
    -----
    p         : number of features
    n         : number of samples
    c         : number of classes
    sig       : noise intensity
    mu        : signal intensity
    eps       : signal sparsity
    test_frac : fraction of data to be held as test set
    
    Returns:
    --------
    Dictionary with experiment results
    """
    # Generate synthetic data
    centroids = synthetic_data_gen.sample_centroids(
        num_classes=c, 
        num_features=p, 
        eps=eps, 
        power=mu, 
        non_nulls_location='fixed'
    )
    true_mask = centroids != 0
    
    X, y = synthetic_data_gen.sample_normal_clusters(centroids, n, sig)
    
    # Split into train and test
    train_split_mask = np.random.rand(len(X)) > test_frac
    X_train = X[train_split_mask]
    y_train = y[train_split_mask]
    X_test = X[~train_split_mask]
    y_test = y[~train_split_mask]
    
    # Calculate distance statistics
    dist_mat = pairwise_distances(centroids)
    mat_idx = [(i, j) for i in range(len(dist_mat)) for j in range(len(dist_mat)) if i < j]
    delta_mean = np.mean([dist_mat[p] for p in mat_idx])
    delta_min = np.min(dist_mat + 1e9 * np.eye(len(dist_mat)))
    delta_std = np.std([dist_mat[p] for p in mat_idx])
    delta_th = mu * np.sqrt(eps * p / 2)
    
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
    cs_oracle_mu.fit(X_train, y_train)  # This is benign, just sets up structure
    cs_oracle_mu._cls_mean = centroids
    norms = np.linalg.norm(cs_oracle_mu._cls_mean, axis=1, keepdims=True)
    eps_norm = 1e-10
    cs_oracle_mu._mat = np.where(
        norms >= eps_norm, 
        cs_oracle_mu._cls_mean / norms, 
        np.zeros_like(cs_oracle_mu._cls_mean)
    )
    cs_oracle_mu._mat = np.nan_to_num(cs_oracle_mu._mat, nan=0.0, posinf=0.0, neginf=0.0)
    acc_oracle_mu = eval_accuracy(cs_oracle_mu, X_test, y_test)
    
    # Calculate Higher Criticism statistics
    hc_OvA = MultiTest(pvals_OvA).hc()[0]
    hc_DP = MultiTest(pvals_DP).hc()[0]
    
    return dict({
        'naive': acc_naive,
        'OvA': acc_OvA,
        'DivPersuit': acc_DP,
        'acc_oracle_thr': acc_oracle_t,
        'cs_oracle_mu': acc_oracle_mu,
        'hc_OvA': hc_OvA,
        'hc_DP': hc_DP,
        'fdr_OvA': get_FDR(csf_OvA, true_mask),
        'fdr_DP': get_FDR(csf_DP, true_mask),
        'eps': eps,
        'n': n, 'p': p, 'mu': mu, 'c': c,
        'delta_mean': delta_mean,
        'delta_min': delta_min,
        'delta_std': delta_std,
        'delta_th': delta_th,
        'mask_sum': np.sum(true_mask),
    })


def rho(beta):
    """Phase transition curve function."""
    return ((1 - np.sqrt(1 - beta)) ** 2) * (beta >= .75) + (beta - 1/2) * (beta < .75)


def main():
    """Main function to run experiments and generate figures."""
    print_method_descriptions()
    
    # Experiment parameters
    p = 10000
    n = int(2 * np.log(p) ** 2)
    c = 10
    sig = 1
    r = .05 * c
    beta = .7
    test_frac = .2
    
    # Parameter ranges for grid search
    bb = np.linspace(0.5, 0.9, 5)
    rr = np.linspace(0.01, 0.3, 7)
    nMonte = 10
    
    print(f"\nRunning experiments with:")
    print(f"  - Features (p): {p}")
    print(f"  - Samples (n): {n}")
    print(f"  - Classes (c): {c}")
    print(f"  - Monte Carlo iterations: {nMonte}")
    print(f"  - Beta range: {bb[0]:.2f} to {bb[-1]:.2f}")
    print(f"  - r range: {rr[0]:.3f} to {rr[-1]:.3f}")
    print(f"\nTotal experiments: {nMonte * len(bb) * len(rr)}")
    print("\nStarting experiments...\n")
    
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
    print(f"\nMean accuracies:")
    methods = ['naive', 'OvA', 'DivPersuit', 'cs_oracle_mu']
    for method in methods:
        print(f"  {method}: {df[method].mean():.4f}")
    
    # Create figures directory
    os.makedirs(FIGS_DIR, exist_ok=True)
    
    # Figure 1: Bar plot of mean accuracies
    print("\nGenerating Figure 1: Mean accuracy comparison...")
    mean_accuracies = [df[val].mean() for val in methods]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=methods, y=mean_accuracies)
    plt.ylabel("Mean Accuracy", fontsize=12)
    plt.xlabel("Method", fontsize=12)
    plt.title("Mean Accuracy of Each Classification Method", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(FIGS_DIR, 'mean_accuracy_comparison.png')
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
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        g = sns.heatmap(mat[:, ::-1, k].T, annot=False, fmt='.3f', cmap='viridis')
        plt.title(f"{val} (accuracy)", fontsize=14)
        g.set_xticklabels([f'{b:.2f}' for b in bb])
        g.set_xlabel(r'$\beta$', fontsize=12)
        g.set_ylabel(r'$r$', fontsize=12)
        g.set_yticklabels([f'{r:.3f}' for r in rr[::-1]])
        plt.tight_layout()
        fig_path = os.path.join(FIGS_DIR, f'{val}.png')
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
    fig_path = os.path.join(FIGS_DIR, 'phase_transition_curve.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to: {fig_path}")
    
    print("\n" + "="*80)
    print("All experiments completed and figures saved!")
    print(f"Figures saved to: {FIGS_DIR}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

