import numpy as np
from scipy.stats import f as fdist

# Prefer package-local MultiTest; fall back to top-level if available
try:
    from .multitest import MultiTest  # packaged with centroid_similarity
except Exception:  # pragma: no cover - defensive import for repo usage
    from multitest import MultiTest


class CentroidSimilarity(object):
    """
    Classify based on most similar centroid.

    At training, we average the response of each feature over classes and
    store class centroids (averages). At prediction, we select the class
    with the maximum cosine similarity to the test sample.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._global_mean = np.mean(X, 0)
        self._global_std = np.std(X, 0)

        self._cls_mean = np.zeros((len(self.classes_), X.shape[1]))
        self._cls_std = np.zeros((len(self.classes_), X.shape[1]))
        self._cls_n = np.zeros((len(self.classes_), X.shape[1]))

        for i, _ in enumerate(self.classes_):
            X_cls = X[y == self.classes_[i]]
            self._cls_mean[i] = np.mean(X_cls, 0)
            self._cls_std[i] = np.std(X_cls, 0)
            self._cls_n[i] = len(X_cls)

        self.set_mask(np.ones_like(self._cls_mean))

    def set_mask(self, mask):
        means = self._cls_mean * mask
        self._mask = mask
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        eps = 1e-10
        # Use safe division to avoid warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = np.where(norms >= eps, means / norms, np.zeros_like(means))
        self._mat = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    def prob_func(self, response):
        # Numerically stable sigmoid to prevent overflow
        # Clip response to prevent exp overflow
        response_clipped = np.clip(response, -500, 500)
        exp_response = np.exp(response_clipped)
        return exp_response / (1 + exp_response)

    def get_centroids(self):
        return self._mat * self._mask

    def predict_log_proba(self, X):
        centroids = self.get_centroids()
        # Clip input values to prevent overflow in matrix multiplication
        # This is a safety measure - ideally inputs should be normalized
        X_clipped = np.clip(X, -1e10, 1e10)
        centroids_clipped = np.clip(centroids, -1e10, 1e10)
        # Use safe matrix multiplication - suppress all numerical warnings
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            response = X_clipped @ centroids_clipped.T
        # Handle any remaining inf/nan values
        response = np.nan_to_num(response, nan=0.0, posinf=500.0, neginf=-500.0)
        return self.prob_func(response)

    def predict(self, X):
        probs = self.predict_log_proba(X)
        return np.argmax(probs, 1)

    def eval_accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_mask_prec_recall(self, true_mask):
        mask = self.get_mask().astype(bool)
        true_mask = np.asarray(true_mask).astype(bool)
        tp = np.sum(mask * true_mask)
        fp = np.sum(mask * ~true_mask)
        fn = np.sum(~mask * true_mask)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return precision, recall

    def get_mask_f1(self, true_mask):
        precision, recall = self.get_mask_prec_recall(true_mask)
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def get_mask(self):
        return self._mask

    def get_mask_fdr(self, true_mask):
        mask = self.get_mask().astype(bool)
        true_mask = np.asarray(true_mask).astype(bool)
        fp = np.sum(mask * ~true_mask)
        tp = np.sum(mask * true_mask)
        return fp / (fp + tp) if (fp + tp) > 0 else 0.0


class CentroidSimilarityFeatureSelection(CentroidSimilarity):
    """
    Same as CentroidSimilarity, but masks features based on:
        - 'one_vs_all': two-sample test (one class vs rest)
        - 'diversity_pursuit': full one-way ANOVA across classes
    """

    def fit(self, X, y, method='one_vs_all'):
        super().fit(X, y)
        self._cls_response = np.zeros(len(self.classes_))
        mask = np.ones_like(self._cls_mean)

        for i, _ in enumerate(self.classes_):
            mask[i] = self.get_cls_mask(i, method=method)

        self.set_mask(mask)

    def get_pvals(self, cls_id, method='one_vs_all'):
        mu1 = self._cls_mean[cls_id]
        n1 = self._cls_n[cls_id]
        std1 = self._cls_std[cls_id]
        nG = self._cls_n.sum(0)
        stdG = self._global_std
        muG = self._global_mean

        assert method in ['one_vs_all', 'diversity_pursuit']
        if method == 'one_vs_all':
            pvals, _, _ = one_vs_all_ANOVA(n1, nG, mu1, muG, std1, stdG)
        else:
            pvals, _, _ = diversity_pursuit_ANOVA(self._cls_n, self._cls_mean, self._cls_std)
        return pvals

    def get_cls_mask(self, cls_id, method='one_vs_all'):
        pvals = self.get_pvals(cls_id, method=method)
        mt = MultiTest(pvals)
        hc, hct = mt.hc_star(gamma=.2)
        self._cls_response[cls_id] = hc
        mask = pvals < hct
        return mask


def diversity_pursuit_ANOVA(nn, mm, ss):
    """
    Vectorized F-test across all classes to find discriminating features.
    nn: class counts (k,)
    mm: per-class means (k,p)
    ss: per-class std (k,p)
    """
    muG = np.sum(mm * nn, 0) / np.sum(nn, 0)
    SSres = np.sum(nn * (mm - muG) ** 2, 0)
    SSfit = np.sum(nn * (ss ** 2), 0)

    dfn = len(nn) - 1
    dfd = np.sum(nn, 0) - len(nn)
    F = (SSres / dfn) / (SSfit / dfd)
    return fdist.sf(F, dfn, dfd), SSres, SSfit


def one_vs_all_ANOVA(n1, nG, mu1, muG, std1, stdG):
    n2 = nG - n1
    mu2 = (muG * nG - mu1 * n1) / (nG - n1)
    SSres = n1 * (mu1 - muG) ** 2 + n2 * (mu2 - muG) ** 2
    SStot = stdG ** 2 * nG
    SSfit = SStot - SSres

    F = (SSres / 1) / (SSfit / (nG - 2))
    return fdist.sf(F, 1, nG - 2), SSres, SSfit
