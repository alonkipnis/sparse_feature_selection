import numpy as np
from scipy.stats import f as fdist
from multitest import MultiTest


class CentroidSimilarity(object):
    """
    Classify based on most similar centroid. 
    
    At training, we average the response of each feature over
    classes. We store the class centroids (averages).
    
    At prediction, we give the highest probability to the class
    that is most similar to the test sample (in Eucleadian distance 
    or cosine similarity). 
    
    """
    def __init__(self):
        pass
    
    def fit(self, X, y):
        """
        
        Args:
        :X:  training data as a matrix n X p of n samples 
            with p features each
        :y:  labels. For multi-label, simply pass the same x value
            with different labels. 
        
        """
        self.classes_ = np.unique(y)
        self._global_mean = np.mean(X, 0)
        self._global_std = np.std(X, 0)
        
        self._cls_mean = np.zeros((len(self.classes_),X.shape[1]))
        self._cls_std = np.zeros((len(self.classes_),X.shape[1]))
        self._cls_n = np.zeros((len(self.classes_),X.shape[1]))
        
        for i,_ in enumerate(self.classes_):
            X_cls = X[y == self.classes_[i]]
            self._cls_mean[i] = np.mean(X_cls, 0)
            self._cls_std[i] = np.std(X_cls, 0)
            self._cls_n[i] = len(X_cls)
        
        self.set_mask(np.ones_like(self._cls_mean))
        
    def set_mask(self, mask):
        means = self._cls_mean * mask
        self._mask = mask
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        # Avoid division by zero: if norm is zero, keep zeros
        # Use a small epsilon to prevent numerical issues
        eps = 1e-10
        # Normalize only where norm is non-zero, otherwise keep zeros
        self._mat = np.where(norms >= eps, means / norms, np.zeros_like(means))
        # Ensure no NaN or inf values from numerical issues
        self._mat = np.nan_to_num(self._mat, nan=0.0, posinf=0.0, neginf=0.0)
        # since we normalize the matrix, max inner product 
        # is equivalent to max cosine similarity
        
    def prob_func(self, response):
        return np.exp(response) / (1 + np.exp(response))
    
    def get_centroids(self):
        return self._mat * self._mask
        
    def predict_log_proba(self, X):
        response = X @ self.get_centroids().T  # inner products
        return self.prob_func(response)
    
    def predict(self, X):
        probs = self.predict_log_proba(X)
        return np.argmax(probs, 1)  # max inner product

    def eval_accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


    def get_mask_prec_recall(self, true_mask):
        mask = self.get_mask()
        # Convert to boolean arrays for bitwise operations
        mask = mask.astype(bool)
        true_mask = np.asarray(true_mask).astype(bool)
        tp = np.sum(mask * true_mask)
        fp = np.sum(mask * ~true_mask)
        fn = np.sum(~mask * true_mask)
        tn = np.sum(~mask * ~true_mask)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return precision, recall
    
    def get_mask_f1(self, true_mask):
        precision, recall = self.get_mask_prec_recall(true_mask)
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def get_mask(self):
        return self._mask

    def get_mask_fdr(self, true_mask):
        mask = self.get_mask()
        # Convert to boolean arrays for bitwise operations
        mask = mask.astype(bool)
        true_mask = np.asarray(true_mask).astype(bool)
        fp = np.sum(mask * ~true_mask)
        tp = np.sum(mask * true_mask)
        return fp / (fp + tp) if (fp + tp) > 0 else 0.0

class CentroidSimilarityFeatureSelection(CentroidSimilarity):
    """
    Same as CentroidSimilarity, but now we mask
    some of the features based on two methods:
        'one_vs_all':  two-sample test
        'diversity_pursuit' ('all vs all'): full one-way ANOVA
    
    """
    
    def fit(self, X, y, method='one_vs_all'):
        
        super().fit(X, y)
        self._cls_response = np.zeros(len(self.classes_))
        mask = np.ones_like(self._cls_mean)
        
        for i, cls in enumerate(self.classes_):
            mask[i] = self.get_cls_mask(i, method=method)
            
        self.set_mask(mask)
        
    
    def get_pvals(self, cls_id, method='one_vs_all'):
        """
        compute P-values associated with each feature
        for the given class
        """
        
        mu1 = self._cls_mean[cls_id]
        n1 = self._cls_n[cls_id]
        std1 = self._cls_std[cls_id]
        nG = self._cls_n.sum(0)
        stdG = self._global_std
        muG = self._global_mean

        assert(method in ['one_vs_all', 'diversity_pursuit'])
        if method == 'one_vs_all' :
            pvals,_,_ = one_vs_all_ANOVA(n1, nG, mu1, muG, std1, stdG)
        if method == 'diversity_pursuit':
            pvals,_,_ = diversity_pursuit_ANOVA(self._cls_n,
                                                self._cls_mean,
                                                self._cls_std)
        return pvals

    
    def get_cls_mask(self, cls_id, method='one_vs_all'):
        """
        compute class feature mask
        """        
        pvals = self.get_pvals(cls_id, method=method)
        
        mt = MultiTest(pvals)
        hc, hct = mt.hc_star(gamma=.2)
        self._cls_response[cls_id] = hc
        mask = pvals < hct
        
        return mask
        
def diversity_pursuit_ANOVA(nn, mm, ss):
    """
    F-test for discoverying discriminating features
    
    The test is vectorized along the last dimention where
    different entires corresponds to different features
    
    Args:
    -----
    :nn:  vector indicating the number of elements in each class
    :mm:  matrix of means; the (i,j) entry is the mean response of
          class i in feature j
    :ss:  matrix of standard errors; the (i,j) entry is the standard
          error of class i in feature j
    
    """
    muG = np.sum(mm * nn, 0) / np.sum(nn, 0) # global mean
    SSres = np.sum(nn * (mm - muG) ** 2, 0)
    SSfit = np.sum(nn * (ss ** 2), 0)
    #SSerr = SStot - SSfit

    dfn = len(nn) - 1
    dfd = np.sum(nn, 0) - len(nn)

    F = ( SSres / dfn ) / ( SSfit / dfd )
    return fdist.sf(F, dfn, dfd), SSres, SSfit
        
        
def one_vs_all_ANOVA(n1, nG, mu1, muG, std1, stdG):
    n2 = nG - n1
    mu2 = (muG * nG - mu1 * n1) / (nG - n1)
    SSres = n1 * (mu1 - muG) ** 2 + n2 * (mu2 - muG) ** 2
    SStot = stdG ** 2 * nG
    SSfit = SStot - SSres 

    F = ( SSres / 1 ) / ( SSfit / (nG - 2) )
    return fdist.sf(F, 1, nG - 2), SSres, SSfit        
        