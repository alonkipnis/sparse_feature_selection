import numpy as np


class MultiTest:
    """
    Minimal Higher-Criticism utilities compatible with this repo's usage.

    Provides:
      - hc(gamma=0.2) -> (hc_stat, threshold)
      - hc_star(gamma=0.2) -> (hc_stat, threshold)

    threshold corresponds to the p-value at which the HC statistic is maximized.
    """

    def __init__(self, pvals):
        pv = np.asarray(pvals, dtype=float)
        pv = pv[np.isfinite(pv)]
        # clamp for numerical stability
        self.pvals = np.clip(pv, 1e-300, 1.0 - 1e-16)

    def _hc_core(self, gamma: float = 0.2):
        p = self.pvals
        n = p.size
        if n == 0:
            return 0.0, 1.0

        ps = np.sort(p)
        idx = np.arange(1, n + 1)

        # Consider k in [1, floor(n*(1-gamma))], omit extreme tail
        upper = int(np.floor(n * (1.0 - float(gamma))))
        upper = max(1, min(n, upper))

        k = idx[:upper]
        p_k = ps[:upper]

        denom = np.sqrt(p_k * (1.0 - p_k))
        # avoid division by zero
        valid = denom > 0
        hc_vals = np.full_like(p_k, -np.inf, dtype=float)
        hc_vals[valid] = np.sqrt(n) * (k[valid] / n - p_k[valid]) / denom[valid]

        max_i = int(np.argmax(hc_vals)) if hc_vals.size else 0
        hc_stat = float(hc_vals[max_i]) if hc_vals.size else 0.0
        thr = float(p_k[max_i]) if p_k.size else 1.0
        return hc_stat, thr

    def hc(self, gamma: float = 0.2):
        return self._hc_core(gamma)

    def hc_star(self, gamma: float = 0.2):
        return self._hc_core(gamma)

