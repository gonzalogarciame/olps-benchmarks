from __future__ import annotations

import numpy as np

from strategies.pattern_matching.corn import CORN
from strategies.pattern_matching.corn_u import _CORNPool


class CORNK(_CORNPool):
    # CORN-K, book Algorithm 8.3: like CORN-U but over a 2-D grid of
    # W * P experts CORN(w, rho) for w = 1..W and rho in {0, 1/P, ...,
    # (P-1)/P}, and only the K experts with the highest current wealth
    # get weight (1/K each, book Eq. "top K"); the rest get 0. The 1/K
    # is uniform across the selected K, so it cancels the same way
    # CORN-U's 1/W does, leaving a wealth-weighted average over just
    # the top-K subset (see _CORNPool's docstring for why this class
    # keeps real state rather than replaying from scratch).
    #
    # W=10, P=5, K=3 are not book-cited defaults -- the book sweeps W
    # and rho separately (Sec. 13.3.1) but never reports a specific
    # (W, P, K) it recommends. 50 experts (W*P) rather than CORN-U's 10
    # is already five times the per-period cost; these are a tractable
    # starting grid, not a claim that they're optimal.
    def __init__(self, n_assets: int, W: int = 10, P: int = 5, K: int = 3):
        self._W = W
        self._P = P
        self._K = K
        rhos = [p / P for p in range(P)]
        experts = [CORN(n_assets, w=w, rho=rho) for w in range(1, W + 1) for rho in rhos]
        super().__init__(n_assets, experts)

    def _combine(self) -> np.ndarray:
        top_k = np.argsort(self._wealth)[-self._K :]
        wealth = self._wealth[top_k]
        total = wealth.sum()
        if total <= 0:
            return self.initial_portfolio()
        return (wealth @ self._last_b[top_k]) / total
