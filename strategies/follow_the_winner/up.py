from __future__ import annotations

import numpy as np

from strategies.base import Strategy


class UP(Strategy):
    def __init__(self, n_assets: int, n_samples: int = 1000, seed: int = 0):
        super().__init__(n_assets)
        rng = np.random.default_rng(seed)
        # random portfolios, all equally likely
        self._samples = rng.dirichlet(np.ones(n_assets), size=n_samples)

    def update(self, history: np.ndarray) -> np.ndarray:
        growth = self._samples @ history.T  # b^(k) · x_tau, per sample per period
        wealth = growth.prod(axis=1)  # S_t(b^(k))
        return (wealth @ self._samples) / wealth.sum()
