from __future__ import annotations

import numpy as np

from strategies.base import Strategy


class UP(Strategy):
    # Universal Portfolio (Cover, 1991): b_{t+1} is the wealth-weighted
    # average, under a uniform prior over the simplex, of every constant
    # rebalanced portfolio b. The book's integral over Delta_m is
    # approximated here by Monte Carlo: n_samples portfolios drawn
    # uniformly from the simplex stand in for the continuous prior.
    def __init__(self, n_assets: int, n_samples: int = 1000, seed: int = 0):
        super().__init__(n_assets)
        rng = np.random.default_rng(seed)
        self._samples = rng.dirichlet(np.ones(n_assets), size=n_samples)

    def update(self, history: np.ndarray) -> np.ndarray:
        growth = self._samples @ history.T  # b^(k) . x_tau, per sample per period
        wealth = growth.prod(axis=1)  # S_t(b^(k))
        return (wealth @ self._samples) / wealth.sum()
