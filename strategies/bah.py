from __future__ import annotations

import numpy as np

from strategies.base import Strategy


class BAH(Strategy):
    def __init__(self, n_assets: int, initial_weights: np.ndarray | None = None):
        super().__init__(n_assets)
        self._b1 = None if initial_weights is None else np.asarray(initial_weights, dtype=float)

    def initial_portfolio(self) -> np.ndarray:
        if self._b1 is not None:
            return self._b1
        return super().initial_portfolio()

    def update(self, history: np.ndarray) -> np.ndarray:
        b1 = self.initial_portfolio()
        cumulative_growth = np.prod(history, axis=0)
        drifted = b1 * cumulative_growth
        return drifted / drifted.sum()
