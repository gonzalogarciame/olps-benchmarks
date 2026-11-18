from __future__ import annotations

import numpy as np

from strategies.base import Strategy


class EG(Strategy):
    def __init__(self, n_assets: int, eta: float = 0.05):
        super().__init__(n_assets)
        self._eta = eta

    def update(self, history: np.ndarray) -> np.ndarray:
        b = self.initial_portfolio()
        for x_t in history:
            # each day changes based on the day before
            b = b * np.exp(self._eta * x_t / (b @ x_t))
            b = b / b.sum()
        return b
