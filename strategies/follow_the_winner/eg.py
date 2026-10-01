from __future__ import annotations

import numpy as np

from strategies.base import Strategy


class EG(Strategy):
    # Exponential Gradient (Helmbold, Schapire, Singer & Warmuth, 1998).
    # b_{t+1,i} = b_{t,i} * exp(eta * x_{t,i} / (b_t . x_t)), renormalized.
    # eta = 0.05 is the value the book cites as empirically best.
    def __init__(self, n_assets: int, eta: float = 0.05):
        super().__init__(n_assets)
        self._eta = eta

    def update(self, history: np.ndarray) -> np.ndarray:
        b = self.initial_portfolio()
        for x_t in history:
            b = b * np.exp(self._eta * x_t / (b @ x_t))
            b = b / b.sum()
        return b
