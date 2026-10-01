from __future__ import annotations

import numpy as np

from strategies.base import Strategy
from strategies.simplex import project_to_simplex


class OLMAR2(Strategy):
    # Online Moving Average Reversion (Li & Hoi, 2012), book Ch.11, the
    # exponential-moving-average variant (MAR-2). Predicts the next price
    # relative recursively from an exponentially-weighted average of ALL
    # past prices rather than a fixed trailing window (book Eq. 11.2):
    # x_hat_{t+1} = alpha*1 + (1-alpha)*x_hat_t/x_t, x_hat_1 = 1. Same
    # passive-aggressive update/projection as the SMA variant
    # (strategies/olmar.py, book Prop. 11.1), only the prediction step
    # differs. eps=10 matches the SMA variant's real-market default (Sec.
    # 13.3.4); alpha is this variant's own parameter in place of w, with
    # no book-stated default value (only that Fig. 13.10 tests a range) --
    # 0.5 is our own interpretive choice, the midpoint of the book's
    # stated valid range (0,1).
    def __init__(self, n_assets: int, eps: float = 10.0, alpha: float = 0.5):
        super().__init__(n_assets)
        self._eps = eps
        self._alpha = alpha

    def update(self, history: np.ndarray) -> np.ndarray:
        b = self.initial_portfolio()
        x_hat = np.ones(self.n_assets)  # x_hat_1 = 1, book Algorithm 11.1 init
        for x_t in history:
            x_hat = self._alpha + (1 - self._alpha) * x_hat / x_t
            x_bar = x_hat.mean()
            denom = np.sum((x_hat - x_bar) ** 2)
            gap = self._eps - b @ x_hat
            lam = max(0.0, gap / denom) if denom > 0 else 0.0
            b = project_to_simplex(b + lam * (x_hat - x_bar))
        return b
