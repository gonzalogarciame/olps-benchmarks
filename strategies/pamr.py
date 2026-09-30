from __future__ import annotations

import numpy as np

from strategies.base import Strategy
from strategies.simplex import project_to_simplex


class PAMR(Strategy):
    # Passive Aggressive Mean Reversion (Li, Hoi & Gopalkrishnan, 2012),
    # book Ch.9. loss_t = max(0, b.x_t - eps); tau_t = loss_t / ||x_t -
    # xbar_t*1||^2 (xbar_t = mean of x_t across assets, book Prop. 9.1);
    # b_{t+1} = project_to_simplex(b_t - tau_t*(x_t - xbar_t*1)).
    # eps = 0.5 is the book's own real-market default ("we choose eps =
    # 0.5 in the experiments", Sec. 13.3.2) -- eps >= 1 degrades PAMR
    # toward uniform CRP (same section), which eps=1 would (it also
    # happens to reproduce Table 9.1's illustrative two-asset example
    # exactly, but 0.5 is what the book actually recommends for real
    # data).
    def __init__(self, n_assets: int, eps: float = 0.5):
        super().__init__(n_assets)
        self._eps = eps

    def update(self, history: np.ndarray) -> np.ndarray:
        b = self.initial_portfolio()
        for x_t in history:
            x_bar = x_t.mean()
            loss = max(0.0, b @ x_t - self._eps)
            denom = np.sum((x_t - x_bar) ** 2)
            tau = loss / denom if denom > 0 else 0.0
            b = project_to_simplex(b - tau * (x_t - x_bar))
        return b
