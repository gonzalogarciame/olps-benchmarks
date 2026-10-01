from __future__ import annotations

import numpy as np

from strategies.base import Strategy
from strategies.simplex import project_to_simplex


class OLMAR(Strategy):
    # Online Moving Average Reversion (Li & Hoi, 2012), book Ch.11, the
    # simple-moving-average variant (MAR-1). x_hat predicts the next price
    # relative from a window of w trailing relatives (book Eq. 11.1):
    # x_hat = (1/w) * (1 + 1/x_t + 1/(x_t*x_{t-1}) + ... ), reusing only
    # price relatives since that's what `history` gives us, not raw
    # prices. b_{t+1} is then a passive-aggressive step toward eps on the
    # predicted relative (book Prop. 11.1), projected onto the simplex.
    # w=5, eps=10 are the book's own real-market reference settings (Sec.
    # 13.3.4, Figs. 13.9-13.10: "fixed w=5" while varying eps, "fixed
    # eps=10" while varying w/alpha) -- not Table 11.2's eps=2, which is
    # illustrative for that section's small two-asset toy markets only.
    def __init__(self, n_assets: int, eps: float = 10.0, w: int = 5):
        super().__init__(n_assets)
        self._eps = eps
        self._w = w

    def update(self, history: np.ndarray) -> np.ndarray:
        b = self.initial_portfolio()
        for t in range(len(history)):
            x_hat = self._predict(history[: t + 1])
            x_bar = x_hat.mean()
            denom = np.sum((x_hat - x_bar) ** 2)
            gap = self._eps - b @ x_hat
            lam = max(0.0, gap / denom) if denom > 0 else 0.0
            b = project_to_simplex(b + lam * (x_hat - x_bar))
        return b

    def _predict(self, relatives: np.ndarray) -> np.ndarray:
        tail = relatives[-(self._w - 1):][::-1]  # x_t, x_{t-1}, ..., most recent first
        x_hat = np.ones(self.n_assets)
        cumprod = np.ones(self.n_assets)
        for x in tail:
            cumprod = cumprod * x
            x_hat += 1.0 / cumprod
        return x_hat / self._w
