from __future__ import annotations

import numpy as np

from strategies.base import Strategy


class CRP(Strategy):
    def __init__(self, n_assets: int, b: np.ndarray | None = None):
        super().__init__(n_assets)
        self._b = np.full(n_assets, 1.0 / n_assets) if b is None else np.asarray(b, dtype=float)

    def initial_portfolio(self) -> np.ndarray:
        return self._b

    def update(self, history: np.ndarray) -> np.ndarray:
        # Unlike BAH, which lets weights drift with the market, CRP
        # actively rebalances back to the same fixed b every period.
        return self._b
