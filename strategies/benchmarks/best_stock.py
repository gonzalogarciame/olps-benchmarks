from __future__ import annotations

import numpy as np

from strategies.benchmarks.bah import BAH


class BestStock(BAH):
    def __init__(self, n_assets: int, full_history: np.ndarray):
        # total growth of each asset's price over all the days
        cumulative_returns = np.prod(full_history, axis=0)
        best_asset = np.argmax(cumulative_returns)
        b1 = np.zeros(n_assets)
        b1[best_asset] = 1.0
        super().__init__(n_assets, initial_weights=b1)
