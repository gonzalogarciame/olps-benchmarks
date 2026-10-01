from __future__ import annotations

import numpy as np

from strategies.benchmarks.bah import BAH


class BestStock(BAH):
    # NOT a real online strategy: picking the best asset requires seeing
    # the entire price relative history in advance, which is only possible
    # in hindsight. It exists purely as an unrealizable upper-bound
    # benchmark (Li & Hoi, Section 3.2) to compare real strategies against,
    # not as something an online decision maker could ever execute.

    def __init__(self, n_assets: int, full_history: np.ndarray):
        cumulative_returns = np.prod(full_history, axis=0)
        best_asset = np.argmax(cumulative_returns)
        b1 = np.zeros(n_assets)
        b1[best_asset] = 1.0
        super().__init__(n_assets, initial_weights=b1)
