from __future__ import annotations

import numpy as np

from strategies.base import Strategy


def run_backtest(strategy: Strategy, price_relatives: np.ndarray) -> np.ndarray:
    n_periods = price_relatives.shape[0]
    wealth = np.empty(n_periods + 1)
    wealth[0] = 1.0  # S_0 = 1

    for t in range(n_periods):
        history = price_relatives[:t]  # x_1, ..., x_{t-1}
        b_t = strategy.get_portfolio(history)  # b_1 is uniform when history is empty
        x_t = price_relatives[t]
        s_t = b_t @ x_t
        wealth[t + 1] = wealth[t] * s_t

    return wealth
