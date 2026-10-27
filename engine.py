from __future__ import annotations

from typing import Callable

import numpy as np

from strategies.base import Strategy


def run_portfolios(strategy: Strategy, price_relatives: np.ndarray) -> np.ndarray:
    n_periods = price_relatives.shape[0]
    b = np.empty((n_periods, price_relatives.shape[1]))
    for t in range(n_periods):
        history = price_relatives[:t]  # x_1, ..., x_{t-1}
        b[t] = strategy.get_portfolio(history)  # b_1 is uniform when history is empty
    return b


def wealth_from_portfolios(b: np.ndarray, price_relatives: np.ndarray) -> np.ndarray:
    # S_0 = 1, S_t = S_{t-1} * (b_t . x_t)
    return np.concatenate([[1.0], np.cumprod((b * price_relatives).sum(axis=1))])


def run_backtest(strategy: Strategy, price_relatives: np.ndarray) -> np.ndarray:
    return wealth_from_portfolios(run_portfolios(strategy, price_relatives), price_relatives)


def chain_segments(segment_wealths: list[np.ndarray]) -> np.ndarray:
    # Wealth carries across segments by rebasing each segment's S_0=1 series
    # onto the previous segment's final wealth; the shared boundary point is
    # dropped once, not duplicated.
    wealth = np.array([1.0])
    for segment_wealth in segment_wealths:
        wealth = np.concatenate([wealth[:-1], wealth[-1] * segment_wealth])
    return wealth


def run_backtest_segmented(
    strategy_factory: Callable[[int, np.ndarray], Strategy],
    segments: list[np.ndarray],
) -> np.ndarray:
    # For a universe that changes over time (e.g. DJIA reconstitution): a
    # fresh strategy instance per segment is correct, not a workaround --
    # asset columns mean different tickers before/after a reconstitution
    # even when the count doesn't change, so history has to reset at each
    # boundary the same way it already resets to empty at the start of
    # any single run_backtest call.
    return chain_segments(
        [run_backtest(strategy_factory(segment.shape[1], segment), segment) for segment in segments]
    )
