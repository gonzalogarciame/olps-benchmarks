from __future__ import annotations

import cvxpy as cp
import numpy as np

from strategies.base import Strategy


class CORN(Strategy):
    # CORrelation-driven Nonparametric learning (Li, Hoi, Sahoo & Zhao,
    # 2011), book Ch.8, the single-expert CORN(w, rho) of Algorithm 8.1 --
    # not the CORN-U/CORN-K multi-expert combinations of Sec. 8.3, which
    # average many of these across a range of w (or w and rho). Unlike
    # Anticor/PAMR/OLMAR, which measure reversion asset-by-asset, CORN
    # looks for days whose w-period lead-up resembled today's: history is
    # scanned for every window x_{i-w}^{i-1} whose Pearson correlation
    # with the latest window x_{t-w+1}^t is >= rho (book Sec 8.2); the
    # "correlation-similar" set C_t is the day *after* each such window,
    # x_i, since that's what the latest window is expected to resemble
    # next. b_{t+1} is then the best constant rebalanced portfolio over
    # just that similar-day set (book Eq. 8.1):
    #   b_{t+1} = argmax_b prod_{i in C_t} (b . x_i)
    # reformulated as maximizing sum_{i in C_t} log(b . x_i), the
    # equivalent log-optimal utility the book itself uses (Ch.6 Eq. 6.1,
    # "log-optimal" utility U_L). C_t empty (including t <= w) falls back
    # to uniform, per the book's own Algorithm 8.1.
    #
    # w=5, rho=0.1 are not a single book-cited "real-market default" the
    # way PAMR/OLMAR's parameters are -- the book's own sensitivity study
    # (Sec 13.3.1, Figs 13.3-13.4) instead recommends CORN-U specifically
    # *because* single-expert CORN is sensitive to w and that sensitivity
    # is dataset-dependent, with no one best value. w=5 and rho=0.1 are
    # simply the two fixed values those two sensitivity figures hold
    # constant while sweeping the other parameter -- the closest thing to
    # a book reference point for a single expert, not a recommendation to
    # use only one. CORN-U/CORN-K (Algorithms 8.2/8.3), which combine many
    # (w, rho) experts by wealth-weighted average -- the same idea as
    # strategies/follow_the_winner/up.py's Monte Carlo ensemble, but over
    # CORN experts instead of random CRPs -- are not implemented here.
    def __init__(self, n_assets: int, w: int = 5, rho: float = 0.1):
        super().__init__(n_assets)
        self._w = w
        self._rho = rho

    def update(self, history: np.ndarray) -> np.ndarray:
        t = len(history)
        if t <= self._w:
            return self.initial_portfolio()

        latest_window = history[t - self._w : t].ravel()
        similar_days = [
            history[i - 1]
            for i in range(self._w + 1, t + 1)
            if self._correlation(history[i - self._w - 1 : i - 1].ravel(), latest_window) >= self._rho
        ]
        if not similar_days:
            return self.initial_portfolio()

        return self._bcrp(np.array(similar_days))

    @staticmethod
    def _correlation(a: np.ndarray, b: np.ndarray) -> float:
        std_a, std_b = a.std(), b.std()
        if std_a == 0 or std_b == 0:
            # book Sec 8.2: a zero-volatility window has no defined
            # linear relationship, so the pair is treated as uncorrelated
            # rather than left undefined.
            return 0.0
        return float(np.mean((a - a.mean()) * (b - b.mean())) / (std_a * std_b))

    def _bcrp(self, similar_days: np.ndarray) -> np.ndarray:
        b = cp.Variable(self.n_assets)
        objective = cp.Maximize(cp.sum(cp.log(similar_days @ b)))
        constraints = [b >= 0, cp.sum(b) == 1]
        cp.Problem(objective, constraints).solve()
        # same interior-point noise ONS's projection clips (strategies/
        # follow_the_winner/ons.py) -- tiny negative entries on what
        # should be exact zeros.
        return np.maximum(b.value, 0.0)
