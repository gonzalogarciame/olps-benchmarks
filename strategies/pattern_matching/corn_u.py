from __future__ import annotations

import numpy as np

from strategies.base import Strategy
from strategies.pattern_matching.corn import CORN


class _CORNPool(Strategy):
    # Shared machinery for CORN-U (Algorithm 8.2) and CORN-K (Algorithm
    # 8.3), book Ch.8 -- both maintain a pool of CORN(w, rho) experts,
    # each tracking its own hypothetical cumulative wealth S_t(w, rho)
    # as if it had been trading alone the whole time, and combine their
    # *next* recommendations weighted by that wealth. Subclasses differ
    # only in which experts get nonzero weight in the combination
    # (_combine): all of them for CORN-U, only the top K for CORN-K.
    #
    # Unlike every other strategy here, this one is NOT stateless by
    # choice. The book's own Algorithm 8.2/8.3 pseudocode is already
    # written as an explicitly incremental process -- S_t(w, rho) is
    # defined as S_{t-1}(w, rho) times this period's realized return,
    # not as something recomputed from the full history each time.
    # Translating that literally into this project's usual "replay
    # everything from scratch on every update() call" contract (Section
    # ons.py's stateless design) doesn't just cost more here the way it
    # does for ONS -- it changes complexity class entirely: each of the
    # W (or W*P) sub-experts would need its own O(t) correlation scan
    # recomputed at every one of the engine's T outer calls, each of
    # which itself replays sub-expert history up to that point, giving
    # O(experts * T^3) instead of the O(experts * T^2) a single
    # incremental pass actually costs -- not slower, infeasible even to
    # benchmark. So this class keeps real state (each expert's running
    # wealth and most recent recommendation) between calls.
    #
    # That reintroduces exactly the risk ons.py's own comment flags for
    # why it stayed stateless: silently assuming update() is always
    # called once per period, in increasing order. Guarded against
    # directly, not assumed away: every call checks that `history` is
    # exactly one period longer than what was last processed, and falls
    # back to a full from-scratch replay (slow, but correct) whenever
    # it isn't -- a cold start, a skipped period, or reuse against a
    # different history. The fast, stateful path only ever runs when
    # the calling pattern actually matches what it requires.
    def __init__(self, n_assets: int, experts: list[CORN]):
        super().__init__(n_assets)
        self._experts = experts
        self._reset()

    def _reset(self) -> None:
        n = len(self._experts)
        self._wealth = np.ones(n)
        self._last_b = np.full((n, self.n_assets), 1.0 / self.n_assets)
        self._last_t = 0

    def _advance(self, history: np.ndarray) -> None:
        x_t = history[-1]
        for i in range(len(self._experts)):
            self._wealth[i] *= self._last_b[i] @ x_t
        for i, expert in enumerate(self._experts):
            self._last_b[i] = expert.update(history)
        self._last_t = len(history)

    def update(self, history: np.ndarray) -> np.ndarray:
        t = len(history)
        if t == 0:
            return self.initial_portfolio()
        if t != self._last_t + 1:
            self._reset()
            for tau in range(1, t + 1):
                self._advance(history[:tau])
        else:
            self._advance(history)
        return self._combine()

    def _combine(self) -> np.ndarray:
        raise NotImplementedError


class CORNU(_CORNPool):
    # CORN-U, book Algorithm 8.2: W experts CORN(w, rho) for w = 1..W,
    # fixed rho, combined by a uniform prior q(w) = 1/W weighted by each
    # expert's own wealth (book Eq. 8.2) -- which cancels the uniform
    # prior in the ratio, leaving a plain wealth-weighted average. The
    # book recommends this over a single CORN(w, rho) specifically
    # because single-expert performance is sensitive to w and the best
    # w is dataset-dependent (Sec. 8.4); W=10 here is a modest ceiling
    # chosen for tractability, not a book-cited value -- the book's own
    # sensitivity figures sweep W up to 30 (Sec. 13.3.1, Fig. 13.4).
    def __init__(self, n_assets: int, W: int = 10, rho: float = 0.1):
        self._W = W
        experts = [CORN(n_assets, w=w, rho=rho) for w in range(1, W + 1)]
        super().__init__(n_assets, experts)

    def _combine(self) -> np.ndarray:
        total = self._wealth.sum()
        if total <= 0:
            return self.initial_portfolio()
        return (self._wealth @ self._last_b) / total
