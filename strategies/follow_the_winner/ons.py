from __future__ import annotations

import cvxpy as cp
import numpy as np

from strategies.base import Strategy


class ONS(Strategy):
    # Online Newton Step (Agarwal, Hazan, Kale & Schapire, 2006), via
    # Follow the Regularized Leader (book Ch.4.4). beta is the FTRL
    # trade-off parameter; delta is the Newton step scale. Recomputed
    # from scratch on every call, like the other strategies here, rather
    # than carrying A/c as running state -- see paper/main.tex, Follow
    # the Winner, for why.
    def __init__(self, n_assets: int, beta: float = 1.0, delta: float = 1.0 / 8):
        super().__init__(n_assets)
        self._beta = beta
        self._delta = delta

    def update(self, history: np.ndarray) -> np.ndarray:
        a = np.eye(self.n_assets)
        c = np.zeros(self.n_assets)
        b = self.initial_portfolio()
        for x_t in history:
            grad = x_t / (b @ x_t)
            a = a + np.outer(grad, grad)
            c = c + grad
            y = self._delta * np.linalg.solve(a, (1 + 1 / self._beta) * c)
            b = self._project(y, a)
        return b

    def _project(self, y: np.ndarray, a: np.ndarray) -> np.ndarray:
        b = cp.Variable(self.n_assets)
        objective = cp.Minimize(cp.quad_form(b - y, a))
        constraints = [b >= 0, cp.sum(b) == 1]
        cp.Problem(objective, constraints).solve()
        # interior-point solvers routinely return tiny negative noise
        # (~1e-9 to 1e-20) on entries the constraint should hold at exactly
        # zero; harmless numerically, but strict enough to fail
        # Strategy._validate's -1e-9 tolerance on long/noisy backtests
        return np.maximum(b.value, 0.0)
