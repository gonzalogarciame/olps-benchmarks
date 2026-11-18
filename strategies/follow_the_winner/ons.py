from __future__ import annotations

import cvxpy as cp
import numpy as np

from strategies.base import Strategy


class ONS(Strategy):
    def __init__(self, n_assets: int, beta: float = 1.0, delta: float = 1.0 / 8):
        super().__init__(n_assets)
        self._beta = beta
        self._delta = delta

    def update(self, history: np.ndarray) -> np.ndarray:
        # a is A_t 
        a = np.eye(self.n_assets)
        # c is c_t
        c = np.zeros(self.n_assets)
        b = self.initial_portfolio()
        for x_t in history:
            # grad is and g_tau
            grad = x_t / (b @ x_t)
            a = a + np.outer(grad, grad)
            c = c + grad
            # Newton step, solving the system instead of inverting a
            y = self._delta * np.linalg.solve(a, (1 + 1 / self._beta) * c)
            b = self._project(y, a)
        return b

    def _project(self, y: np.ndarray, a: np.ndarray) -> np.ndarray:
        b = cp.Variable(self.n_assets)
        # distance defined by a
        objective = cp.Minimize(cp.quad_form(b - y, a))
        constraints = [b >= 0, cp.sum(b) == 1]
        cp.Problem(objective, constraints).solve()
        # the solver may return small negative weights
        return np.maximum(b.value, 0.0)
