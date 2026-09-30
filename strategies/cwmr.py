from __future__ import annotations

import numpy as np

from strategies.base import Strategy
from strategies.simplex import project_to_simplex


class CWMR(Strategy):
    # Confidence Weighted Mean Reversion (Li, Hoi, Zhao & Gopalkrishnan,
    # 2013), book Ch.10, the CWMR-Var variant (Prop. 10.1). b is modeled
    # as N(mu, Sigma) with Sigma diagonal (sigma2, one variance per asset:
    # lower variance = higher confidence in that mu_i). Each period solves
    # a scalar Lagrange multiplier lam from a quadratic (book Eq. B.12,
    # coefficients a/b/c below) trading off staying close to (mu, Sigma)
    # against keeping Pr[mu.x_t <= eps] above confidence phi. phi = 1.64
    # (~90% confidence, book: "does not affect performance much") and
    # eps = 0.5 (book: "our empirical setting eps=0.5") are the book's
    # own real-market defaults, Sec. 13.3.3.
    def __init__(self, n_assets: int, phi: float = 1.64, eps: float = 0.5):
        super().__init__(n_assets)
        self._phi = phi
        self._eps = eps

    def update(self, history: np.ndarray) -> np.ndarray:
        mu = self.initial_portfolio()
        sigma2 = np.full(self.n_assets, 1.0 / self.n_assets**2)
        for x_t in history:
            m = mu @ x_t
            v = sigma2 @ x_t**2
            w = sigma2 @ x_t
            x_bar = w / sigma2.sum()

            a = 2 * self._phi * v**2 - 2 * self._phi * x_bar * v * w
            b = 2 * self._phi * self._eps * v - 2 * self._phi * v * m + v - x_bar * w
            c = self._eps - m - self._phi * v
            lam = self._solve_lambda(a, b, c)

            mu = mu - lam * sigma2 * (x_t - x_bar)
            sigma2 = 1.0 / (1.0 / sigma2 + 2 * lam * self._phi * x_t**2)

            mu = project_to_simplex(mu)
            sigma2 = sigma2 / (self.n_assets * sigma2.sum())
        return mu

    @staticmethod
    def _solve_lambda(a: float, b: float, c: float) -> float:
        if a != 0:
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                return 0.0
            sqrt_disc = np.sqrt(discriminant)
            roots = [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]
            return max(max(roots), 0.0)
        if b != 0:
            return max(-c / b, 0.0)
        return 0.0
