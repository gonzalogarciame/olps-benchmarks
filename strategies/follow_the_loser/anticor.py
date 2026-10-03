from __future__ import annotations

import numpy as np

from strategies.base import Strategy


class Anticor(Strategy):
    # Anticorrelation (Borodin, El-Yaniv & Gogan, 2004), book Sec. 5.2.
    # Compares two adjacent windows of size w on log price relatives (LX1:
    # the w periods before the most recent w; LX2: the most recent w) via
    # their cross-correlation matrix Mcor. When asset i outperformed asset
    # j in LX2 and they're positively cross-correlated, Anticor claims a
    # transfer from i to j sized by Mcor(i,j) minus i and j's own negative
    # autocorrelation, normalized so the claims from any one asset never
    # exceed its current weight -- this keeps b on the simplex without a
    # separate projection step. w is not stated by the book itself
    # (unlike PAMR/OLMAR's cited defaults). w=12 is tuned, not guessed:
    # swept on the 2007-2009 crisis window, confirmed on the independent
    # 2000-2026 window (paper, Sec. anticor-olmar2-tuning), where it
    # roughly doubles w=5's APY (18.71% vs 8.45%) and is the only value
    # tested that is both strong and significant on both windows. Tuned
    # for the DJIA universe specifically -- verified directly not to
    # transfer to the cross-asset ETF universe, which keeps w=5
    # (run_etf.py).
    def __init__(self, n_assets: int, w: int = 12):
        super().__init__(n_assets)
        self._w = w

    def update(self, history: np.ndarray) -> np.ndarray:
        b = self.initial_portfolio()
        n = len(history)
        for t in range(2 * self._w, n + 1):
            window1 = history[t - 2 * self._w : t - self._w]
            window2 = history[t - self._w : t]
            b = self._transfer(b, window1, window2)
        return b

    def _transfer(self, b: np.ndarray, window1: np.ndarray, window2: np.ndarray) -> np.ndarray:
        lx1 = np.log(window1)
        lx2 = np.log(window2)
        mu1 = lx1.mean(axis=0)
        mu2 = lx2.mean(axis=0)
        sigma1 = lx1.std(axis=0, ddof=1)
        sigma2 = lx2.std(axis=0, ddof=1)

        mcov = (lx1 - mu1).T @ (lx2 - mu2) / (self._w - 1)
        denom = np.outer(sigma1, sigma2)
        mcor = np.divide(mcov, denom, out=np.zeros_like(mcov), where=denom != 0)

        neg_auto = np.maximum(0.0, -np.diag(mcor))
        claim = mcor + neg_auto[:, None] + neg_auto[None, :]
        active = (mu2[:, None] >= mu2[None, :]) & (mcor > 0)
        np.fill_diagonal(active, False)
        claim = np.where(active, claim, 0.0)

        row_sums = claim.sum(axis=1)
        transfer = np.divide(
            claim * b[:, None], row_sums[:, None], out=np.zeros_like(claim), where=row_sums[:, None] != 0
        )
        return b - transfer.sum(axis=1) + transfer.sum(axis=0)
