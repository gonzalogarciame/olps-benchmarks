from __future__ import annotations

from strategies.follow_the_loser.pamr import PAMR


class PAMR2(PAMR):
    # PAMR-2 (book Eq. 9.4/9.8): quadratic slack penalty -- softens tau by
    # growing the denominator instead of capping it outright like PAMR-1,
    # same motivation (bound how much a single noisy price relative can
    # move the portfolio). C=1.0, same reasoning as PAMR1's choice (the
    # book's own worked-example value, not one chosen by sweeping this
    # project's own data).
    def __init__(self, n_assets: int, eps: float = 0.5, C: float = 1.0):
        super().__init__(n_assets, eps)
        self._C = C

    def _tau(self, loss: float, denom: float) -> float:
        return loss / (denom + 1.0 / (2 * self._C))
