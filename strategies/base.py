from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Strategy(ABC):
    def __init__(self, n_assets: int):
        self.n_assets = n_assets

    def initial_portfolio(self) -> np.ndarray:
        return np.full(self.n_assets, 1.0 / self.n_assets)

    @abstractmethod
    def update(self, history: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_portfolio(self, history: np.ndarray) -> np.ndarray:
        if history.shape[0] == 0:
            b = self.initial_portfolio()
        else:
            b = self.update(history)
        self._validate(b)
        return b

    def _validate(self, b: np.ndarray) -> None:
        if b.shape != (self.n_assets,):
            raise ValueError(
                f"portfolio has shape {b.shape}, expected ({self.n_assets},)"
            )
        if np.any(b < -1e-9):
            raise ValueError(f"portfolio has negative entries: {b}")
        if not np.isclose(b.sum(), 1.0, atol=1e-6):
            raise ValueError(f"portfolio does not sum to 1: sum={b.sum()}")
