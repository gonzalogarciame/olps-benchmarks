from __future__ import annotations

import numpy as np
from scipy import stats

# book defaults (Ch.2.3): one trading year = 252 periods, Rf = 4%/year
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.04


def periodic_returns(wealth: np.ndarray) -> np.ndarray:
    return wealth[1:] / wealth[:-1] - 1


def apy(wealth: np.ndarray, trading_days_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    years = (len(wealth) - 1) / trading_days_per_year
    return wealth[-1] ** (1 / years) - 1


def volatility(wealth: np.ndarray, trading_days_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return periodic_returns(wealth).std(ddof=1) * np.sqrt(trading_days_per_year)


def sharpe_ratio(
    wealth: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    return (apy(wealth, trading_days_per_year) - risk_free_rate) / volatility(
        wealth, trading_days_per_year
    )


def drawdown(wealth: np.ndarray) -> np.ndarray:
    # running peak includes t itself, so a new high correctly shows DD(t) = 0
    return np.maximum.accumulate(wealth) - wealth


def max_drawdown(wealth: np.ndarray) -> float:
    return drawdown(wealth).max()


def calmar_ratio(wealth: np.ndarray, trading_days_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return apy(wealth, trading_days_per_year) / max_drawdown(wealth)


def t_test(wealth: np.ndarray) -> tuple[float, float]:
    # null hypothesis: the strategy is not profitable (population mean = 0)
    pl = periodic_returns(wealth)
    standard_error = pl.std(ddof=1) / np.sqrt(len(pl))
    t_stat = pl.mean() / standard_error
    p_value = stats.t.sf(t_stat, df=len(pl) - 1)
    return t_stat, p_value
