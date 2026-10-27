import numpy as np
import pandas as pd
import yfinance as yf


def get_prices(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    # yfinance-backed for now; swap the body of this function when
    # Bloomberg access is available, everything else calls get_price_relatives.
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    prices = raw["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    return prices


def get_price_relatives(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    prices = get_prices(tickers, start, end, interval)
    return (prices.pct_change() + 1).dropna()


def benchmark_wealth(ticker: str, start: str, end: str, dates: pd.DatetimeIndex) -> np.ndarray:
    # Buy-and-hold of one index fund, compounded over exactly the dates a
    # universe's strategies trade on: each segment of a segmented universe
    # loses its first day to pct_change, and the benchmark skips the same
    # days so the comparison covers identical periods.
    x = get_price_relatives([ticker], start, end)[ticker].reindex(dates)
    assert not x.isna().any(), f"{ticker} has no price relative on {x.index[x.isna()][:3].tolist()}"
    return np.concatenate([[1.0], np.cumprod(x.values)])
