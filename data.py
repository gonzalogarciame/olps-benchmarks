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
