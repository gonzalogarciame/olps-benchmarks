import pandas as pd
import yfinance as yf


def get_price_relatives(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
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

    price_relatives = prices.pct_change() + 1
    price_relatives = price_relatives.dropna()

    return price_relatives
