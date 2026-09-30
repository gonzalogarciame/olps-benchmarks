from __future__ import annotations

# Point-in-time DJIA constituents from 2007-06-01 through 2009-12-31,
# reconstructed from the index's known reconstitution history instead of
# a single fixed snapshot -- the fix for the survivorship bias problem
# documented in paper/main.tex (Sec. survivorship): a real backtest
# window has to track membership changes, not assume they don't happen.
# Dates and tickers verified against Wikipedia's "List of changes to the
# Dow Jones Industrial Average" and contemporary news coverage (Reuters/
# NBC/Deseret News reporting on the 2008-2009 changes specifically).
UNIVERSE_2007 = [
    "MMM", "KO", "JPM", "AA", "DD", "MCD", "AXP", "XOM", "MRK", "AIG",
    "GE", "MSFT", "T", "GM", "PFE", "MO", "HPQ", "PG", "BA", "HD",
    "UTX", "CAT", "INTC", "VZ", "HON", "IBM", "WMT", "C", "JNJ", "DIS",
]

# (effective date, tickers removed, tickers added)
RECONSTITUTIONS = [
    ("2008-02-19", ["MO", "HON"], ["BAC", "CVX"]),
    ("2008-09-22", ["AIG"], ["KFT"]),
    ("2009-06-08", ["GM", "C"], ["CSCO", "TRV"]),
]

# yfinance cannot serve historical prices for these -- confirmed directly
# against the live API, not assumed. GM: its pre-bankruptcy equity was
# cancelled in the 2009 Chapter 11 reorganization (by the time it left
# the Dow, the index itself listed it as "Motors Liquidation Company",
# not GM); KFT: the original Kraft Foods Inc. has no single successor
# after its 2012 split into Mondelez and Kraft Foods Group. Same failure
# mode already documented for WBA in the 2023 snapshot (run.py) -- the
# data source's blind spot toward securities that didn't survive to the
# present, not a bug to route around.
UNAVAILABLE = {"GM", "KFT"}

# Not a survivorship-bias gap: United Technologies is still trading, just
# renamed to Raytheon Technologies in 2020, and yfinance's continuous
# price history for it now lives under "RTX" instead of "UTX".
TICKER_ALIASES = {"UTX": "RTX"}


def segments(start: str, end: str) -> list[tuple[str, str, list[str]]]:
    tickers = list(UNIVERSE_2007)
    result = []
    seg_start = start
    for date, removed, added in RECONSTITUTIONS:
        if not (start < date < end):
            continue
        result.append((seg_start, date, [t for t in tickers if t not in UNAVAILABLE]))
        tickers = [t for t in tickers if t not in removed] + added
        seg_start = date
    result.append((seg_start, end, [t for t in tickers if t not in UNAVAILABLE]))
    return result


def fetch_symbol(ticker: str) -> str:
    return TICKER_ALIASES.get(ticker, ticker)
