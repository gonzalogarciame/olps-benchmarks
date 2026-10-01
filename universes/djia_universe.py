from __future__ import annotations

# Point-in-time DJIA constituents from 2000-01-01 onward, reconstructed from
# the index's known reconstitution history instead of a single fixed
# snapshot -- the fix for the survivorship bias problem documented in
# paper/main.tex (Sec. survivorship): a real backtest window has to track
# membership changes, not assume they don't happen. This table originally
# covered only 2007-2009; extended back to 2000 so the backtest can span a
# much longer, less deliberately-chosen range (dot-com bust, 2008 crisis,
# the 2009-2019 bull run, COVID, and the recent AI-driven run) instead of
# windows picked specifically to showcase one kind of market. Dates and
# tickers verified against Wikipedia's "List of changes to the Dow Jones
# Industrial Average" and, for the post-2012 changes, contemporary news/S&P
# Dow Jones Indices press releases directly (CNBC, Axios, press.spglobal.com).
#
# A few seats changed corporate name/ticker without a real Dow membership
# change (Philip Morris -> Altria kept ticker MO; AlliedSignal -> Honeywell
# kept the seat under HON after their 1999 merger). Those are *not* listed
# below as reconstitution events -- the seat's canonical ticker in this
# table is simply its ticker as of 2000-01-01, used continuously until an
# actual membership change occurs.
UNIVERSE_2000 = [
    "AA", "XOM", "MCD", "HON", "GE", "MRK", "AXP", "GM", "MSFT", "T",
    "HPQ", "MMM", "BA", "HD", "MO", "CAT", "INTC", "PG", "C", "IBM",
    "SBC", "KO", "IP", "UTX", "DD", "JNJ", "WMT", "EK", "JPM", "DIS",
]

# (effective date, tickers removed, tickers added)
RECONSTITUTIONS = [
    ("2004-04-08", ["T", "EK", "IP"], ["AIG", "PFE", "VZ"]),
    ("2008-02-19", ["MO", "HON"], ["BAC", "CVX"]),
    ("2008-09-22", ["AIG"], ["KFT"]),
    ("2009-06-08", ["GM", "C"], ["CSCO", "TRV"]),
    ("2012-09-24", ["KFT"], ["UNH"]),
    ("2013-09-23", ["AA", "BAC", "HPQ"], ["GS", "NKE", "V"]),
    ("2015-03-19", ["SBC"], ["AAPL"]),
    ("2017-09-01", ["DD"], ["DWDP"]),
    ("2018-06-26", ["GE"], ["WBA"]),
    ("2019-04-02", ["DWDP"], ["DOW"]),
    ("2020-08-31", ["XOM", "PFE", "UTX"], ["AMGN", "HON", "CRM"]),
    ("2024-02-26", ["WBA"], ["AMZN"]),
    ("2024-11-08", ["DOW", "INTC"], ["NVDA", "SHW"]),
    ("2026-06-29", ["VZ"], ["GOOGL"]),
]

# yfinance cannot serve historical prices for these -- confirmed directly
# against the live API, not assumed, each a distinct failure mode:
#   GM   -- pre-bankruptcy equity cancelled in the 2009 Chapter 11
#           reorganization (by the time it left the Dow, the index itself
#           listed it as "Motors Liquidation Company", not GM).
#   KFT  -- the original Kraft Foods Inc. has no single successor after
#           its 2012 split into Mondelez and Kraft Foods Group.
#   WBA  -- taken private in 2025; no historical data at all, not even for
#           the years it was actively trading (2018-2024).
#   SBC  -- still trades today, renamed AT&T Inc. (ticker T) after
#           acquiring the original AT&T Corp in 2005 -- but ticker T
#           already meant that different, unrelated-by-then company for
#           the first half of SBC's own Dow tenure (2000-2005), so
#           aliasing SBC to T would silently misattribute AT&T Corp's
#           returns to SBC's seat for that period. Treated as unavailable
#           for its whole tenure rather than risk that.
#   DWDP -- DowDuPont existed only 2017-2019 before splitting three ways
#           (Dow Inc, new DuPont, Corteva); no single successor carries
#           its trading history.
#   EK   -- Eastman Kodak's pre-2004 history is not served under its old
#           ticker at all, independent of (and years before) its later
#           2012 bankruptcy.
# Each is the same underlying problem already documented for WBA in the
# original 2023 snapshot: a data source tracking only currently-listed
# securities makes it structurally difficult to include anything that
# didn't survive to the present under its original ticker, even when nothing
# else about the company's disappearance was unusual or even negative (SBC
# is, today, one of the largest and most enduring companies in the index).
UNAVAILABLE = {"GM", "KFT", "WBA", "SBC", "DWDP", "EK"}

# Not survivorship-bias gaps: both companies are still trading, just
# renamed (United Technologies -> Raytheon Technologies, 2020), and
# yfinance's continuous price history for them lives under the new ticker.
TICKER_ALIASES = {"UTX": "RTX"}


def segments(start: str, end: str) -> list[tuple[str, str, list[str]]]:
    tickers = list(UNIVERSE_2000)
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
