from __future__ import annotations

# Cross-asset-class universe, built to address two problems the DJIA
# universe can't: (1) correlation -- 28-30 DJIA constituents are all US
# large-caps, so every strategy is really trading one equity-beta factor
# under different rebalancing rules, not genuinely different bets; (2)
# survivorship bias of a different shape than djia_universe.py's -- picking
# "ETFs that look uncorrelated today" would silently select for the
# handful of ETFs that never closed, out of the thousands (mostly
# leveraged/thematic) that have. The fix for both: one large, continuously-
# traded ETF per major asset class, chosen for what it represents (the
# flagship/largest fund tracking that asset class) rather than for its
# correlation with anything else, and verified -- not assumed -- to have
# traded without a gap from inception to today (see INCEPTIONS below).
#
# The first version of this file also included IYR (REITs), EFA (developed
# ex-US equities) and EEM (emerging-market equities), on the assumption
# that different geographies/sectors would decorrelate from SPY. Checked
# against real daily returns 2007-04-11..2026-09-01 (yfinance), that
# assumption was wrong: SPY-EFA corr = 0.89, SPY-EEM = 0.82, SPY-IYR =
# 0.75 -- international and real-estate equities move with US equities
# almost as tightly as DJIA constituents move with each other, so they
# were dropped as redundant rather than kept for the sake of asset count.
# UUP (US dollar) was added in their place. Mean |correlation| across all
# pairs dropped from 0.403 (the original 7) to 0.247 (the 5 below) on the
# same window -- a real reduction, not an assumption.
#
# Each entry is (ticker, inception date, asset class / risk factor):
#   SPY  1993-01-29  US large-cap equities (S&P 500)
#   TLT  2002-07-30  US Treasuries, 20+ year (interest-rate duration)
#   GLD  2004-11-18  gold (safe-haven / inflation hedge)
#   DBC  2006-02-06  broad commodities (energy, agriculture, metals)
#   UUP  2007-03-01  US dollar index (currency, negatively correlated
#                    with almost everything else here: -0.07..-0.42)
# One ETF per factor, not several near-duplicates per asset class (e.g.
# no second Treasury-duration ETF alongside TLT) -- two assets dominated
# by the same factor would just reintroduce the correlation problem this
# universe exists to avoid.
#
# Inception dates and continuous trading confirmed directly against
# yfinance (first/last available bar per ticker), the same kind of check
# djia_universe.py does for UNAVAILABLE -- none of these five has a gap
# or a delisting, unlike several DJIA seats.
INCEPTIONS = [
    ("SPY", "1993-01-29"),
    ("TLT", "2002-07-30"),
    ("GLD", "2004-11-18"),
    ("DBC", "2006-02-06"),
    ("UUP", "2007-03-01"),
]


def segments(start: str, end: str) -> list[tuple[str, str, list[str]]]:
    # Mirrors djia_universe.segments's signature so run_etf.py can reuse
    # run_backtest_segmented exactly as run.py does -- here the universe
    # only ever grows (an ETF is added once it exists; none has closed),
    # so there's no removed-tickers case to handle, unlike a real index's
    # reconstitution history.
    tickers: list[str] = []
    result = []
    seg_start = start
    for ticker, inception in INCEPTIONS:
        if not (start < inception < end):
            if inception <= start:
                tickers.append(ticker)
            continue
        if tickers:
            result.append((seg_start, inception, list(tickers)))
        tickers.append(ticker)
        seg_start = inception
    result.append((seg_start, end, list(tickers)))
    return result
