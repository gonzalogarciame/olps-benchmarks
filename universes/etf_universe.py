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
# Each entry is (ticker, inception date, asset class / risk factor):
#   SPY  1993-01-29  US large-cap equities (S&P 500)
#   IYR  2000-06-19  US real estate (REITs)
#   EFA  2001-08-27  developed-market ex-US equities (MSCI EAFE)
#   TLT  2002-07-30  US Treasuries, 20+ year (interest-rate duration)
#   EEM  2003-04-14  emerging-market equities (MSCI EM)
#   GLD  2004-11-18  gold (safe-haven / inflation hedge)
#   DBC  2006-02-06  broad commodities (energy, agriculture, metals)
# Picked one per factor rather than several near-duplicates per asset
# class (e.g. no second Treasury-duration or investment-grade-credit
# ETF alongside TLT) -- two assets dominated by the same factor would
# just reintroduce the correlation problem this universe exists to avoid.
#
# Inception dates and continuous trading confirmed directly against
# yfinance (first/last available bar per ticker), the same kind of check
# djia_universe.py does for UNAVAILABLE -- none of these seven has a gap
# or a delisting, unlike several DJIA seats.
INCEPTIONS = [
    ("SPY", "1993-01-29"),
    ("IYR", "2000-06-19"),
    ("EFA", "2001-08-27"),
    ("TLT", "2002-07-30"),
    ("EEM", "2003-04-14"),
    ("GLD", "2004-11-18"),
    ("DBC", "2006-02-06"),
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
