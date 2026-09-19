from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import pandas as pd

from data import get_price_relatives
from engine import run_backtest
from strategies.bah import BAH
from strategies.best_stock import BestStock
from strategies.crp import CRP
from strategies.eg import EG
from strategies.ons import ONS
from strategies.up import UP

# Dow Jones Industrial Average constituents as of 2023-01-01 -- the index's
# membership was unchanged from 2020-08-31 until 2024-02-26, so this is the
# correct point-in-time list, not today's DJIA (using today's list here would
# be the textbook survivorship-bias mistake this repo is trying to avoid).
# One of the 30, Walgreens Boots Alliance (WBA), is left out: it was taken
# private in 2025 and yfinance no longer serves any historical data for it at
# all, even for 2023 when it was still trading -- a live example of exactly
# the bias this list is trying to fix, not an oversight. See paper/main.tex.
TICKERS = [
    "AAPL", "MSFT", "JPM", "WMT", "V", "JNJ", "CSCO", "CVX", "KO", "CAT",
    "MRK", "PG", "UNH", "HD", "GS", "IBM", "AXP", "AMGN", "CRM", "DIS",
    "MCD", "BA", "MMM", "TRV", "HON", "NKE", "VZ", "INTC", "DOW",
]
START = "2023-01-01"
END = "2024-01-01"
PLOT_PATH = "wealth_curves.png"

# distinct dash patterns so nearly-overlapping wealth curves stay
# distinguishable even where color alone would not separate them
LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]


def main() -> None:
    price_relatives_df = get_price_relatives(TICKERS, START, END)
    price_relatives = price_relatives_df.values
    n_assets = price_relatives.shape[1]

    # wealth has one more point than price_relatives (S_0 = 1 before any
    # period), so prepend a date for that point too
    dates = price_relatives_df.index
    plot_dates = dates.insert(0, dates[0] - pd.Timedelta(days=1))

    strategies = {
        "BAH": BAH(n_assets),
        "CRP": CRP(n_assets),
        "BestStock": BestStock(n_assets, price_relatives),
        "UP": UP(n_assets),
        "EG": EG(n_assets),
        "ONS": ONS(n_assets),
    }

    print(f"{len(price_relatives)} periods, {n_assets} assets: {TICKERS}")

    fig, ax = plt.subplots()
    for (name, strategy), linestyle in zip(strategies.items(), itertools.cycle(LINESTYLES)):
        wealth = run_backtest(strategy, price_relatives)
        print(f"{name:10s} final wealth = {wealth[-1]:.4f}")
        # BestStock is a hindsight-only upper bound (see strategies/best_stock.py),
        # not a real strategy, so it's left off the graph -- printed above for
        # reference, but not plotted alongside strategies that make causal decisions
        if name != "BestStock":
            ax.plot(plot_dates, wealth, label=name, linestyle=linestyle)

    ax.set_xlabel("date")
    ax.set_ylabel("cumulative wealth")
    ax.set_title(f"OLPS strategies on DJIA constituents as of {START} (n={n_assets})")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(PLOT_PATH)
    print(f"saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
