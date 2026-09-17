from __future__ import annotations

import matplotlib.pyplot as plt

from data import get_price_relatives
from engine import run_backtest
from strategies.bah import BAH
from strategies.best_stock import BestStock
from strategies.crp import CRP
from strategies.up import UP

TICKERS = ["AAPL", "MSFT", "GOOG"]
START = "2023-01-01"
END = "2024-01-01"
PLOT_PATH = "wealth_curves.png"


def main() -> None:
    price_relatives_df = get_price_relatives(TICKERS, START, END)
    price_relatives = price_relatives_df.values
    n_assets = price_relatives.shape[1]

    strategies = {
        "BAH": BAH(n_assets),
        "CRP": CRP(n_assets),
        "BestStock": BestStock(n_assets, price_relatives),
        "UP": UP(n_assets),
    }

    print(f"{len(price_relatives)} periods, {n_assets} assets: {TICKERS}")

    fig, ax = plt.subplots()
    for name, strategy in strategies.items():
        wealth = run_backtest(strategy, price_relatives)
        print(f"{name:10s} final wealth = {wealth[-1]:.4f}")
        ax.plot(wealth, label=name)

    ax.set_xlabel("period")
    ax.set_ylabel("cumulative wealth")
    ax.set_title(f"OLPS strategies on {', '.join(TICKERS)}")
    ax.legend()
    fig.savefig(PLOT_PATH)
    print(f"saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
