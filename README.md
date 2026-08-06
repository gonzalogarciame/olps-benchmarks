# olps-benchmarks

Online Portfolio Selection (OLPS) algorithms and benchmarks, implemented
following *Online Portfolio Selection: Principles and Algorithms* (Li & Hoi).

## Structure

- `data.py` — downloads adjusted close prices (via `yfinance`) and converts
  them into price relatives `x[t,i] = price[t,i] / price[t-1,i]`.
- `strategies/base.py` — `Strategy`, the abstract base class all algorithms
  implement (Algorithm A.1). Strategies are stateless: `update(history)`
  is a pure function of the price-relative history up to `t-1` and returns
  the portfolio `b_t`.
- `strategies/bah.py` — Buy-and-Hold (Chapter 3.1): invests once at `t=1`
  and never rebalances, so its weights drift with relative asset
  performance.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Example

```python
from data import get_price_relatives
from strategies.bah import BAH

price_relatives = get_price_relatives(
    tickers=["AAPL", "MSFT"], start="2023-01-01", end="2024-01-01"
)

strategy = BAH(n_assets=price_relatives.shape[1])
b_t = strategy.get_portfolio(price_relatives.values[:10])
```

## Status

Early stage — data pipeline and BAH strategy in place. More OLPS algorithms
(CRP, UP, EG, ONS, ...) to follow.
