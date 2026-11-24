"""
Data pipeline: download daily prices via yfinance, compute log returns,
cache to CSV. Idempotent — re-running skips downloads already cached.

Usage
-----
    from src.data.download import load_returns
    r = load_returns("SPY")               # pandas.Series of log returns
    r = load_returns("SPY", start="1993-01-29", end="2024-12-31")

Files land in data/cache/{ticker}.csv.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


DEFAULT_START = "1993-01-29"   # SPY inception
DEFAULT_END = "2024-12-31"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _cache_path(ticker: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker}.csv"


def download_prices(ticker: str, start: str = DEFAULT_START,
                    end: str = DEFAULT_END, force: bool = False) -> pd.DataFrame:
    """Download adjusted daily prices from Yahoo, cache to CSV."""
    import yfinance as yf   # local import so the module is importable without yfinance

    path = _cache_path(ticker)
    if path.exists() and not force:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                     progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    # yfinance sometimes returns a MultiIndex column for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "close"})
    df.to_csv(path)
    return df


def load_returns(ticker: str, start: str = DEFAULT_START,
                 end: str = DEFAULT_END, force: bool = False) -> pd.Series:
    """
    Return log returns as a Series indexed by date.
    Convention: r_t = log(P_t / P_{t-1}), fraction (not %).
    """
    df = download_prices(ticker, start=start, end=end, force=force)
    close = df["close"].astype(float)
    r = np.log(close / close.shift(1)).dropna()
    r.name = ticker
    return r


def load_panel(tickers: list[str], start: str = DEFAULT_START,
               end: str = DEFAULT_END, force: bool = False) -> pd.DataFrame:
    """Load a wide panel of log returns, one column per ticker."""
    cols = {t: load_returns(t, start=start, end=end, force=force)
            for t in tickers}
    return pd.concat(cols, axis=1).dropna(how="all")


if __name__ == "__main__":
    r = load_returns("SPY")
    print(f"SPY log returns: {len(r):d} obs, "
          f"{r.index.min().date()} -> {r.index.max().date()}, "
          f"mean={r.mean():.5f}, std={r.std():.5f}")
