"""
Walk-forward VaR evaluation harness.

For each asset, slide a (train_size, test_size) window across the return
series with step `step_size`; refit each model on the training window,
generate a VaR series on the test window, store the concatenated
(date, model, ticker, VaR, return, violation) records to disk.

Nothing here is model-specific — models just need to expose:
    .fit(train_returns)                       -> self
    .forecast(test_returns, train_returns)    -> pd.Series aligned to test_returns
    .name                                     -> str
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd
import numpy as np


def walk_forward_windows(n: int, train_size: int, test_size: int,
                         step_size: int):
    """Yield (train_start, train_end, test_start, test_end) index tuples."""
    train_start = 0
    while train_start + train_size + test_size <= n:
        train_end = train_start + train_size
        test_end = train_end + test_size
        yield train_start, train_end, train_end, test_end
        train_start += step_size


def evaluate_model_on_ticker(model_factory, returns: pd.Series,
                             train_size: int = 1000, test_size: int = 250,
                             step_size: int = 250,
                             ticker: str = "?") -> pd.DataFrame:
    """
    Return a long-format DataFrame with columns
        [date, ticker, model, window_id, var, ret, violation]
    """
    rows = []
    for wid, (ts, te, vs, ve) in enumerate(
            walk_forward_windows(len(returns), train_size, test_size, step_size)):
        train = returns.iloc[ts:te]
        test = returns.iloc[vs:ve]
        model = model_factory()
        model.fit(train)
        var = model.forecast(test, train)
        df = pd.DataFrame({
            "date": test.index,
            "ticker": ticker,
            "model": model.name,
            "window_id": wid,
            "var": var.values,
            "ret": test.values,
        })
        df["violation"] = (df["ret"] < df["var"]).astype(int)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def evaluate_all(model_factories: dict, returns_by_ticker: dict,
                 train_size: int = 1000, test_size: int = 250,
                 step_size: int = 250,
                 out_path: str | Path | None = None) -> pd.DataFrame:
    """
    Run every (model, ticker) combination.

    Parameters
    ----------
    model_factories : dict[str, callable]
        {"HistSim": lambda: HistoricalSimulationVaR(...), ...}. We use
        callables so each window gets a *fresh* instance (important for LSTM).
    returns_by_ticker : dict[str, pd.Series]
    """
    frames = []
    for tkr, r in returns_by_ticker.items():
        for label, factory in model_factories.items():
            print(f"  {tkr} | {label} ...", flush=True)
            df = evaluate_model_on_ticker(
                factory, r, train_size=train_size,
                test_size=test_size, step_size=step_size, ticker=tkr,
            )
            frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_parquet(out_path, index=False)
    return all_df
