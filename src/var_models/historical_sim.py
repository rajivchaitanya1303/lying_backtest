"""
Historical Simulation VaR — rolling empirical quantile.
Zero parameters, zero learning. Textbook baseline.

Sign convention (project-wide): VaR is the *negative* tail quantile of
returns. A violation is `returns < var_forecast`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class HistoricalSimulationVaR:
    """
    Rolling window empirical quantile VaR estimator.

    Parameters
    ----------
    window : int
        Length of rolling look-back window (Basel default = 250 trading days).
    alpha : float
        Nominal tail probability, e.g. 0.01 for 99% VaR.
    """

    def __init__(self, window: int = 250, alpha: float = 0.01):
        self.window = int(window)
        self.alpha = float(alpha)

    # -- required by walk-forward harness --------------------------------
    def fit(self, train_returns: pd.Series) -> "HistoricalSimulationVaR":
        # nothing to fit; we operate on the rolling window at forecast time
        self._train = train_returns
        return self

    def forecast(self, test_returns: pd.Series,
                 train_returns: pd.Series) -> pd.Series:
        """
        For each date t in test_returns, estimate VaR from the trailing
        `window` returns ending at t-1 (no look-ahead).
        """
        combined = pd.concat([train_returns, test_returns])
        # rolling quantile at each date; forecast for t uses r_{t-window..t-1}
        var = (combined.shift(1)
                       .rolling(self.window, min_periods=self.window)
                       .quantile(self.alpha))
        return var.reindex(test_returns.index)

    @property
    def name(self) -> str:
        return f"HistSim(w={self.window})"
