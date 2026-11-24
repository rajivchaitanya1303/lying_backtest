"""
GARCH(1,1) VaR with Student-t innovations.

Refit on each rolling training window; one-step-ahead conditional VaR
from the fitted t-distribution quantile.

Requires the `arch` package (Kevin Sheppard).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class GarchVaR:
    """
    GARCH(1,1) with Student-t innovations.

    Parameters
    ----------
    alpha : float
        Nominal tail probability.
    refit_every : int
        Re-estimate GARCH parameters every N test steps. Default 1 (each day).
        Set higher for speed if you're OK with slightly stale parameters.
    scale : float
        Multiply raw returns by this factor before fitting (arch prefers
        percentage returns for numerical stability). We use 100.0.
    """

    def __init__(self, alpha: float = 0.01, refit_every: int = 1,
                 scale: float = 100.0):
        self.alpha = float(alpha)
        self.refit_every = int(refit_every)
        self.scale = float(scale)

    def fit(self, train_returns: pd.Series) -> "GarchVaR":
        self._train = train_returns
        return self

    def forecast(self, test_returns: pd.Series,
                 train_returns: pd.Series) -> pd.Series:
        from arch import arch_model
        from scipy.stats import t as student_t

        # scale for numerical stability
        r_train = train_returns * self.scale
        r_test = test_returns * self.scale

        # running series we feed into GARCH; extended one day at a time
        # (this is the standard rolling-refit setup, ~expensive but honest)
        history = list(r_train.values)
        var_out = np.full(len(r_test), np.nan)

        model = None
        res = None
        steps_since_fit = self.refit_every  # force first fit
        for i, (idx, r_t) in enumerate(r_test.items()):
            if steps_since_fit >= self.refit_every:
                model = arch_model(pd.Series(history), mean="Zero",
                                   vol="Garch", p=1, q=1, dist="t")
                res = model.fit(disp="off", show_warning=False)
                steps_since_fit = 0

            fc = res.forecast(horizon=1, reindex=False)
            sigma_next = float(np.sqrt(fc.variance.values[-1, 0]))
            nu = float(res.params.get("nu", 8.0))
            # standardized-t quantile at alpha, then rescale
            q_std = student_t.ppf(self.alpha, df=nu)
            # arch's t is standardized (var=1); mean is zero (we set mean="Zero")
            var_scaled = q_std * sigma_next
            var_out[i] = var_scaled / self.scale

            history.append(r_t)
            steps_since_fit += 1

        return pd.Series(var_out, index=test_returns.index, name=self.name)

    @property
    def name(self) -> str:
        return "GARCH(1,1)-t"
