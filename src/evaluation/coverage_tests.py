"""
Coverage backtests for VaR forecasts.

Standard tests used by the conditional-coverage evaluation layer:
    * Kupiec (1995) POF (Proportion-of-Failures)
    * Christoffersen (1998) Independence + Conditional Coverage
    * Engle & Manganelli (2004) Dynamic Quantile (DQ)
    * Acerbi & Szekely (2014) Z2 (Expected Shortfall)

Conventions
-----------
`returns` and `var_forecast` are time-aligned Series. `var_forecast` is the
*negative* tail quantile (e.g. 0.05 quantile of returns). A violation (hit)
is `returns < var_forecast`. `alpha` is the nominal coverage level (e.g.
0.05 for 5% VaR); expected empirical violation rate is `alpha`.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BacktestResult:
    n_obs: int
    n_violations: int
    empirical_rate: float
    nominal_rate: float

    kupiec_lr: float
    kupiec_pvalue: float

    christoffersen_ind_lr: float
    christoffersen_ind_pvalue: float
    christoffersen_cc_lr: float
    christoffersen_cc_pvalue: float

    dq_stat: float
    dq_pvalue: float

    es_z2: float | None = None
    es_z2_pvalue: float | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def kupiec_pof(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) Proportion-of-Failures LR test."""
    n = len(hits)
    x = int(hits.sum())
    if n == 0:
        return float("nan"), float("nan")
    p = x / n
    if p in (0.0, 1.0):
        if x == 0:
            lr = -2.0 * (n * np.log(1 - alpha))
        else:
            lr = -2.0 * (n * np.log(alpha))
    else:
        lr = -2.0 * (
            x * np.log(alpha) + (n - x) * np.log(1 - alpha)
            - x * np.log(p) - (n - x) * np.log(1 - p)
        )
    pval = 1.0 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(pval)


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) independence test (first-order Markov chain)."""
    n = len(hits)
    if n < 2:
        return float("nan"), float("nan")
    n00 = n01 = n10 = n11 = 0
    for t in range(1, n):
        a, b = hits[t - 1], hits[t]
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        elif a == 1 and b == 1: n11 += 1
    total = n00 + n01 + n10 + n11
    p0 = (n01 + n11) / total if total > 0 else 0.0
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    if p0 in (0.0, 1.0) or p01 in (0.0, 1.0) or p11 in (0.0, 1.0):
        return 0.0, 1.0
    ll_null = ((n00 + n10) * np.log(1 - p0) + (n01 + n11) * np.log(p0))
    ll_alt = (n00 * np.log(1 - p01) + n01 * np.log(p01)
              + n10 * np.log(1 - p11) + n11 * np.log(p11))
    lr = -2.0 * (ll_null - ll_alt)
    pval = 1.0 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(pval)


def christoffersen_cc(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Conditional coverage: unconditional coverage + independence, chi^2_2."""
    lr_uc, _ = kupiec_pof(hits, alpha)
    lr_ind, _ = christoffersen_independence(hits)
    if not (np.isfinite(lr_uc) and np.isfinite(lr_ind)):
        return float("nan"), float("nan")
    lr_cc = lr_uc + lr_ind
    pval = 1.0 - stats.chi2.cdf(lr_cc, df=2)
    return float(lr_cc), float(pval)


def engle_manganelli_dq(hits: np.ndarray, var_forecast: np.ndarray,
                        alpha: float, n_lags: int = 4) -> tuple[float, float]:
    """
    Engle-Manganelli (2004) Dynamic Quantile test. Regress
        (hit_t - alpha)
    on a constant, n_lags lagged hits, and current var_forecast; joint Wald
    test that all coefficients are zero.
    """
    n = len(hits)
    if n <= n_lags + 2:
        return float("nan"), float("nan")
    h = hits.astype(float) - alpha
    rows, targets = [], []
    for t in range(n_lags, n):
        x = [1.0] + [h[t - l] + alpha for l in range(1, n_lags + 1)] + [var_forecast[t]]
        rows.append(x)
        targets.append(h[t])
    X = np.asarray(rows)
    y = np.asarray(targets)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        sigma2 = float(np.var(residuals, ddof=X.shape[1]))
        if sigma2 <= 0 or not np.isfinite(sigma2):
            return float("nan"), float("nan")
        XtX_inv = np.linalg.pinv(X.T @ X)
        cov = sigma2 * XtX_inv
        wald = float(beta @ np.linalg.pinv(cov) @ beta)
        pval = 1.0 - stats.chi2.cdf(wald, df=X.shape[1])
        return wald, float(pval)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")


def acerbi_szekely_z2(returns: pd.Series, var_forecast: pd.Series,
                      es_forecast: pd.Series, alpha: float) -> tuple[float, float]:
    """Acerbi & Szekely (2014) Z2 statistic for Expected Shortfall."""
    df = pd.concat([returns.rename("r"), var_forecast.rename("v"),
                    es_forecast.rename("e")], axis=1).dropna()
    T = len(df)
    if T == 0:
        return float("nan"), float("nan")
    I = (df["r"] < df["v"]).astype(float).values
    contrib = I * df["r"].values / (alpha * T * df["e"].values)
    z2 = float(contrib.sum() + 1.0)
    sigma = max(np.std(contrib) * np.sqrt(T), 1e-9)
    pval = 2.0 * (1.0 - stats.norm.cdf(abs(z2) / sigma))
    return z2, float(pval)


def backtest_var(returns: pd.Series, var_forecast: pd.Series, alpha: float,
                 es_forecast: pd.Series | None = None) -> BacktestResult:
    """Run all VaR coverage backtests; ES Z2 test if es_forecast is provided."""
    df = pd.concat([returns.rename("r"), var_forecast.rename("v")], axis=1).dropna()
    if es_forecast is not None:
        df = pd.concat([df, es_forecast.rename("e")], axis=1).dropna()
    hits = (df["r"].values < df["v"].values).astype(int)
    n = len(hits)
    n_viol = int(hits.sum())
    emp = n_viol / max(n, 1)

    lr_uc, p_uc = kupiec_pof(hits, alpha)
    lr_ind, p_ind = christoffersen_independence(hits)
    lr_cc, p_cc = christoffersen_cc(hits, alpha)
    dq, p_dq = engle_manganelli_dq(hits, df["v"].values, alpha)

    z2 = z2_p = None
    if es_forecast is not None:
        z2, z2_p = acerbi_szekely_z2(df["r"], df["v"], df["e"], alpha)

    return BacktestResult(
        n_obs=n, n_violations=n_viol, empirical_rate=emp, nominal_rate=alpha,
        kupiec_lr=lr_uc, kupiec_pvalue=p_uc,
        christoffersen_ind_lr=lr_ind, christoffersen_ind_pvalue=p_ind,
        christoffersen_cc_lr=lr_cc, christoffersen_cc_pvalue=p_cc,
        dq_stat=dq, dq_pvalue=p_dq,
        es_z2=z2, es_z2_pvalue=z2_p,
    )
