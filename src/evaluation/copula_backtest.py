"""
Selection-aware and conditional-coverage backtest layer.

Two building blocks:

  1. Classical coverage tests implemented in `coverage_tests.py`
     (Kupiec POF, Christoffersen independence + conditional coverage,
     Engle-Manganelli dynamic quantile, Acerbi-Szekely Z2).
     Christoffersen's Markov-chain independence test is a discrete-copula
     treatment of hit-pair dependence; DQ generalises it to arbitrary lags.
     Together they detect the violation clustering that Basel ignores.

  2. A Romano-Wolf step-down multiple-testing correction across all
     (model x window) pairs. This gives a selection-aware p-value: even
     if the analyst picks the best model per window ex post, the
     step-down adjusted rejection rate controls FWER.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .coverage_tests import (
    backtest_var, kupiec_pof, christoffersen_cc, engle_manganelli_dq,
)


# ---------------------------------------------------------------------------
# 1) per-window coverage backtest (Kupiec + Christoffersen CC + DQ)
# ---------------------------------------------------------------------------
def coverage_per_window(results: pd.DataFrame, alpha: float = 0.01
                        ) -> pd.DataFrame:
    """
    Apply the coverage backtest to each (ticker, model, window_id)
    slice. Returns one row per slice.
    """
    rows = []
    for (tkr, mdl, wid), grp in results.groupby(
            ["ticker", "model", "window_id"]):
        grp = grp.sort_values("date")
        res = backtest_var(pd.Series(grp["ret"].values),
                           pd.Series(grp["var"].values), alpha)
        d = res.as_dict()
        d.update({"ticker": tkr, "model": mdl, "window_id": wid})
        rows.append(d)
    df = pd.DataFrame(rows)
    # convenience: pass/fail at 5% for each component
    df["kupiec_pass"] = df["kupiec_pvalue"] >= 0.05
    df["christoffersen_cc_pass"] = df["christoffersen_cc_pvalue"] >= 0.05
    df["dq_pass"] = df["dq_pvalue"] >= 0.05
    # combined "copula" pass: both marginal (Kupiec) and dependence
    # (Christoffersen CC) must hold => joint coverage OK
    df["copula_pass"] = df["kupiec_pass"] & df["christoffersen_cc_pass"]
    return df


# ---------------------------------------------------------------------------
# 2) Romano-Wolf step-down correction across all (model, window) pairs
# ---------------------------------------------------------------------------
def romano_wolf_stepdown(pvalues: np.ndarray, B: int = 2000,
                         seed: int = 0) -> np.ndarray:
    """
    Step-down maxT adjusted p-values with a simulated null.

    Each hypothesis i has statistic t_i = -log p_i. Under the null the p-values
    are independent Uniform(0,1), so the reference distribution of the largest
    statistic among m remaining hypotheses is simulated directly. The adjusted
    p-value of the statistic ranked i (i = 0, 1, ...) is the simulated
    probability that the maximum of the K - i null statistics still in play
    reaches t_i, made monotone along the step-down order.

    Parameters
    ----------
    pvalues : (K,) array of raw p-values from K tests
    B       : number of Monte Carlo replicates for the null distribution
    """
    rng = np.random.default_rng(seed)
    K = len(pvalues)
    # transform p -> test statistic t = -log(p) (larger = more evidence)
    t = -np.log(np.clip(pvalues, 1e-300, 1.0))
    order = np.argsort(-t)     # descending
    t_sorted = t[order]

    adj = np.empty(K)
    running_max = -np.inf
    for i in range(K):
        # step-down: null threshold shrinks as we peel off rejected
        null_max_i = (-np.log(rng.uniform(size=(B, K - i)))).max(axis=1)
        p_i = float(np.mean(null_max_i >= t_sorted[i]))
        # monotonicity enforcement
        running_max = max(running_max, p_i)
        adj[i] = running_max

    out = np.empty(K)
    out[order] = adj
    return out


def selection_aware_summary(cov_df: pd.DataFrame,
                            column: str = "kupiec_pvalue"
                            ) -> pd.DataFrame:
    """
    Add a Romano-Wolf adjusted p-value column, then compute the
    selection-aware pass rate per model.
    """
    df = cov_df.copy()
    df["rw_adj_p"] = romano_wolf_stepdown(df[column].values)
    df["rw_pass"] = df["rw_adj_p"] >= 0.05
    out = (df.groupby("model")
             .agg(raw_pass_rate=("kupiec_pass", "mean"),
                  copula_pass_rate=("copula_pass", "mean"),
                  rw_pass_rate=("rw_pass", "mean"))
             .reset_index())
    return out


# ---------------------------------------------------------------------------
# 3) Step-down maxT with a data bootstrap
# ---------------------------------------------------------------------------
def stepdown_bootstrap(results: pd.DataFrame, alpha: float,
                       B: int = 2000, block: int = 5, seed: int = 0
                       ) -> pd.DataFrame:
    """
    Romano-Wolf style step-down adjusted p-values from a data bootstrap.

    Hypotheses: one per (ticker, model, window), H_k: P(violation) = alpha.
    Statistic: |z_k| with z_k = (X_k - alpha*T) / sqrt(T*alpha*(1-alpha)),
    X_k the violation count in the window.

    Resampling: inside each (ticker, window) the days are resampled with a
    circular moving-block bootstrap (block length `block`), using the same
    day indices for every model, so dependence between models and within a
    window is kept. Windows are resampled independently of each other.
    Bootstrap statistics are centred at the observed value,
    z*_k = (X*_k - X_k) / sqrt(T*alpha*(1-alpha)), which imposes the null.

    Step-down: sort |z_k| in descending order. The adjusted p-value of rank i
    is the share of bootstrap draws in which max over ranks >= i of |z*|
    reaches |z_(i)|, made monotone along the order.
    """
    rng = np.random.default_rng(seed)
    keys, obs, boot = [], [], []
    for (tkr, wid), grp in results.groupby(["ticker", "window_id"]):
        wide = (grp.sort_values("date")
                   .pivot(index="date", columns="model", values="violation"))
        h = wide.values.astype(float)               # (T, n_models)
        T = h.shape[0]
        sd = np.sqrt(T * alpha * (1 - alpha))
        nb = int(np.ceil(T / block))
        starts = rng.integers(0, T, size=(B, nb))
        idx = ((starts[:, :, None] + np.arange(block)) % T
               ).reshape(B, -1)[:, :T]              # (B, T)
        x_obs = h.sum(axis=0)                        # (n_models,)
        x_star = h[idx].sum(axis=1)                  # (B, n_models)
        for j, mdl in enumerate(wide.columns):
            keys.append((tkr, mdl, wid))
            obs.append(abs(x_obs[j] - alpha * T) / sd)
            boot.append(np.abs(x_star[:, j] - x_obs[j]) / sd)
    obs = np.array(obs)
    boot = np.column_stack(boot)                     # (B, K)
    K = len(obs)
    order = np.argsort(-obs)
    adj = np.empty(K)
    running = 0.0
    for i in range(K):
        remaining = order[i:]
        p_i = float(np.mean(boot[:, remaining].max(axis=1) >= obs[order[i]]))
        running = max(running, p_i)
        adj[i] = running
    out = np.empty(K)
    out[order] = adj
    df = pd.DataFrame(keys, columns=["ticker", "model", "window_id"])
    df["stepdown_adj_p"] = out
    return df
