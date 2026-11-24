"""
Basel Traffic Light backtest.

Reference: Basel Committee (1996; reaffirmed in BCBS 2019 "Minimum capital
requirements for market risk"). Given a 250-day test window and 99% VaR:

    green  : violations <= 4      (accept model)
    yellow : 5 <= violations <= 9 (increase multiplier / investigate)
    red    : violations >= 10     (reject model)

Zones are calibrated for alpha = 0.01 on 250 days. For other alpha or
test lengths we return raw violation counts and let the caller decide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom


ZONES = ["green", "yellow", "red"]


def classify_zone(n_violations: int, n_obs: int = 250,
                  alpha: float = 0.01) -> str:
    """Basel zones (only exact at alpha=0.01, n=250)."""
    if n_obs == 250 and abs(alpha - 0.01) < 1e-9:
        if n_violations <= 4:
            return "green"
        if n_violations <= 9:
            return "yellow"
        return "red"
    # Generalization for other (n, alpha): match Basel cumulative
    # probability thresholds (~95% and ~99.99%) on the binomial.
    p_yellow = binom.ppf(0.95, n_obs, alpha)   # green upper bound
    p_red = binom.ppf(0.9999, n_obs, alpha)    # yellow upper bound
    if n_violations <= p_yellow:
        return "green"
    if n_violations <= p_red:
        return "yellow"
    return "red"


def summarize_windows(results: pd.DataFrame, alpha: float = 0.01
                      ) -> pd.DataFrame:
    """
    From long-format evaluation output, compute per-(ticker, model, window)
    violation counts and Basel zones.

    Returns columns: ticker, model, window_id, n_obs, n_violations,
                     empirical_rate, zone
    """
    grouped = (results
               .groupby(["ticker", "model", "window_id"])
               .agg(n_obs=("ret", "size"),
                    n_violations=("violation", "sum"))
               .reset_index())
    grouped["empirical_rate"] = grouped["n_violations"] / grouped["n_obs"]
    grouped["zone"] = grouped.apply(
        lambda r: classify_zone(int(r["n_violations"]),
                                int(r["n_obs"]), alpha),
        axis=1,
    )
    return grouped


def pass_rate_table(windows: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to (model x ticker): green/yellow/red counts and Basel pass
    rate (green = pass). One row per (model, ticker).
    """
    counts = (windows
              .groupby(["model", "ticker", "zone"])
              .size()
              .unstack(fill_value=0)
              .reindex(columns=ZONES, fill_value=0)
              .reset_index())
    counts["total"] = counts[ZONES].sum(axis=1)
    counts["basel_pass_rate"] = counts["green"] / counts["total"]
    return counts
