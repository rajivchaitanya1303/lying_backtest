"""
Violation clustering + exceedance severity.

Two questions:

    (a) When violations happen, do they come in runs?
        -> avg / max consecutive violation run length per model.

    (b) When a violation happens, how bad is it?
        -> mean/max exceedance = (var - ret), a positive quantity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _run_lengths(hits: np.ndarray) -> list[int]:
    """List of consecutive-1 run lengths."""
    runs, cur = [], 0
    for h in hits:
        if h == 1:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return runs


def clustering_table(results: pd.DataFrame) -> pd.DataFrame:
    """
    Per (ticker, model): avg run length, max run length, number of runs.
    """
    rows = []
    for (tkr, mdl), grp in results.groupby(["ticker", "model"]):
        grp = grp.sort_values("date")
        runs = _run_lengths(grp["violation"].values.astype(int))
        rows.append({
            "ticker": tkr,
            "model": mdl,
            "n_violations": int(grp["violation"].sum()),
            "n_runs": len(runs),
            "mean_run_len": float(np.mean(runs)) if runs else 0.0,
            "max_run_len": int(np.max(runs)) if runs else 0,
        })
    return pd.DataFrame(rows)


def exceedance_table(results: pd.DataFrame) -> pd.DataFrame:
    """
    Per (ticker, model): mean and max exceedance among violations.
    exceedance = var - ret  (positive when violation occurs).
    """
    rows = []
    for (tkr, mdl), grp in results.groupby(["ticker", "model"]):
        viol = grp[grp["violation"] == 1]
        if len(viol) == 0:
            rows.append({"ticker": tkr, "model": mdl,
                         "n_violations": 0,
                         "mean_exceedance": np.nan,
                         "max_exceedance": np.nan})
            continue
        ex = (viol["var"] - viol["ret"]).values
        rows.append({
            "ticker": tkr,
            "model": mdl,
            "n_violations": len(viol),
            "mean_exceedance": float(np.mean(ex)),
            "max_exceedance": float(np.max(ex)),
        })
    return pd.DataFrame(rows)


def _pinball_loss_per_window(results: pd.DataFrame,
                              alpha: float) -> pd.DataFrame:
    """
    Mean pinball loss per (ticker, model, window_id).
    Pinball_alpha(v, r) = max( alpha*(r - v), (alpha - 1)*(r - v) ).
    Lower is better.
    """
    diff = results["ret"] - results["var"]
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    out = (results.assign(_pinball=loss)
                  .groupby(["ticker", "model", "window_id"])["_pinball"]
                  .mean()
                  .reset_index()
                  .rename(columns={"_pinball": "pinball_mean"}))
    return out


def selection_bias_demo(results: pd.DataFrame,
                        basel_windows: pd.DataFrame,
                        coverage_df: pd.DataFrame | None = None,
                        alpha: float = 0.01) -> pd.DataFrame:
    """
    For each (ticker, window_id) select a model per a given rule; report
    the Basel pass rate of the picked model against the average across
    all models. The gap is the selection-bias inflation.

    Three selection rules reported to demonstrate the inflation is not
    an artefact of one particular criterion:

        1. fewest_violations  - lowest Basel violation count (default;
           matches Basel's own capital-multiplier logic)
        2. highest_kupiec_p   - largest Kupiec POF p-value (highest
           marginal coverage agreement)
        3. lowest_pinball     - smallest mean pinball loss per window
           (best point score under the loss the LSTM optimises)

    Requires coverage_df (output of coverage_per_window) for rule 2. If
    coverage_df is None, only rules 1 and 3 are evaluated.
    """
    all_pass = float(basel_windows["zone"].eq("green").mean())
    n_windows = int(basel_windows.groupby(["ticker", "window_id"]).ngroups)

    def _pass_rate(picked: pd.DataFrame) -> float:
        return float(picked["zone"].eq("green").mean())

    rows = []

    # Rule 1: fewest violations
    r1 = (basel_windows.sort_values("n_violations")
                        .groupby(["ticker", "window_id"])
                        .head(1))
    rows.append({
        "rule": "fewest_violations",
        "avg_basel_pass_rate_all_models": all_pass,
        "selection_picked_pass_rate": _pass_rate(r1),
        "selection_inflation": _pass_rate(r1) - all_pass,
        "n_windows": n_windows,
    })

    # Rule 2: highest Kupiec p-value
    if coverage_df is not None and "kupiec_pvalue" in coverage_df.columns:
        merged = basel_windows.merge(
            coverage_df[["ticker", "model", "window_id", "kupiec_pvalue"]],
            on=["ticker", "model", "window_id"], how="left")
        r2 = (merged.sort_values("kupiec_pvalue", ascending=False)
                     .groupby(["ticker", "window_id"])
                     .head(1))
        rows.append({
            "rule": "highest_kupiec_p",
            "avg_basel_pass_rate_all_models": all_pass,
            "selection_picked_pass_rate": _pass_rate(r2),
            "selection_inflation": _pass_rate(r2) - all_pass,
            "n_windows": n_windows,
        })

    # Rule 3: lowest mean pinball loss
    pinball = _pinball_loss_per_window(results, alpha)
    merged3 = basel_windows.merge(pinball, on=["ticker", "model", "window_id"])
    r3 = (merged3.sort_values("pinball_mean")
                  .groupby(["ticker", "window_id"])
                  .head(1))
    rows.append({
        "rule": "lowest_pinball",
        "avg_basel_pass_rate_all_models": all_pass,
        "selection_picked_pass_rate": _pass_rate(r3),
        "selection_inflation": _pass_rate(r3) - all_pass,
        "n_windows": n_windows,
    })

    return pd.DataFrame(rows)
