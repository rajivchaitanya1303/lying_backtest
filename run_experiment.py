"""
End-to-end experiment driver for "Your Backtest Is Lying".

Steps
-----
1. Download SPY + EWG + EWJ + EEM daily returns via yfinance (cached).
2. Walk-forward evaluate HistSim / GARCH(1,1)-t / LSTM-VaR at alpha=0.01
   and alpha=0.025 on each ticker.
3. Layer 1: Basel traffic light on every (ticker, model, window).
4. Layer 2: coverage tests (Kupiec, Christoffersen CC, DQ)
            + Romano-Wolf step-down selection-aware p-values.
5. Clustering analysis (run lengths) and exceedance severity.
6. Selection-bias demonstration (best-of-three per window).
7. Save tables to results/tables/ and equity-curve figures to
   results/figures/.

Run from project root:

    python run_experiment.py
    python run_experiment.py --alpha 0.025 --no-lstm    # quick smoke test

All outputs land under `results/`. Nothing is written outside the
project folder.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# make src importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.download import load_returns                      # noqa: E402
from src.var_models import (HistoricalSimulationVaR,            # noqa: E402
                             GarchVaR, LstmVaR)
from src.evaluation.walk_forward import evaluate_all            # noqa: E402
from src.evaluation.basel_traffic_light import (                # noqa: E402
    summarize_windows, pass_rate_table)
from src.evaluation.copula_backtest import (                    # noqa: E402
    coverage_per_window, stepdown_bootstrap)
from src.evaluation.clustering_analysis import (                # noqa: E402
    clustering_table, exceedance_table, selection_bias_demo)
from src.evaluation.crisis_alignment import fisher_alignment    # noqa: E402


# ---------------------------------------------------------------------------
# 0) config
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = ["SPY", "EWG", "EWJ", "EEM"]
DEFAULT_TRAIN, DEFAULT_TEST, DEFAULT_STEP = 1000, 250, 250
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGS = RESULTS / "figures"


def main(alpha: float = 0.01, tickers: list[str] = None,
         include_lstm: bool = True, lstm_epochs: int = 200,
         start: str = "1993-01-29", end: str = "2024-12-31",
         results_dir: str = None):
    global RESULTS, TABLES, FIGS
    if results_dir:
        RESULTS = Path(results_dir).resolve()
        TABLES, FIGS = RESULTS / "tables", RESULTS / "figures"
    tickers = tickers or DEFAULT_TICKERS
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading returns for {tickers} ({start} -> {end}) ...")
    returns = {t: load_returns(t, start=start, end=end) for t in tickers}
    for t, r in returns.items():
        print(f"      {t}: {len(r)} obs")

    # -----------------------------------------------------------------------
    print(f"[2/6] Walk-forward eval at alpha={alpha} "
          f"(train={DEFAULT_TRAIN}, test={DEFAULT_TEST}, step={DEFAULT_STEP})")
    factories = {
        "HistSim": lambda: HistoricalSimulationVaR(
            window=250, alpha=alpha),
        "GARCH": lambda: GarchVaR(alpha=alpha, refit_every=25),
    }
    if include_lstm:
        factories["LSTM"] = lambda: LstmVaR(
            alpha=alpha, seq_len=60, hidden_dim=64,
            epochs=lstm_epochs, lr=1e-3, batch_size=64, seed=0)

    all_results = evaluate_all(
        factories, returns,
        train_size=DEFAULT_TRAIN,
        test_size=DEFAULT_TEST,
        step_size=DEFAULT_STEP,
        out_path=RESULTS / f"raw_forecasts_alpha{alpha:.3f}.parquet",
    )
    print(f"      -> {len(all_results):,} rows saved")

    # -----------------------------------------------------------------------
    print("[3/6] Basel traffic light per window ...")
    windows = summarize_windows(all_results, alpha=alpha)
    windows.to_csv(TABLES / f"basel_windows_alpha{alpha:.3f}.csv", index=False)
    basel_agg = pass_rate_table(windows)
    basel_agg.to_csv(TABLES / f"basel_pass_rate_alpha{alpha:.3f}.csv",
                     index=False)
    print(basel_agg.to_string(index=False))

    # -----------------------------------------------------------------------
    print("[4/6] Coverage tests + Romano-Wolf selection-aware ...")
    cov = coverage_per_window(all_results, alpha=alpha)
    cov.to_csv(TABLES / f"coverage_per_window_alpha{alpha:.3f}.csv",
               index=False)
    adj = stepdown_bootstrap(all_results, alpha)
    cov = cov.merge(adj, on=["ticker", "model", "window_id"], how="left")
    cov["rw_pass"] = cov["stepdown_adj_p"] >= 0.05
    sel = (cov.groupby("model")
              .agg(raw_pass_rate=("kupiec_pass", "mean"),
                   copula_pass_rate=("copula_pass", "mean"),
                   rw_pass_rate=("rw_pass", "mean"))
              .reset_index())
    sel.to_csv(TABLES / f"selection_aware_pass_alpha{alpha:.3f}.csv",
               index=False)
    print(sel.to_string(index=False))

    # -----------------------------------------------------------------------
    print("[5/6] Clustering + exceedance + crisis alignment ...")
    clust = clustering_table(all_results)
    clust.to_csv(TABLES / f"clustering_alpha{alpha:.3f}.csv", index=False)
    exc = exceedance_table(all_results)
    exc.to_csv(TABLES / f"exceedance_alpha{alpha:.3f}.csv", index=False)

    # selection bias under three selection rules
    sel_bias = selection_bias_demo(all_results, windows,
                                   coverage_df=cov, alpha=alpha)
    sel_bias.to_csv(TABLES / f"selection_bias_alpha{alpha:.3f}.csv",
                    index=False)

    # Fisher exact test on (red x crisis-adjacent) per model
    align = fisher_alignment(windows, all_results)
    align.to_csv(TABLES / f"crisis_alignment_alpha{alpha:.3f}.csv",
                 index=False)

    print("      clustering (head):")
    print(clust.head(9).to_string(index=False))
    print("      selection-bias demo (multi-rule):")
    print(sel_bias.to_string(index=False))
    print("      crisis alignment (Fisher exact):")
    print(align[["ticker", "model", "n_windows",
                 "a_red_crisis", "b_red_noncrisis",
                 "c_nored_crisis", "d_nored_noncrisis",
                 "fisher_p_one_sided"]].to_string(index=False))

    # -----------------------------------------------------------------------
    print("[6/6] Equity-curve figures for crisis periods ...")
    for crisis_year in (2008, 2020, 2022):
        plot_crisis(all_results, crisis_year, alpha)
    print(f"      figures in {FIGS}")

    # -----------------------------------------------------------------------
    print("\n=== HEADLINE ===")
    headline = (basel_agg.groupby("model")["basel_pass_rate"].mean()
                .rename("basel").to_frame()
                .join(sel.set_index("model")[
                    ["copula_pass_rate", "rw_pass_rate"]]))
    headline["basel_minus_copula"] = headline["basel"] - headline[
        "copula_pass_rate"]
    print(headline.round(3).to_string())
    headline.to_csv(TABLES / f"headline_alpha{alpha:.3f}.csv")
    print(f"\nDone. All artifacts under {RESULTS}")


def plot_crisis(results: pd.DataFrame, year: int, alpha: float):
    """Plot each model's VaR vs realized returns for one calendar year."""
    for tkr in results["ticker"].unique():
        sub = results[(results["ticker"] == tkr)].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub[sub["date"].dt.year == year]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 4.5))
        # returns (grey scatter)
        ret_series = sub.drop_duplicates("date").sort_values("date")
        ax.plot(ret_series["date"], ret_series["ret"], color="0.6",
                lw=0.6, label="realized return")
        # VaR per model
        for mdl, grp in sub.groupby("model"):
            grp = grp.sort_values("date")
            ax.plot(grp["date"], grp["var"], lw=1.2, label=mdl)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"{tkr} VaR forecasts vs realized returns — {year} "
                     f"(alpha={alpha})")
        ax.set_ylabel("log return")
        ax.legend(loc="lower left", fontsize=8, ncol=4)
        fig.tight_layout()
        fig.savefig(FIGS / f"varcurve_{tkr}_{year}_alpha{alpha:.3f}.png",
                    dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="Tail probability (0.01 = 99%% VaR).")
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--no-lstm", action="store_true",
                    help="Skip LSTM (fast smoke test).")
    ap.add_argument("--lstm-epochs", type=int, default=200)
    ap.add_argument("--start", default="1993-01-29")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--results-dir", default=None,
                    help="Write outputs here instead of results/.")
    args = ap.parse_args()
    main(alpha=args.alpha, tickers=args.tickers,
         include_lstm=not args.no_lstm, lstm_epochs=args.lstm_epochs,
         start=args.start, end=args.end, results_dir=args.results_dir)
