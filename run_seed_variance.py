"""
Initialisation robustness for the LSTM VaR model.

Re-runs the LSTM walk-forward evaluation for several random seeds at a fixed
epoch count, then reports each metric per seed together with the mean and
standard deviation across seeds. Historical Simulation and GARCH are
deterministic given the data, so only the LSTM is repeated.

    python run_seed_variance.py --alpha 0.01 --seeds 0 1 2 3 4 --epochs 200
    python run_seed_variance.py --alpha 0.01 --tickers SPY --seeds 0 1 2

Output: results/tables/seed_variance_alpha{a}.csv (one row per seed)
        results/tables/seed_variance_summary_alpha{a}.csv (mean and std)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.download import load_returns                         # noqa: E402
from src.var_models import LstmVaR                                 # noqa: E402
from src.evaluation.walk_forward import evaluate_all               # noqa: E402
from src.evaluation.basel_traffic_light import summarize_windows   # noqa: E402
from src.evaluation.copula_backtest import coverage_per_window     # noqa: E402

TABLES = ROOT / "results" / "tables"


def main(alpha, tickers, seeds, epochs):
    TABLES.mkdir(parents=True, exist_ok=True)
    returns = {t: load_returns(t) for t in tickers}
    rows = []
    for seed in seeds:
        res = evaluate_all(
            {"LSTM": lambda s=seed: LstmVaR(alpha=alpha, seq_len=60,
                                           hidden_dim=64, epochs=epochs,
                                           lr=1e-3, batch_size=64, seed=s)},
            returns, train_size=1000, test_size=250, step_size=250,
            out_path=None)
        win = summarize_windows(res, alpha=alpha)
        cov = coverage_per_window(res, alpha=alpha)
        rows.append({
            "seed": seed,
            "epochs": epochs,
            "basel_pass": float(win["zone"].eq("green").mean()),
            "kupiec_pass": float(cov["kupiec_pass"].mean()),
            "independence_pass": float(
                (cov["christoffersen_ind_pvalue"] >= 0.05).mean()),
            "cond_coverage_pass": float(cov["christoffersen_cc_pass"].mean()),
            "kupiec_and_cc_pass": float(cov["copula_pass"].mean()),
            "total_violations": int(cov["n_violations"].sum()),
            "n_windows": int(len(cov)),
        })
        print(rows[-1])
    df = pd.DataFrame(rows)
    tag = f"alpha{alpha:.3f}"
    df.to_csv(TABLES / f"seed_variance_{tag}.csv", index=False)
    cols = [c for c in df.columns if c.endswith("_pass") or c == "total_violations"]
    summ = df[cols].agg(["mean", "std"]).T
    summ.to_csv(TABLES / f"seed_variance_summary_{tag}.csv")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--tickers", nargs="+", default=["SPY", "EWG", "EWJ", "EEM"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args()
    main(a.alpha, a.tickers, a.seeds, a.epochs)
