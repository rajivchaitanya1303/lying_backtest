"""
Summary tables for the paper, computed from the stored forecasts and
per-window coverage tests.

For each tail level the script writes, under results/tables/:

    layer_decomposition_alpha{a}.csv   Basel, Kupiec, independence,
                                       conditional coverage, the combined
                                       Kupiec-and-CC rule, and the step-down
                                       adjusted pass rate, by model, plus the
                                       count of Kupiec rejections from too
                                       few and from too many violations
    selection_rules_alpha{a}.csv       ex-post selection under three rules
    crisis_alignment_alpha{a}.csv      Fisher exact test by ticker and model

Run after run_experiment.py:

    python paper_tables.py
    python paper_tables.py --alphas 0.01 0.025
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.evaluation.basel_traffic_light import summarize_windows     # noqa: E402
from src.evaluation.copula_backtest import stepdown_bootstrap        # noqa: E402
from src.evaluation.clustering_analysis import selection_bias_demo   # noqa: E402
from src.evaluation.crisis_alignment import fisher_alignment         # noqa: E402

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"


def build(alpha: float, results: Path = RESULTS) -> None:
    global TABLES
    TABLES = results / "tables"
    tag = f"alpha{alpha:.3f}"
    raw = pd.read_parquet(results / f"raw_forecasts_{tag}.parquet")
    cov = pd.read_csv(TABLES / f"coverage_per_window_{tag}.csv")
    windows = summarize_windows(raw, alpha=alpha)

    cov["kupiec"] = cov["kupiec_pvalue"] >= 0.05
    cov["independence"] = cov["christoffersen_ind_pvalue"] >= 0.05
    cov["cond_coverage"] = cov["christoffersen_cc_pvalue"] >= 0.05
    cov["kupiec_and_cc"] = cov["kupiec"] & cov["cond_coverage"]
    sd = stepdown_bootstrap(raw, alpha)
    cov = cov.merge(sd, on=["ticker", "model", "window_id"], how="left")
    cov["stepdown"] = cov["stepdown_adj_p"] >= 0.05

    basel = (windows.assign(basel=windows["zone"].eq("green"))
                    .groupby("model")["basel"].mean())
    out = (cov.groupby("model")[["kupiec", "independence", "cond_coverage",
                                 "kupiec_and_cc", "stepdown"]].mean()
              .join(basel)
              .join(cov.groupby("model").size().rename("n_windows"))
              [["basel", "kupiec", "independence", "cond_coverage",
                "kupiec_and_cc", "stepdown", "n_windows"]])
    nominal = alpha * cov["n_obs"]
    fail = ~cov["kupiec"]
    cov["kupiec_fail_few"] = fail & (cov["n_violations"] < nominal)
    cov["kupiec_fail_many"] = fail & (cov["n_violations"] > nominal)
    out = out.join(cov.groupby("model")[["kupiec_fail_few",
                                         "kupiec_fail_many"]].sum())
    out.round(4).to_csv(TABLES / f"layer_decomposition_{tag}.csv")
    print(f"\nalpha = {alpha}")
    print(out.round(3).to_string())

    sel = selection_bias_demo(raw, windows, coverage_df=cov, alpha=alpha)
    sel.round(4).to_csv(TABLES / f"selection_rules_{tag}.csv", index=False)
    print(sel.round(3).to_string(index=False))

    align = fisher_alignment(windows, raw)
    align.round(6).to_csv(TABLES / f"crisis_alignment_{tag}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.025])
    ap.add_argument("--results-dir", default=str(RESULTS))
    args = ap.parse_args()
    for a in args.alphas:
        build(a, Path(args.results_dir).resolve())
