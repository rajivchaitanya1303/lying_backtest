# Your Backtest Is Lying To You

**Conditional-Coverage and Selection-Aware Evaluation of ML Value-at-Risk Models.**

Code and data release for the paper. The empirical study
asks whether the Basel traffic-light backtest, the industry standard for
VaR model validation, can distinguish a genuinely calibrated tail model
from one that has learned to track recent returns. Three VaR estimators
(Historical Simulation, GARCH(1,1)-t, LSTM quantile regression) are
walk-forward evaluated on SPY 1993-2024 plus three international ETFs,
under Basel and two additional layers: a coverage and conditional-coverage layer
(Kupiec, Christoffersen, Engle-Manganelli DQ) and a selection-aware
layer (Romano-Wolf step-down on the model x window p-value matrix).

## Repo layout

```
backtest_lying/
├── src/
│   ├── data/download.py             # yfinance loader + CSV cache
│   ├── var_models/
│   │   ├── historical_sim.py
│   │   ├── garch.py                 # arch package, Student-t
│   │   └── lstm_var.py              # PyTorch, quantile regression
│   └── evaluation/
│       ├── walk_forward.py          # rolling train/test harness
│       ├── basel_traffic_light.py   # green/yellow/red zones
│       ├── coverage_tests.py        # Kupiec, Christoffersen, DQ, AS Z2
│       ├── copula_backtest.py       # coverage + Romano-Wolf step-down
│       └── clustering_analysis.py   # run lengths + exceedance severity
├── paper_tables.py                  # decomposition, selection, crisis tables
├── run_seed_variance.py             # LSTM seed-variance check
├── data/cache/                      # yfinance-cached CSVs (in-repo)
├── results/
│   ├── tables/                      # CSV outputs
│   └── figures/                     # PNG equity curves for 2008/2020/2022
├── notebooks/exploration.ipynb
├── run_experiment.py                # single-command driver
├── run_all.ps1                      # env setup + smoke + full runs
├── requirements.txt
└── LICENSE                          # MIT
```

## How to run

### 1. One-time environment setup

```powershell
cd D:\Rajiv\Papers\Behavioral Risk\backtest_lying
python -m venv .btl-venv
.btl-venv\Scripts\activate
pip install -r requirements.txt
```

CPU-only PyTorch (recommended, works on any machine):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Smoke test (2 minutes, no LSTM, SPY only)

```powershell
python run_experiment.py --no-lstm --tickers SPY
```

### 3. Full experiment

```powershell
python run_experiment.py                          # alpha = 0.01 (99% VaR)
python run_experiment.py --alpha 0.025            # 97.5% VaR (FRTB)
```

Runtime on a modern Intel i5: roughly 45 to 90 minutes for the full
4-ticker run at each alpha. No GPU required.

### 4. One-command end-to-end

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

Creates the venv, installs deps, runs smoke test, then both full runs.

### 5. Inspect outputs

```
results/tables/basel_pass_rate_alpha0.010.csv        <- Basel headline
results/tables/coverage_per_window_alpha0.010.csv    <- per-window Kupiec/CC/DQ
results/tables/selection_aware_pass_alpha0.010.csv   <- Romano-Wolf headline
results/tables/clustering_alpha0.010.csv             <- avg/max run length
results/tables/exceedance_alpha0.010.csv             <- severity
results/tables/selection_bias_alpha0.010.csv         <- cherry-pick demo
results/tables/headline_alpha0.010.csv               <- one-row-per-model summary
results/tables/layer_decomposition_alpha0.010.csv    <- every layer, by model (paper_tables.py)
results/tables/selection_rules_alpha0.010.csv        <- three selection rules (paper_tables.py)
results/tables/crisis_alignment_alpha0.010.csv       <- Fisher exact test (paper_tables.py)
```

## What each evaluation layer catches

| Layer | Detects | Ignores |
|---|---|---|
| Basel traffic light | Marginal violation frequency | Clustering, selection |
| Kupiec POF | Marginal frequency (LR test) | Clustering, selection |
| Christoffersen CC | Frequency + first-order clustering (Markov-chain, equivalently discrete copula on hit pairs) | Higher-order clustering, selection |
| Engle-Manganelli DQ | Frequency + arbitrary-lag clustering | Selection |
| Step-down maxT with a moving-block bootstrap across the (model x window) matrix | Rejections that survive a family-wise adjustment | Cross-ticker dependence, the selected model's pass rate |

`copula_backtest.py` combines Kupiec and Christoffersen conditional
coverage into a joint pass flag and adds the step-down adjustment.

After `run_experiment.py`, run `python paper_tables.py` for the summary
tables, and `python run_seed_variance.py` for the seed check.

## Reproducibility

- Seeds fixed: `numpy.random.default_rng(0)`, `torch.manual_seed(0)`.
- Yahoo Finance prices cached at `data/cache/{TICKER}.csv` and committed
  to the repo, so a reviewer without Yahoo access can still reproduce.
- Every table is CSV; every figure is PNG. `results/raw_forecasts_*.parquet`
  contains every (date, model, ticker, var, ret, violation) row for
  independent re-analysis.

## License

MIT. See `LICENSE`.
