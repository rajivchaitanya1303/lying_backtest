# run_all.ps1
# One-shot: env setup + smoke test + full experiment (alpha 0.01 and 0.025).
# Safe to re-run: skips venv creation, pip install, and yfinance downloads
# if they're already done.
#
# Usage from PowerShell:
#     cd "D:\Rajiv\Papers\Behavioral Risk\backtest_lying"
#     powershell -ExecutionPolicy Bypass -File .\run_all.ps1

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Section($msg) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "  $msg"
    Write-Host "============================================================"
}

# 0) verify python -----------------------------------------------------
Section "0/5  Checking Python"
try {
    $pyVer = (python --version) 2>&1
    Write-Host "  found: $pyVer"
    if ($pyVer -match "Microsoft Store" -or $pyVer -match "was not found") {
        throw "Python is the Microsoft Store stub or not installed. Install from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'."
    }
} catch {
    Write-Host "  ERROR: $_" -ForegroundColor Red
    exit 1
}

# 1) venv --------------------------------------------------------------
Section "1/5  Python venv"
$VenvDir = ".btl-venv"
if (-not (Test-Path "$VenvDir\Scripts\python.exe")) {
    Write-Host "  creating $VenvDir (a few seconds) ..."
    python -m venv $VenvDir
    Write-Host "  created $VenvDir"
} else {
    Write-Host "  $VenvDir already exists, reusing"
}
$py = "$VenvDir\Scripts\python.exe"

# 2) deps --------------------------------------------------------------
Section "2/5  Installing dependencies (CPU-only torch)"
Write-Host "  upgrading pip ..."
& $py -m pip install --upgrade pip
Write-Host "  installing core scientific stack (~2-4 min) ..."
& $py -m pip install --progress-bar on `
    numpy pandas scipy scikit-learn matplotlib statsmodels `
    lightgbm yfinance arch pyarrow
Write-Host "  installing CPU-only torch (~200 MB, 2-5 min) ..."
& $py -m pip install --progress-bar on torch --index-url https://download.pytorch.org/whl/cpu

# 3) sanity ------------------------------------------------------------
Section "3/5  Sanity check"
& $py -c "import yfinance, arch, torch, sklearn, scipy, pandas, numpy, matplotlib; print('deps OK, torch', torch.__version__)"

# 4) smoke test --------------------------------------------------------
Section "4/5  Smoke test (SPY, no LSTM, ~2 min)"
& $py run_experiment.py --no-lstm --tickers SPY

# 5) full runs ---------------------------------------------------------
Section "5/5  Full run - alpha 0.01 (99% VaR)"
& $py run_experiment.py

Section "5/5  Full run - alpha 0.025 (97.5% VaR / FRTB)"
& $py run_experiment.py --alpha 0.025

Section "DONE"
Write-Host "Results in:  $PSScriptRoot\results\tables"
Write-Host "Figures in:  $PSScriptRoot\results\figures"
Get-ChildItem .\results\tables -Filter "headline_*.csv" | ForEach-Object {
    Write-Host ""
    Write-Host "--- $($_.Name) ---"
    Get-Content $_.FullName
}
