# run_extra_experiments.ps1
# Runs the seed-variance check, the four-ETF 99% and 97.5% runs, and the epoch runs
# one after another, writing everything to separate folders. Safe to re-run:
# a step is skipped when its output already exists. A failed step is logged
# and the script moves on to the next one.
#
# Usage from PowerShell:
#     cd "D:\Rajiv\Papers\Behavioral Risk\backtest_lying"
#     powershell -ExecutionPolicy Bypass -File .\run_extra_experiments.ps1
#
# Progress is written to logs\extra_experiments.log. When all steps are
# finished the file logs\EXTRA_DONE.txt is created.

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path logs | Out-Null
$Log = "logs\extra_experiments.log"
$Py = ".btl-venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

# keep the machine awake while the script runs
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Power]::SetThreadExecutionState([uint32]2147483649)

function Log($m) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
    Write-Host $line
    Add-Content -Path $Log -Value $line
}

function Step($name, $doneFile, [scriptblock]$cmd) {
    if (Test-Path $doneFile) { Log "SKIP  $name (found $doneFile)"; return }
    Log "START $name"
    & $cmd 2>&1 | Tee-Object -FilePath ("logs\" + $name + ".log") | Out-Null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $doneFile)) {
        Log "FAIL  $name (exit $LASTEXITCODE, expected $doneFile)"
    } else {
        Log "DONE  $name"
    }
}

Step "seed_variance" "results\tables\seed_variance_summary_alpha0.010.csv" {
    & $Py run_seed_variance.py --alpha 0.01 --tickers SPY --seeds 0 1 2 3 4 --epochs 200
}

Step "full_alpha0.01" "results_rerun_099\tables\headline_alpha0.010.csv" {
    & $Py run_experiment.py --alpha 0.01 --results-dir results_rerun_099
}
if (Test-Path "results_rerun_099\tables\headline_alpha0.010.csv") {
    Step "tables_alpha0.01" "results_rerun_099\tables\layer_decomposition_alpha0.010.csv" {
        & $Py paper_tables.py --alphas 0.01 --results-dir results_rerun_099
    }
}

Step "epochs_50" "results_epochs_50\tables\headline_alpha0.010.csv" {
    & $Py run_experiment.py --alpha 0.01 --tickers SPY --lstm-epochs 50 --results-dir results_epochs_50
}
Step "epochs_400" "results_epochs_400\tables\headline_alpha0.010.csv" {
    & $Py run_experiment.py --alpha 0.01 --tickers SPY --lstm-epochs 400 --results-dir results_epochs_400
}

Step "full_alpha0.025" "results_rerun_0975\tables\headline_alpha0.025.csv" {
    & $Py run_experiment.py --alpha 0.025 --results-dir results_rerun_0975
}
if (Test-Path "results_rerun_0975\tables\headline_alpha0.025.csv") {
    Step "tables_alpha0.025" "results_rerun_0975\tables\layer_decomposition_alpha0.025.csv" {
        & $Py paper_tables.py --alphas 0.025 --results-dir results_rerun_0975
    }
}

Log "ALL STEPS FINISHED"
Set-Content -Path "logs\EXTRA_DONE.txt" -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
[void][Win.Power]::SetThreadExecutionState([uint32]2147483648)
