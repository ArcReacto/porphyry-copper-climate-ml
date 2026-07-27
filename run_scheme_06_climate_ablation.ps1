$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Setup = Join-Path $ProjectRoot "setup_env.ps1"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Virtual environment not found. Creating it with setup_env.ps1..."
    powershell -ExecutionPolicy Bypass -File $Setup
}

& $Python -c "import imblearn" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python dependency imbalanced-learn not found. Updating environment..."
    & $Python -m pip install imbalanced-learn
}

Push-Location $ProjectRoot
try {
    & $Python .\scripts\stage_06_method_comparison\27_climate_ablation_balanced_rf.py
    & $Python .\scripts\stage_06_method_comparison\28_climate_ablation_random_forest.py
}
finally {
    Pop-Location
}
