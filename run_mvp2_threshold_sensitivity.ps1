$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Setup = Join-Path $ProjectRoot "setup_env.ps1"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Virtual environment not found. Creating it with setup_env.ps1..."
    powershell -ExecutionPolicy Bypass -File $Setup
}

Push-Location $ProjectRoot
try {
    & $Python .\scripts\stage_07_climate_decoupling\29_define_decoupled_feature_roles.py
    & $Python .\scripts\stage_07_climate_decoupling\30_diagnose_element_climate_correlation.py
    & $Python .\scripts\stage_07_climate_decoupling\34_mvp2_threshold_sensitivity.py
}
finally {
    Pop-Location
}
