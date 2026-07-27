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
    & $Python .\scripts\stage_05_causal_graph\41_run_concept_climate_decoupling.py @args
}
finally {
    Pop-Location
}
