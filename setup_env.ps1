$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot ".venv"

if (-not (Test-Path -LiteralPath $VenvPath)) {
    py -3 -m venv $VenvPath
}

$Python = Join-Path $VenvPath "Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

Write-Host "Environment ready:"
Write-Host $Python
Write-Host "Run checks with:"
Write-Host "`"$Python`" scripts\stage_01_data_alignment\00_check_environment.py"
