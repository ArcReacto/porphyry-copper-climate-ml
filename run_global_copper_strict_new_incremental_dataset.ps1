$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Script = Join-Path $ProjectRoot "scripts\stage_13_global_copper_catalog\81_build_strict_new_incremental_dataset.py"

if (!(Test-Path $Python)) {
    throw "Python environment not found: $Python"
}

& $Python $Script
