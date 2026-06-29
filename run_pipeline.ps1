$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Setup = Join-Path $ProjectRoot "setup_env.ps1"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Virtual environment not found. Creating it with setup_env.ps1..."
    powershell -ExecutionPolicy Bypass -File $Setup
}
else {
    & $Python -c "import shapefile" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python dependency pyshp not found. Updating environment with setup_env.ps1..."
        powershell -ExecutionPolicy Bypass -File $Setup
    }
}

Push-Location $ProjectRoot
try {
    & $Python .\scripts\00_check_environment.py
    & $Python .\scripts\01_prepare_mines.py
    & $Python .\scripts\07_build_sample_table.py
    & $Python .\scripts\02_prepare_geochem2_nure.py
    & $Python .\scripts\03_prepare_geochem1_usgs.py
    & $Python .\scripts\04_prepare_gravity.py
    & $Python .\scripts\05_spatial_align_features.py
    & $Python .\scripts\13_align_cmmi_gravity_derivatives.py
    & $Python .\scripts\11_align_terrain_geology.py
    & $Python .\scripts\12_align_climate.py
    & $Python .\scripts\06_quality_report.py
    & $Python .\scripts\08_make_analysis_subsets.py
    & $Python .\scripts\14_make_environment_groups.py
}
finally {
    Pop-Location
}
