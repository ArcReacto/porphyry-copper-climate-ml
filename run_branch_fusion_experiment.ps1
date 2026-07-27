param(
    [string[]]$Datasets = @(),
    [string]$BranchJson = "",
    [string]$OutputRoot = "",
    [int]$NEstimators = 500,
    [int]$MaxDepth = 10,
    [int]$MinSamplesLeaf = 5,
    [int]$NSplits = 5,
    [double]$ResidualThreshold = 0.2,
    [ValidateSet("rf", "hist_gradient_boosting", "logistic_regression")]
    [string]$BaseModel = "rf",
    [switch]$Quick,
    [switch]$IncludeNeutral
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ScriptPath = Join-Path $ProjectRoot "scripts\stage_10_branch_fusion\38_branch_fusion_negative_ratio_experiment.py"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Virtual environment not found. Creating/updating it first..."
    & (Join-Path $ProjectRoot "setup_env.ps1")
}

if (-not $BranchJson) {
    $BranchJson = Join-Path $ProjectRoot "config\feature_branches\feature_branch_assignment.json"
}
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $ProjectRoot "outputs\branch_fusion_experiments"
}

$argsList = @(
    $ScriptPath,
    "--branch-json", $BranchJson,
    "--output-root", $OutputRoot,
    "--n-estimators", "$NEstimators",
    "--max-depth", "$MaxDepth",
    "--min-samples-leaf", "$MinSamplesLeaf",
    "--n-splits", "$NSplits",
    "--residual-threshold", "$ResidualThreshold",
    "--base-model", "$BaseModel"
)

if ($Quick) {
    $argsList += "--quick"
}

if ($IncludeNeutral) {
    $argsList += "--include-neutral"
}

if ($Datasets.Count -gt 0) {
    $argsList += "--datasets"
    $argsList += $Datasets
}

& $VenvPython @argsList
