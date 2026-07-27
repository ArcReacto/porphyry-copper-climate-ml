param(
    [string]$DatasetName = "known_mining_neutral_ratio_1_10_supervised_all_features_v1",
    [double]$GraphScoreThreshold = 0.50,
    [double]$SpearmanThreshold = 0.30
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (!(Test-Path $VenvPython)) {
    throw "Virtual environment Python not found: $VenvPython"
}

$RunDir = Join-Path $ProjectRoot "outputs\standardized_runs\$DatasetName"
$ClimatePred = Join-Path $RunDir "02_climate_decoupling\climate_decoupling_predictions.csv"
$ConceptPred = Join-Path $RunDir "04_concept_climate_decoupling\02_concept_climate_decoupling_predictions.csv"
$GraphDir = Join-Path $ProjectRoot "outputs\paper_optimization\climate_sensitivity_graph\$DatasetName"
$ExtendedDir = Join-Path $ProjectRoot "outputs\paper_optimization\extended_metrics\$DatasetName"
$GraphGuidedDir = Join-Path $ProjectRoot "outputs\paper_optimization\graph_guided_concept_decoupling\$DatasetName"
$FullFeatureGraphGuidedDir = Join-Path $ProjectRoot "outputs\paper_optimization\full_feature_graph_guided_m4\$DatasetName"
$CounterfactualAugDir = Join-Path $ProjectRoot "outputs\paper_optimization\counterfactual_climate_augmentation\$DatasetName"

Write-Host "Paper optimization stage"
Write-Host "Dataset: $DatasetName"
Write-Host "Project root: $ProjectRoot"

& $VenvPython scripts\stage_11_paper_optimization\60_extended_prediction_metrics.py `
    --predictions $ClimatePred $ConceptPred `
    --output-dir $ExtendedDir

& $VenvPython scripts\stage_11_paper_optimization\61_build_climate_sensitivity_graph.py `
    --run-dir $RunDir `
    --output-dir $GraphDir `
    --min-combined-score $GraphScoreThreshold

& $VenvPython scripts\stage_11_paper_optimization\62_graph_guided_concept_decoupling.py `
    --run-dir $RunDir `
    --graph-dir $GraphDir `
    --output-dir $GraphGuidedDir `
    --spearman-threshold $SpearmanThreshold

& $VenvPython scripts\stage_11_paper_optimization\63_full_feature_graph_guided_m4_and_perturbation.py `
    --run-dir $RunDir `
    --graph-dir $GraphDir `
    --output-dir $FullFeatureGraphGuidedDir `
    --spearman-threshold $SpearmanThreshold

& $VenvPython scripts\stage_11_paper_optimization\64_counterfactual_climate_augmentation.py `
    --run-dir $RunDir `
    --graph-dir $GraphDir `
    --output-dir $CounterfactualAugDir `
    --spearman-threshold $SpearmanThreshold

Write-Host ""
Write-Host "Done."
Write-Host "Extended metrics: $ExtendedDir"
Write-Host "Climate sensitivity graph: $GraphDir"
Write-Host "Graph-guided concept decoupling: $GraphGuidedDir"
Write-Host "Full-feature Graph-guided M4: $FullFeatureGraphGuidedDir"
Write-Host "Counterfactual climate augmentation: $CounterfactualAugDir"
