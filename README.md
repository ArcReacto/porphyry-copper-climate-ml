# CD-MPM: Climate-Decoupled Mineral Prospectivity Mapping

This repository contains the experimental code for a multi-source mineral prospectivity mapping project focused on climate-modified surface observations. The current main case study targets porphyry copper deposits in the western United States. The pipeline aligns mineral occurrence labels with geochemical, geophysical, geological, structural, terrain, and climate variables, then evaluates climate-aware prediction and ranking methods.

The central method is **CD-MPM**, a climate-decoupled mineral prospectivity mapping workflow. Instead of treating climate variables only as direct predictors, the workflow uses them to identify climate-associated prospecting features and residualize selected observations before target ranking.

## Project Scope

The project supports:

- multi-source spatial feature alignment for mineral occurrence samples;
- positive, hard-negative, and neutral sample construction;
- baseline mineral prospectivity modeling;
- climate ablation and climate-decoupling experiments;
- climate sensitivity graph construction;
- graph-guided residualization;
- Top-K exploration ranking metrics;
- cross-deposit-type generalization checks.

The main benchmark currently uses a controlled **1:10 positive-to-hard-negative supervised dataset**. Neutral samples are retained in dataset packages but are not used in the main supervised training unless explicitly selected.

## Repository Layout

```text
.
config/                         # Feature branch and experiment configuration files
docs/                           # Notes, experiment summaries, reports, and paper-writing material
figures/                        # Programmatically generated paper figures
figuresnew/                     # Manually redesigned or final figure assets
logs/                           # Environment and pipeline logs
outputs/                        # Generated datasets, model outputs, reports, and figures
scripts/
  stage_01_data_alignment/      # Initial data checks and geochemical/geophysical cleaning
  stage_02_sampling_and_region/
  stage_03_feature_expansion/
  stage_04_model_baseline/
  stage_05_causal_graph/
  stage_06_method_comparison/
  stage_07_climate_decoupling/
  stage_08_standard_workflow/
  stage_09_generalization/
  stage_10_branch_fusion/
  stage_11_paper_optimization/
  stage_12_paper_comparison/
  stage_12_paper_figures/
  stage_13_global_copper_catalog/
src/                            # Shared project utilities, if used by scripts
requirements.txt                # Python dependencies
setup_env.ps1                   # Windows PowerShell environment setup
run_pipeline.ps1                # End-to-end feature alignment pipeline
run_standard_dataset_workflow.ps1
run_paper_optimization_stage.ps1
```

## Data

The repository is designed to work with local geoscience datasets. Large raw and intermediate data files are not expected to be committed to GitHub.

Set a local data root outside the repository, for example:

```text
C:\path\to\local_geoscience_data
```

The data sources include:

- mineral occurrence records and MRDS-derived mining records;
- USGS geochemical data and NURE geochemical data;
- NOAA gravity survey data;
- CMMI gravity derivatives;
- DEM-derived terrain variables;
- geological map and fault features;
- TerraClimate 1991-2020 climatology variables;
- optional external deposit catalogues for generalization checks.

For a clean GitHub version, place data outside the repository and update script arguments or configuration paths as needed.

## Environment Setup

This project is developed for Windows PowerShell and Python 3. The helper script creates a local virtual environment under `.venv/` and installs dependencies from `requirements.txt`.

```powershell
cd "C:\path\to\project"
.\setup_env.ps1
```

After setup, scripts are run through:

```powershell
.\.venv\Scripts\python.exe
```

## Main Workflows

### 1. Build Aligned Feature Tables

Run the end-to-end alignment pipeline:

```powershell
.\run_pipeline.ps1
```

This produces cleaned intermediate files and aligned modeling datasets under:

```text
data_intermediate/
outputs/model_datasets/
```

### 2. Run the Standard Dataset Workflow

For an already constructed modeling dataset, run:

```powershell
.\run_standard_dataset_workflow.ps1 --input-path <model_dataset.parquet>
```

The standard workflow generates:

- baseline model results;
- full/no-climate/residualized climate-decoupling results;
- concept-level causal graph outputs;
- concept-level climate-decoupling outputs.

Default outputs are written to:

```text
outputs/standardized_runs/<dataset_name>/
```

### 3. Run Paper Optimization Experiments

For the current main dataset, run:

```powershell
.\run_paper_optimization_stage.ps1
```

This stage computes extended metrics, builds the climate sensitivity graph, runs graph-guided M4 residualization, and evaluates climate perturbation robustness.

Key outputs are written to:

```text
outputs/paper_optimization/
```

### 4. Run XGBoost M4 Hyperparameter Grid

The graph-guided XGBoost grid searches over:

- graph score threshold `tau`;
- fold-local Spearman threshold `|rho_s|`.

```powershell
.\.venv\Scripts\python.exe .\scripts\stage_11_paper_optimization\70_xgboost_m4_hyperparameter_grid.py
```

Default outputs:

```text
outputs/paper_optimization/xgboost_m4_hyperparameter_grid/
```

## Model Families

The project includes several model and method families:

- **Dummy stratified baseline**: sanity-check baseline.
- **Logistic regression**: linear tabular baseline.
- **Random forest**: tree ensemble baseline used in earlier experiments.
- **HistGradientBoosting**: strong sklearn gradient boosting baseline.
- **XGBoost**: stronger tree boosting model used in paper-oriented experiments.
- **External MPM baselines**: reproduced methods inspired by recent mineral prospectivity mapping literature.
- **M1 Full climate**: uses all features including climate variables.
- **M2 No climate**: removes direct climate variables.
- **M3 Full residualization**: residualizes broad feature sets and serves as an over-decoupling diagnostic.
- **M4 Graph-guided residualization**: residualizes selected climate-associated features using a climate sensitivity graph.

## Evaluation

The main validation protocol is **state-grouped GroupKFold**, where states define validation groups. This is intended as a practical regional validation protocol, not a full metallogenic-belt transfer test.

Reported metrics include:

- ROC-AUC;
- average precision / PR-AUC;
- balanced accuracy;
- precision, recall, and F1;
- Precision@K, Recall@K, F1@K;
- Lift@K;
- NDCG@K.

Top-K metrics are emphasized because exploration is budget-constrained: only a small number of high-ranked targets can be followed up.

## Current Main Experimental Direction

The current paper-oriented storyline is:

1. Multi-source geoscience and climate observations are aligned to candidate sample points.
2. Climate is treated as an observation modifier rather than a direct deposit-forming factor.
3. A climate sensitivity graph identifies stable climate-feature associations.
4. Graph-guided residualization removes selected climate-associated variation from prospecting features.
5. The resulting models are evaluated as target-ranking systems using Top-K metrics.

Recent experiments show that graph-guided M4 can improve head-of-list ranking metrics over no-climate baselines under selected model families, especially when using XGBoost with tuned graph and Spearman thresholds.

## Notes for GitHub Release

Before pushing a public repository, consider excluding:

- `.venv/`;
- `outputs/`;
- `data_intermediate/`;
- raw geoscience data files;
- temporary folders such as `tmp/` and `temp/`;
- `__pycache__/` folders;
- local Word, PowerPoint, and PDF drafts if they are not meant to be shared.

The code is research-oriented and contains exploratory experiments. Reproducing all results requires access to the same local data sources and aligned dataset files.
