# 脚本阶段目录说明

本目录按项目推进阶段整理脚本。脚本编号保留执行顺序；方法对比部分的“方案编号”以 `docs/方法对比实验设计.md` 为准。

## stage_01_data_alignment

早期数据检查、正样本整理、地球化学/地球物理基础数据清洗，以及初始空间对齐。

- `00_check_environment.py`：检查项目配置、依赖、数据文件是否存在。
- `01_prepare_mines.py`：整理斑岩铜矿正样本矿点。
- `02_prepare_geochem2_nure.py`：清洗 NURE 地球化学数据。
- `03_prepare_geochem1_usgs.py`：清洗 USGS 地球化学 parquet 数据。
- `04_prepare_gravity.py`：清洗 NOAA 重力点数据。
- `05_spatial_align_features.py`：把矿点/样本点与地球化学、基础重力数据做空间邻域对齐。
- `06_quality_report.py`：生成对齐质量报告。

## stage_02_sampling_and_region

负样本生成、研究区筛选和样本子集构建。

- `07_build_sample_table.py`：构建正负样本表。
- `08_make_analysis_subsets.py`：生成 western_core 等研究区子集。

## stage_03_feature_expansion

在基础对齐结果上接入更多环境和背景变量。

- `11_align_terrain_geology.py`：接入 DEM 地形、地质图、断层特征。
- `12_align_climate.py`：接入 TerraClimate 气象气候特征。
- `13_align_cmmi_gravity_derivatives.py`：接入 CMMI 补充重力派生特征。
- `14_make_environment_groups.py`：按气候/地形背景划分环境分组。

## stage_04_model_baseline

构建最终建模数据集，并运行四个基础预测模型。

- `09_make_model_dataset.py`：构建 all-features 建模数据集。
- `10_train_baseline.py`：训练 Logistic Regression、Random Forest、HistGradientBoosting、Dummy baseline。

## stage_05_causal_graph

面向因果图的概念变量构建、环境稳定性分析、分环境因果发现和解释图整理。

- `15_create_feature_concept_mapping.py`：把原始字段映射到概念变量。
- `16_build_concept_features.py`：构建概念级特征表。
- `17_train_concept_baseline.py`：训练概念级 baseline。
- `18_environment_stability_analysis.py`：分析不同环境内变量稳定性。
- `19_discover_environment_causal_graphs.py`：分环境发现候选因果边。
- `20_compare_cross_environment_edges.py`：比较跨环境稳定边。
- `21_build_causal_interpretation_graph.py`：生成知识图谱式解释结构。

## stage_06_method_comparison

后续方法对比实验。这里的方案编号对应 `docs/方法对比实验设计.md`。

- `22_compare_ml_baselines.py`：对比方案一，普通机器学习 baseline 对比。
- `23_train_causal_core_baseline.py`：对比方案二，特征空间对比，构建核心因果图特征数据集并训练四个 baseline。
- `24_run_mpm_weighted_overlay.py`：对比方案三，传统 MPM/Fuzzy weighted overlay 加权叠加评分。
- `26_compare_causal_discovery_methods.py`：对比方案四，因果发现方法对比，使用 PC-like 偏相关、标准 PC algorithm 和 GES/BIC 作为对照。
- `25_compare_method_explanations.py`：对比方案五，解释结果对比，对比 ML 重要性、MPM 权重和因果图稳定边的一致性。
- `27_climate_ablation_balanced_rf.py`：对比方案六，气象变量消融主实验，使用 BalancedRandomForest 比较 Full 与 No-Climate。
- `28_climate_ablation_random_forest.py`：对比方案六，气象变量消融鲁棒性实验，使用普通 RandomForest 且不做欠采样。

注意：`25` 和 `26` 的脚本编号反映实际执行顺序，但文档方案编号中，因果发现方法对比是方案四，解释结果对比是方案五。

## stage_07_climate_decoupling

气候解耦与跨环境鲁棒性验证。该阶段用气候变量校正地球化学元素信号，但目标模型不直接输入原始气候变量。

- `29_define_decoupled_feature_roles.py`：定义地球化学、气候、地质结构、元数据和标签等变量角色。
- `30_diagnose_element_climate_correlation.py`：计算元素-气候相关性，识别气候敏感元素。
- `31_compare_climate_residual_models.py`：比较 M1 Full Climate、M2 No Climate、M3 Climate Normalized。
- `32_leave_region_out_validation.py`：做留一州和留一环境组跨区域验证。
- `33_summarize_climate_decoupling_results.py`：汇总阶段结果并生成 Markdown 报告。
- `34_mvp2_threshold_sensitivity.py`：比较 `|Spearman r| >= 0.2/0.3/0.4` 三种气候敏感阈值下的 M4 表现。
- `35_negative_ratio_sensitivity.py`：构建 1:5、1:10、1:20 负样本比例数据集，数据集归档到 `outputs/model_datasets/by_sample_scheme/ratio_*/`，并比较 M1-M4 表现。
- `run_build_sample_scheme_datasets.ps1`：只构建并归档 `1:2`、`1:5`、`1:10`、`1:20` 四套采样方案数据集，不跑模型训练。
- `decoupling_utils.py`：阶段 07 的公共工具函数。

## 常用运行方式

完整数据构建流程仍可从项目根目录运行：

```powershell
.\run_pipeline.ps1
```

单独运行脚本时，使用阶段目录后的新路径，例如：

```powershell
.\.venv\Scripts\python.exe .\scripts\stage_05_causal_graph\17_train_concept_baseline.py
.\.venv\Scripts\python.exe .\scripts\stage_06_method_comparison\22_compare_ml_baselines.py
.\.venv\Scripts\python.exe .\scripts\stage_06_method_comparison\23_train_causal_core_baseline.py
.\.venv\Scripts\python.exe .\scripts\stage_06_method_comparison\24_run_mpm_weighted_overlay.py
.\.venv\Scripts\python.exe .\scripts\stage_06_method_comparison\26_compare_causal_discovery_methods.py
.\.venv\Scripts\python.exe .\scripts\stage_06_method_comparison\25_compare_method_explanations.py
.\.venv\Scripts\python.exe .\scripts\stage_06_method_comparison\27_climate_ablation_balanced_rf.py
.\.venv\Scripts\python.exe .\scripts\stage_06_method_comparison\28_climate_ablation_random_forest.py
.\.venv\Scripts\python.exe .\scripts\stage_07_climate_decoupling\31_compare_climate_residual_models.py
```

方案六也可以直接运行根目录入口：

```powershell
.\run_scheme_06_climate_ablation.ps1
```

阶段 07 可以直接运行根目录入口：

```powershell
.\run_stage_07_climate_decoupling.ps1
```

MVP2 阈值敏感性实验：

```powershell
.\run_mvp2_threshold_sensitivity.ps1
```

负样本比例敏感性实验：

```powershell
.\run_negative_ratio_sensitivity.ps1
```

## stage_08_standard_workflow

标准化数据集实验入口。该阶段不重新采样、不重新空间对齐，而是读取已经构建好的
`model_dataset_*.csv/parquet`，然后把 baseline 模型和气候解耦实验输出到统一目录：

```text
outputs/standardized_runs/<数据集名>/
```

- `40_run_standard_dataset_workflow.py`：对一个或多个建模数据集运行统一实验流程，包含四个 baseline 模型和 M1-M4 气候解耦模型。

运行示例：

```powershell
.\run_standard_dataset_workflow.ps1 --datasets "C:\Users\PC\Desktop\探矿气象项目代码\outputs\model_datasets\model_dataset_western_core_all_features_v1.parquet"
```

M4 在该标准入口中会在每个交叉验证 fold 的训练集内部重新筛选气候敏感地球化学特征，避免复用旧数据集筛选表。

## 标准化因果图入口

stage_05 也新增了一个标准化入口：

```powershell
.\run_standard_causal_graph_workflow.ps1 --datasets "C:\Users\PC\Desktop\探矿气象项目代码\outputs\model_datasets\by_sample_scheme\ratio_1_10\model_dataset_western_core_ratio_1_10_all_features_v1.parquet"
```

对应脚本：

```text
scripts\stage_05_causal_graph\40_run_standard_causal_graph_workflow.py
```

输出目录：

```text
outputs\standardized_runs\<数据集名>\03_causal_graph
```

它会依次完成字段到概念映射、概念特征聚合、概念 baseline、分环境稳定性分析、分环境候选因果边发现和跨环境稳定边比较。
## Concept-level climate decoupling

After running the standardized causal graph workflow, concept features can be used for a concept-level climate decoupling experiment:

```powershell
.\run_concept_climate_decoupling.ps1 --validation-mode core --datasets "C:\Users\PC\Desktop\探矿气象项目代码\outputs\model_datasets\by_sample_scheme\ratio_1_10\model_dataset_western_core_ratio_1_10_all_features_v1.parquet"
```

Corresponding script:

```text
scripts\stage_05_causal_graph\41_run_concept_climate_decoupling.py
```

Output directory:

```text
outputs\standardized_runs\<dataset_name>\04_concept_climate_decoupling
```

This experiment compares all concepts, no climate concepts, all non-climate concepts residualized by climate concepts, and only climate-sensitive concepts residualized.

## stage_09_generalization

Strict generalization and de-regionalization audit:

```powershell
.\run_generalization_audit.ps1 --models hist_gradient_boosting random_forest --max-splits 3 --feature-sets all_features no_climate_no_location core_geo_geochem_geophysics concept_all concept_no_climate --datasets "C:\Users\PC\Desktop\探矿气象项目代码\outputs\model_datasets\by_sample_scheme\ratio_1_10\model_dataset_western_core_ratio_1_10_all_features_v1.parquet"
```

Corresponding script:

```text
scripts\stage_09_generalization\50_generalization_audit.py
```

Output directory:

```text
outputs\generalization_audit*
```

This stage compares stratified K-fold, state-grouped K-fold, spatial-block K-fold, and optional leave-one-state-out validation under several feature ablations.

## Known-mining neutral sample scheme

Build the revised sample scheme where random no-known-mine background points are treated as neutral samples and negatives are sampled near known non-target MRDS mining areas:

```powershell
.\run_known_mining_neutral_dataset.ps1
```

Corresponding script:

```text
scripts\stage_02_sampling_and_region\09_build_known_mining_neutral_dataset.py
```

Output directory:

```text
outputs\model_datasets\by_sample_scheme\known_mining_neutral\known_mining_neutral_ratio_1_10
```

The supervised binary dataset excludes neutral rows and can be used directly by the standard workflow.
