# 斑岩铜探矿 - 气象影响 - 因果推理项目代码

本项目用于把斑岩铜矿点、负样本、地球化学、地球物理、DEM、断层、地质图和气象数据对齐成表格特征，并运行第一版 baseline 模型。
当前也已经生成气候/风化环境分组字段，可用于下一步按 environment 做因果发现。

## 当前主线

当前默认主线只保留一套建模数据集：

```text
western_core_all_features_v1
```

它包含：

```text
地球化学1 geochem1_usgs_*
地球化学2 geochem2_nure_*
NOAA 北美重力 gravity_na_*
CMMI 重力派生 gravity_cmmi_*
DEM 地形 terrain_*
断层 fault_*
CMMI 地质图 geology_*
TerraClimate 气象 climate_*
```

旧的单特征或多特征组合结果已经归档，不再作为默认输出。

## 环境初始化

```powershell
cd 'C:\Users\PC\Desktop\探矿气象项目代码'
.\setup_env.ps1
```

## 运行数据对齐管线

```powershell
cd 'C:\Users\PC\Desktop\探矿气象项目代码'
.\run_pipeline.ps1
```

该脚本会自动检查/创建虚拟环境，然后依次运行样本构建、地球化学、地球物理、CMMI 重力派生、DEM、断层、地质图、气象、质量报告、分析子集和环境分组脚本。

## 环境分组

```powershell
.\.venv\Scripts\python.exe .\scripts\14_make_environment_groups.py
```

主分组字段：

```text
env_causal_group
```

当前 `western_core` 分组：

```text
arid_basin_or_range: 190
semi_arid_transition: 161
snow_influenced_mountain: 77
subhumid_humid_mountain: 46
```

这些 `env_*` 字段是后续因果发现的分组/元数据，不作为默认 baseline 的预测输入特征。

## 生成主线建模数据集

```powershell
.\.venv\Scripts\python.exe .\scripts\09_make_model_dataset.py
```

输出：

```text
outputs\model_datasets\model_dataset_western_core_all_features_v1.parquet
outputs\model_datasets\model_dataset_western_core_all_features_v1.csv
outputs\model_datasets\model_dataset_western_core_all_features_v1_features.txt
```

当前规模：

```text
474 行
158 个正样本
316 个负样本
703 个模型输入特征
727 个总字段，其中包含 env_* 元数据字段
```

## 训练四个 baseline 模型

快速训练，不计算 permutation importance：

```powershell
.\.venv\Scripts\python.exe .\scripts\10_train_baseline.py --skip-permutation
```

默认训练 4 个模型：

```text
DummyClassifier
LogisticRegression
RandomForestClassifier
HistGradientBoostingClassifier
```

输出：

```text
outputs\baseline_results\baseline_report_western_core_all_features_v1.txt
outputs\baseline_results\baseline_cv_summary_western_core_all_features_v1.csv
outputs\baseline_results\baseline_cv_metrics_western_core_all_features_v1.csv
outputs\baseline_results\baseline_group_errors_western_core_all_features_v1.csv
outputs\baseline_results\baseline_feature_importance_western_core_all_features_v1.csv
```

## 重要说明

- `run_pipeline.ps1` 只负责数据对齐和质量报告，不默认训练模型。
- baseline 分数用于判断当前特征表是否有预测可分性，不等同于因果证据。
- 优先看 `groupkfold_state`，因为它按州分组验证，更接近跨地区泛化。
- 详细字段血缘和对齐逻辑见 `docs\对齐逻辑说明.md`。
