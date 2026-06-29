# Baseline 训练结果总结

本文档总结当前主线全特征数据集的 baseline 训练结果。当前结果用于阶段性诊断，不作为最终因果推理结论。

## 1. 当前训练数据集

当前只保留并训练一套主线数据集：

| 数据集 | 研究区 | 样本数 | 正样本 | 负样本 | 输入特征数 |
|---|---|---:|---:|---:|---:|
| `western_core_all_features_v1` | 美国西部主研究区 | 474 | 158 | 316 | 703 |

该建模表总列数为 727，其中包含 14 个 `env_*` 环境分组/元数据字段。训练时这些字段被排除，不作为预测输入。

特征来源包括：

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

旧的单特征或多组合结果已经移到：

```text
C:\Users\PC\Desktop\探矿气象项目代码\outputs\archive_model_runs_20260625_132409
```

## 2. 模型类型

当前训练脚本使用 4 类 baseline 模型：

| 模型 | 脚本名称 | 作用 |
|---|---|---|
| `DummyClassifier` | `dummy_stratified` | 随机基准模型，用来确认其他模型是否超过无信息基线 |
| `LogisticRegression` | `logistic_regression` | 线性 baseline，观察特征整体线性可分性 |
| `RandomForestClassifier` | `random_forest` | 非线性树模型 baseline |
| `HistGradientBoostingClassifier` | `hist_gradient_boosting` | 更强的梯度提升表格模型 baseline |

优先看：

```text
groupkfold_state
```

因为它按州分组验证，同一州不会同时进入训练集和测试集，比普通随机划分更接近“跨地区泛化”检验。

## 3. 当前结果

### 3.1 按州分组验证：`groupkfold_state`

| 模型 | ROC-AUC | AP | Balanced Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `dummy_stratified` | 0.436 | 0.316 | 0.436 | 0.242 | 0.235 | 0.238 |
| `logistic_regression` | 0.872 | 0.778 | 0.786 | 0.780 | 0.692 | 0.711 |
| `random_forest` | 0.956 | 0.914 | 0.830 | 0.816 | 0.781 | 0.777 |
| `hist_gradient_boosting` | 0.935 | 0.905 | 0.818 | 0.840 | 0.732 | 0.762 |

### 3.2 普通分层交叉验证：`stratified_kfold`

| 模型 | ROC-AUC | AP | Balanced Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `dummy_stratified` | 0.419 | 0.309 | 0.419 | 0.230 | 0.240 | 0.235 |
| `logistic_regression` | 0.957 | 0.918 | 0.905 | 0.857 | 0.886 | 0.869 |
| `random_forest` | 0.980 | 0.962 | 0.930 | 0.880 | 0.923 | 0.900 |
| `hist_gradient_boosting` | 0.979 | 0.962 | 0.916 | 0.883 | 0.892 | 0.886 |

## 4. 阶段性判断

1. 全特征数据集明显超过随机基线，说明当前对齐后的表格特征对正负样本有很强的预测可分性。
2. `groupkfold_state` 比 `stratified_kfold` 更严格；在这个验证方式下，Random Forest 的 ROC-AUC 约 0.956，AP 约 0.914，是当前最稳的强 baseline。
3. Logistic Regression 在 `groupkfold_state` 下 ROC-AUC 约 0.870，说明不只是树模型能学到信号，特征整体也有一定线性可分性。
4. `stratified_kfold` 分数更高，但可能偏乐观，因为相近区域或同州样本可能同时出现在训练集和测试集。
5. 当前结果只能说明预测可分性，不能直接解释为“某个气象或地质因子导致斑岩铜矿出现”。后续因果分析仍需要控制混杂、做敏感性分析，并明确气象变量的现代环境含义。

## 5. 输出位置

建模数据集：

```text
C:\Users\PC\Desktop\探矿气象项目代码\outputs\model_datasets\model_dataset_western_core_all_features_v1.parquet
```

训练结果：

```text
C:\Users\PC\Desktop\探矿气象项目代码\outputs\baseline_results\baseline_report_western_core_all_features_v1.txt
C:\Users\PC\Desktop\探矿气象项目代码\outputs\baseline_results\baseline_cv_summary_western_core_all_features_v1.csv
C:\Users\PC\Desktop\探矿气象项目代码\outputs\baseline_results\baseline_feature_importance_western_core_all_features_v1.csv
```
