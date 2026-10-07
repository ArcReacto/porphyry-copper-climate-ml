# 标准化数据集实验报告：known_mining_neutral_ratio_1_10_supervised_all_features_v1

## 1. 数据集概况

| 项目 | 数值 |
|---|---:|
| 行数 | 1738 |
| 列数 | 682 |
| 正样本 | 158 |
| 负样本 | 1580 |

负样本类型：

```json
{
  "known_mining_area_nearby_negative": 1580,
  "": 158
}
```

## 2. 本次标准流程

| 实验 | 比较对象 | 目的 |
|---|---|---|
| baseline_models | Dummy、Logistic Regression、Random Forest、HistGradientBoosting | 判断当前数据集的基础可预测性 |
| climate_decoupling | M1 Full Climate、M2 No Climate、M3 Full Residualized、M4 Sensitive Residualized | 判断气候变量直接输入/气候残差化/部分残差化的影响 |

M4 的气候敏感特征不再复用旧数据集结果，而是在每个交叉验证 fold 的训练集内部用 Spearman 相关重新筛选，默认阈值为 `0.3`。

本次验证模式：`core`。

- `core`：运行 `stratified_kfold` 和 `groupkfold_state`，适合多个采样比例批量比较。
- `exhaustive`：额外运行 `leave_one_state_out` 和 `leave_one_environment_group_out`，适合最终严格验证。

## 3. 结果摘要

| dataset | experiment | cv | model | roc_auc_mean | average_precision_mean | balanced_accuracy_mean | precision_mean | recall_mean | f1_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | baseline_models | groupkfold_state | dummy_stratified | 0.4798 | 0.0917 | 0.4798 | 0.0537 | 0.0529 | 0.0532 |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | baseline_models | groupkfold_state | hist_gradient_boosting | 0.938 | 0.7176 | 0.7847 | 0.7533 | 0.5946 | 0.6168 |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | baseline_models | groupkfold_state | logistic_regression | 0.8696 | 0.4559 | 0.8023 | 0.416 | 0.7595 | 0.5055 |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | baseline_models | groupkfold_state | random_forest | 0.9497 | 0.7414 | 0.7809 | 0.6461 | 0.6008 | 0.5755 |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | baseline_models | stratified_kfold | dummy_stratified | 0.4899 | 0.0896 | 0.4899 | 0.0704 | 0.0633 | 0.0667 |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | baseline_models | stratified_kfold | hist_gradient_boosting | 0.9861 | 0.9282 | 0.9011 | 0.8554 | 0.8161 | 0.8345 |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | baseline_models | stratified_kfold | logistic_regression | 0.9377 | 0.6476 | 0.9118 | 0.6392 | 0.873 | 0.7372 |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | baseline_models | stratified_kfold | random_forest | 0.9835 | 0.8963 | 0.902 | 0.7723 | 0.8286 | 0.7979 |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | climate_decoupling | groupkfold_state | M1_Full_Climate | 0.9497 | 0.7414 | 0.7809 | 0.6461 | 0.6008 | 0.5755 |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | climate_decoupling | groupkfold_state | M2_No_Climate | 0.9506 | 0.7652 | 0.8292 | 0.6534 | 0.7045 | 0.6562 |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | climate_decoupling | groupkfold_state | M3_Climate_Normalized | 0.92 | 0.7277 | 0.7025 | 0.5312 | 0.432 | 0.4592 |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | climate_decoupling | groupkfold_state | M4_Sensitive_Residualized | 0.9556 | 0.7742 | 0.834 | 0.7211 | 0.7076 | 0.6782 |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | climate_decoupling | stratified_kfold | M1_Full_Climate | 0.9835 | 0.8963 | 0.902 | 0.7723 | 0.8286 | 0.7979 |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | climate_decoupling | stratified_kfold | M2_No_Climate | 0.9828 | 0.8913 | 0.8972 | 0.7446 | 0.8222 | 0.7806 |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | climate_decoupling | stratified_kfold | M3_Climate_Normalized | 0.9844 | 0.9094 | 0.8624 | 0.8251 | 0.7405 | 0.779 |
| known_mining_neutral_ratio_1_10_supervised_all_features_v1 | climate_decoupling | stratified_kfold | M4_Sensitive_Residualized | 0.9834 | 0.8965 | 0.8987 | 0.767 | 0.8222 | 0.7928 |

## 4. 输出目录

```text
PACKAGE_ROOT/outputs/standardized_runs/known_mining_neutral_ratio_1_10_supervised_all_features_v1
```
