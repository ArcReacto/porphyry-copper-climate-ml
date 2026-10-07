# Rebuttal 实验 06：矿区去重、跨折近邻与空间缓冲审计

## 数据与唯一键

- 监督样本：1738 条，其中正样本 158、hard negative 1580。
- 原 `sample_id` 唯一值：1668；跨州重复编号涉及 140 行。
- 新 `global_sample_id = state | sample_id` 唯一值：1738，重复 0 行。
- 精确坐标唯一值：1737。

## 重复记录审计

| group_type | duplicate_groups | duplicate_rows | label_conflict_groups | cross_state_groups |
| --- | --- | --- | --- | --- |
| deposit_or_source_id | 55 | 112 | 0 | 0 |
| exact_coordinate | 1 | 2 | 0 | 0 |
| positive_site_name | 2 | 4 | 0 | 1 |
| sample_id | 70 | 140 | 0 | 70 |

`duplicate_groups.csv` 保留了每个重复组的逐行明细；重复并不自动等同于泄漏，需结合是否跨折判断。

## 空间邻近簇

| threshold_km | clusters | multi_sample_clusters | samples_in_multi_sample_clusters | cross_state_clusters | label_mixed_clusters | largest_cluster |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 1696 | 32 | 74 | 0 | 0 | 4 |
| 5.0000 | 1527 | 157 | 368 | 0 | 0 | 8 |
| 10.0000 | 1181 | 281 | 838 | 0 | 0 | 12 |
| 20.0000 | 592 | 212 | 1358 | 3 | 0 | 123 |

DBSCAN 使用球面距离且 `min_samples=1`。`cross_state_clusters` 是州分组验证中最需要关注的跨折邻近簇。

## 州分组折的训练—测试最近距离

| fold | test_group | nearest_train_any_km_min | nearest_train_any_km_median | nearest_train_any_km_p05 | nearest_train_same_label_km_min | nearest_train_same_label_km_median | nearest_train_same_label_km_p05 | nearest_train_opposite_label_km_min | nearest_train_opposite_label_km_median | nearest_train_opposite_label_km_p05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Arizona | 16.4726 | 187.4159 | 59.0253 | 16.4726 | 193.0202 | 59.0253 | 65.3358 | 309.3124 | 112.5115 |
| 2 | Washington | 15.3906 | 136.5891 | 37.8563 | 15.3906 | 138.7252 | 37.8563 | 80.7235 | 426.5389 | 212.2389 |
| 3 | Nevada | 27.6534 | 140.7059 | 61.8544 | 32.7603 | 171.8534 | 65.2963 | 27.6534 | 211.9331 | 97.0031 |
| 4 | California;Colorado;New Mexico;Oregon | 15.3906 | 175.2892 | 37.2570 | 15.3906 | 193.2355 | 41.3446 | 27.6534 | 265.2559 | 80.6775 |
| 5 | Idaho;Montana;Utah;Wyoming | 16.4726 | 210.6925 | 46.7080 | 16.4726 | 214.4858 | 53.0618 | 46.6936 | 324.3444 | 122.3792 |

## 缓冲后训练集规模

| buffer_km | fold | test_group | n_train_before | n_train_after | n_train_removed | positive_train_before | positive_train_after | positive_removed | n_test | positive_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 1 | Arizona | 1100 | 1100 | 0 | 100 | 100 | 0 | 638 | 58 |
| 0.0000 | 2 | Washington | 1331 | 1331 | 0 | 121 | 121 | 0 | 407 | 37 |
| 0.0000 | 3 | Nevada | 1430 | 1430 | 0 | 130 | 130 | 0 | 308 | 28 |
| 0.0000 | 4 | California;Colorado;New Mexico;Oregon | 1540 | 1540 | 0 | 140 | 140 | 0 | 198 | 18 |
| 0.0000 | 5 | Idaho;Montana;Utah;Wyoming | 1551 | 1551 | 0 | 141 | 141 | 0 | 187 | 17 |
| 10.0000 | 1 | Arizona | 1100 | 1100 | 0 | 100 | 100 | 0 | 638 | 58 |
| 10.0000 | 2 | Washington | 1331 | 1331 | 0 | 121 | 121 | 0 | 407 | 37 |
| 10.0000 | 3 | Nevada | 1430 | 1430 | 0 | 130 | 130 | 0 | 308 | 28 |
| 10.0000 | 4 | California;Colorado;New Mexico;Oregon | 1540 | 1540 | 0 | 140 | 140 | 0 | 198 | 18 |
| 10.0000 | 5 | Idaho;Montana;Utah;Wyoming | 1551 | 1551 | 0 | 141 | 141 | 0 | 187 | 17 |
| 20.0000 | 1 | Arizona | 1100 | 1099 | 1 | 100 | 100 | 0 | 638 | 58 |
| 20.0000 | 2 | Washington | 1331 | 1329 | 2 | 121 | 121 | 0 | 407 | 37 |
| 20.0000 | 3 | Nevada | 1430 | 1430 | 0 | 130 | 130 | 0 | 308 | 28 |
| 20.0000 | 4 | California;Colorado;New Mexico;Oregon | 1540 | 1538 | 2 | 140 | 140 | 0 | 198 | 18 |
| 20.0000 | 5 | Idaho;Montana;Utah;Wyoming | 1551 | 1549 | 2 | 141 | 141 | 0 | 187 | 17 |
| 30.0000 | 1 | Arizona | 1100 | 1099 | 1 | 100 | 100 | 0 | 638 | 58 |
| 30.0000 | 2 | Washington | 1331 | 1325 | 6 | 121 | 121 | 0 | 407 | 37 |
| 30.0000 | 3 | Nevada | 1430 | 1429 | 1 | 130 | 129 | 1 | 308 | 28 |
| 30.0000 | 4 | California;Colorado;New Mexico;Oregon | 1540 | 1529 | 11 | 140 | 140 | 0 | 198 | 18 |
| 30.0000 | 5 | Idaho;Montana;Utah;Wyoming | 1551 | 1543 | 8 | 141 | 141 | 0 | 187 | 17 |

缓冲操作仅移除距任一测试点小于阈值的训练样本，测试集不变。每个缓冲阈值均重新进行折内 Spearman 特征选择与残差化。

## 缓冲验证结果

| buffer_km | model | roc_auc_mean | average_precision_mean | f1_mean | top05_f1_mean | top05_ndcg_mean | top10_f1_mean | top10_ndcg_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | M4_Spearman_XGB | 0.9546 | 0.7868 | 0.6730 | 0.6321 | 0.9039 | 0.7051 | 0.8025 |
| 0.0000 | NoClimate_XGB | 0.9448 | 0.7010 | 0.6583 | 0.5372 | 0.7831 | 0.6759 | 0.7478 |
| 10.0000 | M4_Spearman_XGB | 0.9546 | 0.7868 | 0.6730 | 0.6321 | 0.9039 | 0.7051 | 0.8025 |
| 10.0000 | NoClimate_XGB | 0.9448 | 0.7010 | 0.6583 | 0.5372 | 0.7831 | 0.6759 | 0.7478 |
| 20.0000 | M4_Spearman_XGB | 0.9523 | 0.7833 | 0.6607 | 0.6178 | 0.8929 | 0.7011 | 0.8004 |
| 20.0000 | NoClimate_XGB | 0.9460 | 0.7104 | 0.6486 | 0.5372 | 0.7974 | 0.6621 | 0.7480 |
| 30.0000 | M4_Spearman_XGB | 0.9515 | 0.7786 | 0.6587 | 0.6230 | 0.8877 | 0.7062 | 0.7979 |
| 30.0000 | NoClimate_XGB | 0.9454 | 0.7093 | 0.6512 | 0.5372 | 0.7960 | 0.6615 | 0.7473 |

## M4 Spearman 相对 M2

| buffer_km | metric | baseline_mean | candidate_mean | mean_difference | better_folds | equal_folds | worse_folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | average_precision | 0.7010 | 0.7868 | 0.0859 | 4 | 0 | 1 |
| 0.0000 | f1 | 0.6583 | 0.6730 | 0.0147 | 2 | 0 | 3 |
| 0.0000 | top05_f1 | 0.5372 | 0.6321 | 0.0949 | 4 | 1 | 0 |
| 0.0000 | top10_f1 | 0.6759 | 0.7051 | 0.0292 | 3 | 1 | 1 |
| 10.0000 | average_precision | 0.7010 | 0.7868 | 0.0859 | 4 | 0 | 1 |
| 10.0000 | f1 | 0.6583 | 0.6730 | 0.0147 | 2 | 0 | 3 |
| 10.0000 | top05_f1 | 0.5372 | 0.6321 | 0.0949 | 4 | 1 | 0 |
| 10.0000 | top10_f1 | 0.6759 | 0.7051 | 0.0292 | 3 | 1 | 1 |
| 20.0000 | average_precision | 0.7104 | 0.7833 | 0.0729 | 4 | 0 | 1 |
| 20.0000 | f1 | 0.6486 | 0.6607 | 0.0121 | 2 | 0 | 3 |
| 20.0000 | top05_f1 | 0.5372 | 0.6178 | 0.0806 | 4 | 1 | 0 |
| 20.0000 | top10_f1 | 0.6621 | 0.7011 | 0.0390 | 3 | 1 | 1 |
| 30.0000 | average_precision | 0.7093 | 0.7786 | 0.0693 | 4 | 0 | 1 |
| 30.0000 | f1 | 0.6512 | 0.6587 | 0.0075 | 2 | 1 | 2 |
| 30.0000 | top05_f1 | 0.5372 | 0.6230 | 0.0858 | 3 | 2 | 0 |
| 30.0000 | top10_f1 | 0.6615 | 0.7062 | 0.0447 | 3 | 1 | 1 |

## 解释边界

1. 该实验排查近邻记忆和跨州边界泄漏，不证明样本标签本身无噪声。
2. 距离缓冲会同时改变训练规模和空间代表性，性能下降不能全部归因于泄漏。
3. 名称和 deposit/source ID 的重复只用于审计；本实验没有擅自删除同矿床的多记录，避免改变论文数据定义。
4. 若缓冲后结论改变，应在 rebuttal 中报告敏感性，不应只保留无缓冲结果。
