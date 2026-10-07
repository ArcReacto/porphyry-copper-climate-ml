# Rebuttal 实验 01：Fold-local GraphUnion 结果

## 实验边界

本实验在每个 GroupKFold 训练折内重新计算概念相关、环境稳定信号、候选边、图目标集合、Spearman 目标和残差器。测试折只用于最终变换与预测。

## 主结果

| model | roc_auc_mean | average_precision_mean | f1_mean | top05_f1_mean | top05_ndcg_mean | top10_f1_mean | top10_ndcg_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M4_GraphUnion_RF | 0.9493 | 0.7468 | 0.5763 | 0.5949 | 0.8446 | 0.6767 | 0.7553 |
| M4_Spearman_RF | 0.9555 | 0.7631 | 0.6593 | 0.6301 | 0.8858 | 0.6997 | 0.7842 |
| NoClimate_RF | 0.9506 | 0.7652 | 0.6562 | 0.6023 | 0.8789 | 0.6699 | 0.7759 |

## 相对 M2 的逐折差值

| candidate | baseline | metric | baseline_mean | candidate_mean | mean_difference | better_folds | equal_folds | worse_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M4_Spearman_RF | NoClimate_RF | average_precision | 0.7652 | 0.7631 | -0.0021 | 3 | 0 | 2 |
| M4_Spearman_RF | NoClimate_RF | top05_f1 | 0.6023 | 0.6301 | 0.0278 | 3 | 2 | 0 |
| M4_Spearman_RF | NoClimate_RF | top05_ndcg | 0.8789 | 0.8858 | 0.0069 | 2 | 1 | 2 |
| M4_Spearman_RF | NoClimate_RF | top10_f1 | 0.6699 | 0.6997 | 0.0297 | 3 | 1 | 1 |
| M4_Spearman_RF | NoClimate_RF | top10_ndcg | 0.7759 | 0.7842 | 0.0083 | 3 | 0 | 2 |
| M4_GraphUnion_RF | NoClimate_RF | average_precision | 0.7652 | 0.7468 | -0.0184 | 2 | 0 | 3 |
| M4_GraphUnion_RF | NoClimate_RF | top05_f1 | 0.6023 | 0.5949 | -0.0074 | 1 | 3 | 1 |
| M4_GraphUnion_RF | NoClimate_RF | top05_ndcg | 0.8789 | 0.8446 | -0.0343 | 2 | 1 | 2 |
| M4_GraphUnion_RF | NoClimate_RF | top10_f1 | 0.6699 | 0.6767 | 0.0067 | 2 | 2 | 1 |
| M4_GraphUnion_RF | NoClimate_RF | top10_ndcg | 0.7759 | 0.7553 | -0.0206 | 2 | 0 | 3 |

## 每折构图规模

| fold | test_group | n_train | n_test | positive_train | positive_test | stable_concepts | candidate_concepts | stable_edges | selected_graph_concepts | graph_mapped_features | graph_residual_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Arizona | 1100 | 638 | 100 | 58 | 49 | 36 | 35 | 16 | 319 | 318 |
| 2 | Washington | 1331 | 407 | 121 | 37 | 49 | 30 | 32 | 10 | 37 | 36 |
| 3 | Nevada | 1430 | 308 | 130 | 28 | 49 | 33 | 30 | 12 | 46 | 45 |
| 4 | California;Colorado;New Mexico;Oregon | 1540 | 198 | 140 | 18 | 49 | 35 | 32 | 13 | 163 | 162 |
| 5 | Idaho;Montana;Utah;Wyoming | 1551 | 187 | 141 | 17 | 49 | 35 | 28 | 14 | 192 | 191 |

## 解释要求

- 如果 fold-local GraphUnion 仍优于 M2，可将其作为无泄漏图引导结果。
- 如果增益明显收缩，应以 M4 Spearman 作为主方法，并把 GraphUnion 降为探索性知识增强扩展。
- 无论结果方向如何，均不得用 StratifiedKFold 替代本表的跨州结果。