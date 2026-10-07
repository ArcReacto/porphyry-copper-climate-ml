# Rebuttal 实验 05：最终方法负样本比例敏感性

本实验使用 known-mining hard-negative 数据集和同一 5 折按州协议。根据实验 01，GraphUnion 不再作为主方法，本表只比较 M2 与完全折内的 M4 Spearman。AP 会随正例率变化，因此跨比例解释同时参考 Lift 和 Top-K 指标。

## 数据审计

| negative_ratio | dataset_path | rows | positive | negative | states | numeric_features | no_climate_features | climate_adjusters | geochemistry_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1:5 | PACKAGE_ROOT/outputs/model_datasets/by_sample_scheme/known_mining_neutral/known_mining_neutral_ratio_1_5/model_dataset_known_mining_neutral_ratio_1_5_supervised_all_features_v1.parquet | 948 | 158 | 790 | 11 | 664 | 385 | 24 | 312 |
| 1:10 | PACKAGE_ROOT/outputs/model_datasets/by_sample_scheme/known_mining_neutral/known_mining_neutral_ratio_1_10/model_dataset_known_mining_neutral_ratio_1_10_supervised_all_features_v1.parquet | 1738 | 158 | 1580 | 11 | 666 | 385 | 24 | 312 |
| 1:20 | PACKAGE_ROOT/outputs/model_datasets/by_sample_scheme/known_mining_neutral/known_mining_neutral_ratio_1_20/model_dataset_known_mining_neutral_ratio_1_20_supervised_all_features_v1.parquet | 3318 | 158 | 3160 | 11 | 668 | 385 | 24 | 312 |

## 主结果

| negative_ratio | model | roc_auc_mean | average_precision_mean | f1_mean | top05_f1_mean | top05_lift_mean | top05_ndcg_mean | top10_f1_mean | top10_lift_mean | top10_ndcg_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1:10 | M4_Spearman_XGB | 0.9546 | 0.7868 | 0.6730 | 0.6321 | 9.6165 | 0.9039 | 0.7051 | 7.3740 | 0.8025 |
| 1:10 | NoClimate_XGB | 0.9448 | 0.7010 | 0.6583 | 0.5372 | 8.1610 | 0.7831 | 0.6759 | 7.0678 | 0.7478 |
| 1:20 | M4_Spearman_XGB | 0.9540 | 0.6774 | 0.5960 | 0.6258 | 12.7827 | 0.7154 | 0.5176 | 8.0112 | 0.8115 |
| 1:20 | NoClimate_XGB | 0.9433 | 0.6435 | 0.5488 | 0.5770 | 11.7811 | 0.6681 | 0.5175 | 8.0097 | 0.7952 |
| 1:5 | M4_Spearman_XGB | 0.9503 | 0.8477 | 0.7459 | 0.4838 | 5.8667 | 0.9798 | 0.7125 | 5.5967 | 0.9479 |
| 1:5 | NoClimate_XGB | 0.9462 | 0.8412 | 0.7713 | 0.4704 | 5.7000 | 0.9589 | 0.7009 | 5.5060 | 0.9357 |

## M4 相对 M2

| negative_ratio | metric | baseline_mean | candidate_mean | mean_difference | better_folds | equal_folds | worse_folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1:10 | average_precision | 0.7010 | 0.7868 | 0.0859 | 4 | 0 | 1 |
| 1:10 | f1 | 0.6583 | 0.6730 | 0.0147 | 2 | 0 | 3 |
| 1:10 | top05_f1 | 0.5372 | 0.6321 | 0.0949 | 4 | 1 | 0 |
| 1:10 | top05_ndcg | 0.7831 | 0.9039 | 0.1207 | 5 | 0 | 0 |
| 1:10 | top10_f1 | 0.6759 | 0.7051 | 0.0292 | 3 | 1 | 1 |
| 1:10 | top10_ndcg | 0.7478 | 0.8025 | 0.0547 | 4 | 0 | 1 |
| 1:20 | average_precision | 0.6435 | 0.6774 | 0.0339 | 4 | 0 | 1 |
| 1:20 | f1 | 0.5488 | 0.5960 | 0.0472 | 3 | 0 | 2 |
| 1:20 | top05_f1 | 0.5770 | 0.6258 | 0.0489 | 3 | 1 | 1 |
| 1:20 | top05_ndcg | 0.6681 | 0.7154 | 0.0473 | 4 | 0 | 1 |
| 1:20 | top10_f1 | 0.5175 | 0.5176 | 0.0001 | 1 | 2 | 2 |
| 1:20 | top10_ndcg | 0.7952 | 0.8115 | 0.0163 | 4 | 0 | 1 |
| 1:5 | average_precision | 0.8412 | 0.8477 | 0.0065 | 3 | 0 | 2 |
| 1:5 | f1 | 0.7713 | 0.7459 | -0.0254 | 2 | 0 | 3 |
| 1:5 | top05_f1 | 0.4704 | 0.4838 | 0.0134 | 2 | 3 | 0 |
| 1:5 | top05_ndcg | 0.9589 | 0.9798 | 0.0208 | 2 | 3 | 0 |
| 1:5 | top10_f1 | 0.7009 | 0.7125 | 0.0116 | 2 | 2 | 1 |
| 1:5 | top10_ndcg | 0.9357 | 0.9479 | 0.0122 | 3 | 0 | 2 |
