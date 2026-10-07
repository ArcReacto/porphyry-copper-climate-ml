# Rebuttal 实验 02：最终模型空间块验证

每个空间网格只属于一个测试折；每折内部重新构图、选择特征并拟合残差器。2° 为主尺度，3° 为尺度敏感性对照。

## 主结果

| grid_degrees | model | roc_auc_mean | average_precision_mean | f1_mean | top05_f1_mean | top05_ndcg_mean | top10_f1_mean | top10_ndcg_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.0000 | M4_GraphUnion_RF | 0.9628 | 0.7597 | 0.6893 | 0.6305 | 0.8233 | 0.6482 | 0.8391 |
| 2.0000 | M4_Spearman_RF | 0.9657 | 0.7764 | 0.6943 | 0.6146 | 0.8290 | 0.6590 | 0.8646 |
| 2.0000 | NoClimate_RF | 0.9633 | 0.7825 | 0.6763 | 0.6341 | 0.8495 | 0.6524 | 0.8717 |
| 3.0000 | M4_GraphUnion_RF | 0.9568 | 0.7624 | 0.6784 | 0.6210 | 0.8590 | 0.6021 | 0.8159 |
| 3.0000 | M4_Spearman_RF | 0.9558 | 0.7429 | 0.6818 | 0.5969 | 0.8308 | 0.6258 | 0.8199 |
| 3.0000 | NoClimate_RF | 0.9582 | 0.7502 | 0.6780 | 0.5937 | 0.8285 | 0.6440 | 0.8418 |

## 相对 M2 的逐折差值

| grid_degrees | candidate | baseline | metric | baseline_mean | candidate_mean | mean_difference | better_folds | equal_folds | worse_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.0000 | M4_Spearman_RF | NoClimate_RF | average_precision | 0.7825 | 0.7764 | -0.0061 | 2 | 0 | 3 |
| 2.0000 | M4_Spearman_RF | NoClimate_RF | f1 | 0.6763 | 0.6943 | 0.0180 | 3 | 0 | 2 |
| 2.0000 | M4_Spearman_RF | NoClimate_RF | top05_f1 | 0.6341 | 0.6146 | -0.0195 | 1 | 3 | 1 |
| 2.0000 | M4_Spearman_RF | NoClimate_RF | top05_ndcg | 0.8495 | 0.8290 | -0.0204 | 3 | 0 | 2 |
| 2.0000 | M4_Spearman_RF | NoClimate_RF | top10_f1 | 0.6524 | 0.6590 | 0.0066 | 1 | 4 | 0 |
| 2.0000 | M4_Spearman_RF | NoClimate_RF | top10_ndcg | 0.8717 | 0.8646 | -0.0071 | 2 | 0 | 3 |
| 2.0000 | M4_GraphUnion_RF | NoClimate_RF | average_precision | 0.7825 | 0.7597 | -0.0228 | 2 | 0 | 3 |
| 2.0000 | M4_GraphUnion_RF | NoClimate_RF | f1 | 0.6763 | 0.6893 | 0.0131 | 2 | 1 | 2 |
| 2.0000 | M4_GraphUnion_RF | NoClimate_RF | top05_f1 | 0.6341 | 0.6305 | -0.0036 | 1 | 2 | 2 |
| 2.0000 | M4_GraphUnion_RF | NoClimate_RF | top05_ndcg | 0.8495 | 0.8233 | -0.0262 | 3 | 0 | 2 |
| 2.0000 | M4_GraphUnion_RF | NoClimate_RF | top10_f1 | 0.6524 | 0.6482 | -0.0042 | 1 | 2 | 2 |
| 2.0000 | M4_GraphUnion_RF | NoClimate_RF | top10_ndcg | 0.8717 | 0.8391 | -0.0327 | 1 | 0 | 4 |
| 3.0000 | M4_Spearman_RF | NoClimate_RF | average_precision | 0.7502 | 0.7429 | -0.0073 | 3 | 0 | 2 |
| 3.0000 | M4_Spearman_RF | NoClimate_RF | f1 | 0.6780 | 0.6818 | 0.0038 | 1 | 1 | 3 |
| 3.0000 | M4_Spearman_RF | NoClimate_RF | top05_f1 | 0.5937 | 0.5969 | 0.0032 | 2 | 2 | 1 |
| 3.0000 | M4_Spearman_RF | NoClimate_RF | top05_ndcg | 0.8285 | 0.8308 | 0.0023 | 4 | 0 | 1 |
| 3.0000 | M4_Spearman_RF | NoClimate_RF | top10_f1 | 0.6440 | 0.6258 | -0.0182 | 1 | 2 | 2 |
| 3.0000 | M4_Spearman_RF | NoClimate_RF | top10_ndcg | 0.8418 | 0.8199 | -0.0219 | 3 | 0 | 2 |
| 3.0000 | M4_GraphUnion_RF | NoClimate_RF | average_precision | 0.7502 | 0.7624 | 0.0122 | 2 | 0 | 3 |
| 3.0000 | M4_GraphUnion_RF | NoClimate_RF | f1 | 0.6780 | 0.6784 | 0.0004 | 3 | 0 | 2 |
| 3.0000 | M4_GraphUnion_RF | NoClimate_RF | top05_f1 | 0.5937 | 0.6210 | 0.0273 | 3 | 2 | 0 |
| 3.0000 | M4_GraphUnion_RF | NoClimate_RF | top05_ndcg | 0.8285 | 0.8590 | 0.0305 | 5 | 0 | 0 |
| 3.0000 | M4_GraphUnion_RF | NoClimate_RF | top10_f1 | 0.6440 | 0.6021 | -0.0419 | 1 | 1 | 3 |
| 3.0000 | M4_GraphUnion_RF | NoClimate_RF | top10_ndcg | 0.8418 | 0.8159 | -0.0259 | 2 | 0 | 3 |

## 每折样本与图规模

| grid_degrees | fold | n_train | n_test | train_blocks | test_blocks | positive_train | positive_test | stable_edges | selected_graph_concepts | graph_residual_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.0000 | 1 | 1390 | 348 | 72 | 18 | 90 | 68 | 32 | 13 | 183 |
| 2.0000 | 2 | 1390 | 348 | 72 | 18 | 132 | 26 | 37 | 14 | 191 |
| 2.0000 | 3 | 1390 | 348 | 72 | 18 | 141 | 17 | 32 | 11 | 154 |
| 2.0000 | 4 | 1391 | 347 | 72 | 18 | 134 | 24 | 31 | 13 | 162 |
| 2.0000 | 5 | 1391 | 347 | 72 | 18 | 135 | 23 | 36 | 13 | 74 |
| 3.0000 | 1 | 1391 | 347 | 37 | 9 | 140 | 18 | 39 | 13 | 162 |
| 3.0000 | 2 | 1389 | 349 | 37 | 9 | 132 | 26 | 32 | 15 | 251 |
| 3.0000 | 3 | 1391 | 347 | 37 | 9 | 100 | 58 | 29 | 14 | 163 |
| 3.0000 | 4 | 1391 | 347 | 36 | 10 | 120 | 38 | 33 | 13 | 162 |
| 3.0000 | 5 | 1390 | 348 | 37 | 9 | 140 | 18 | 26 | 13 | 162 |