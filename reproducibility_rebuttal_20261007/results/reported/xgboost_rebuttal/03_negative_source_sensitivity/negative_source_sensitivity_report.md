# Rebuttal 实验 03：负样本来源敏感性

`background_only` 使用未标注背景点作为负类代理，仅用于标签方案敏感性分析，不把这些点解释为已证实无矿。三个方案均保持 158:1580 的 1:10 类别比例。

## 样本来源

| negative_scheme | negative_source_scheme | Y_label | rows |
| --- | --- | --- | --- |
| background_only | background_proxy | 0 | 1580 |
| background_only | nan | 1 | 158 |
| hard_only | hard_mrds | 0 | 1580 |
| hard_only | nan | 1 | 158 |
| mixed_50_50 | background_proxy | 0 | 790 |
| mixed_50_50 | hard_mrds | 0 | 790 |
| mixed_50_50 | nan | 1 | 158 |

## 模型结果

| negative_scheme | model | roc_auc_mean | average_precision_mean | f1_mean | top05_f1_mean | top05_ndcg_mean | top10_f1_mean | top10_ndcg_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| background_only | M4_Spearman_XGB | 0.9885 | 0.9449 | 0.8766 | 0.7235 | 1.0000 | 0.8792 | 0.9476 |
| background_only | NoClimate_XGB | 0.9903 | 0.9528 | 0.8973 | 0.7235 | 1.0000 | 0.8694 | 0.9417 |
| hard_only | M4_Spearman_XGB | 0.9546 | 0.7868 | 0.6730 | 0.6321 | 0.9039 | 0.7051 | 0.8025 |
| hard_only | NoClimate_XGB | 0.9448 | 0.7010 | 0.6583 | 0.5372 | 0.7831 | 0.6759 | 0.7478 |
| mixed_50_50 | M4_Spearman_XGB | 0.9588 | 0.7982 | 0.7170 | 0.6334 | 0.8972 | 0.7421 | 0.8243 |
| mixed_50_50 | NoClimate_XGB | 0.9585 | 0.7560 | 0.7559 | 0.6208 | 0.8311 | 0.7670 | 0.8005 |

## M4 Spearman 相对 M2

| negative_scheme | metric | baseline_mean | candidate_mean | mean_difference | better_folds | equal_folds | worse_folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| background_only | average_precision | 0.9528 | 0.9449 | -0.0079 | 3 | 0 | 2 |
| background_only | f1 | 0.8973 | 0.8766 | -0.0207 | 2 | 0 | 3 |
| background_only | top05_f1 | 0.7235 | 0.7235 | 0.0000 | 0 | 5 | 0 |
| background_only | top05_ndcg | 1.0000 | 1.0000 | 0.0000 | 0 | 5 | 0 |
| background_only | top10_f1 | 0.8694 | 0.8792 | 0.0098 | 3 | 1 | 1 |
| background_only | top10_ndcg | 0.9417 | 0.9476 | 0.0058 | 3 | 0 | 2 |
| hard_only | average_precision | 0.7010 | 0.7868 | 0.0859 | 4 | 0 | 1 |
| hard_only | f1 | 0.6583 | 0.6730 | 0.0147 | 2 | 0 | 3 |
| hard_only | top05_f1 | 0.5372 | 0.6321 | 0.0949 | 4 | 1 | 0 |
| hard_only | top05_ndcg | 0.7831 | 0.9039 | 0.1207 | 5 | 0 | 0 |
| hard_only | top10_f1 | 0.6759 | 0.7051 | 0.0292 | 3 | 1 | 1 |
| hard_only | top10_ndcg | 0.7478 | 0.8025 | 0.0547 | 4 | 0 | 1 |
| mixed_50_50 | average_precision | 0.7560 | 0.7982 | 0.0422 | 4 | 0 | 1 |
| mixed_50_50 | f1 | 0.7559 | 0.7170 | -0.0389 | 0 | 0 | 5 |
| mixed_50_50 | top05_f1 | 0.6208 | 0.6334 | 0.0126 | 1 | 2 | 2 |
| mixed_50_50 | top05_ndcg | 0.8311 | 0.8972 | 0.0661 | 3 | 0 | 2 |
| mixed_50_50 | top10_f1 | 0.7670 | 0.7421 | -0.0249 | 2 | 2 | 1 |
| mixed_50_50 | top10_ndcg | 0.8005 | 0.8243 | 0.0237 | 4 | 0 | 1 |