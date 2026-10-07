# Rebuttal 新实验 N3：同特征公平基线

## 协议

- 数据：1:10 hard-negative，1738 条；
- 验证：5 折 GroupKFold by state；
- 分类器、随机种子、缺失值处理与调参预算完全一致；
- 下游分类器：xgboost；XGBoost 正类权重按训练折计算，未在测试折调参；
- Geochemistry-only：312 个特征；
- Multimodal-no-climate：385 个特征；
- 气候调节变量：24 个，仅供残差器使用，不进入 M2/M4 下游分类器。

## 主结果

| feature_view | adjustment | roc_auc_mean | average_precision_mean | f1_mean | top05_f1_mean | top05_ndcg_mean | top10_f1_mean | top10_ndcg_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geochemistry_only | m3_broad | 0.9164 | 0.6488 | 0.5235 | 0.5545 | 0.7465 | 0.6683 | 0.7068 |
| geochemistry_only | m4_spearman | 0.9483 | 0.7665 | 0.6675 | 0.6117 | 0.8746 | 0.6921 | 0.7846 |
| geochemistry_only | raw | 0.9293 | 0.6949 | 0.6511 | 0.5621 | 0.8118 | 0.6442 | 0.7308 |
| multimodal_no_climate | m3_broad | 0.9365 | 0.7193 | 0.5313 | 0.5744 | 0.8195 | 0.6738 | 0.7515 |
| multimodal_no_climate | m4_spearman | 0.9546 | 0.7868 | 0.6730 | 0.6321 | 0.9039 | 0.7051 | 0.8025 |
| multimodal_no_climate | raw | 0.9448 | 0.7010 | 0.6583 | 0.5372 | 0.7831 | 0.6759 | 0.7478 |

## 成对差值

| contrast | metric | left_mean | right_mean | right_minus_left | right_better_folds | equal_folds | right_worse_folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| geochemistry: M4 minus raw | roc_auc | 0.9293 | 0.9483 | 0.0191 | 3 | 0 | 2 |
| geochemistry: M4 minus raw | average_precision | 0.6949 | 0.7665 | 0.0716 | 3 | 0 | 2 |
| geochemistry: M4 minus raw | f1 | 0.6511 | 0.6675 | 0.0165 | 2 | 0 | 3 |
| geochemistry: M4 minus raw | top05_f1 | 0.5621 | 0.6117 | 0.0496 | 3 | 1 | 1 |
| geochemistry: M4 minus raw | top05_ndcg | 0.8118 | 0.8746 | 0.0628 | 4 | 0 | 1 |
| geochemistry: M4 minus raw | top10_f1 | 0.6442 | 0.6921 | 0.0479 | 3 | 1 | 1 |
| geochemistry: M4 minus raw | top10_ndcg | 0.7308 | 0.7846 | 0.0538 | 4 | 0 | 1 |
| multimodal: M4 minus raw | roc_auc | 0.9448 | 0.9546 | 0.0098 | 3 | 0 | 2 |
| multimodal: M4 minus raw | average_precision | 0.7010 | 0.7868 | 0.0859 | 4 | 0 | 1 |
| multimodal: M4 minus raw | f1 | 0.6583 | 0.6730 | 0.0147 | 2 | 0 | 3 |
| multimodal: M4 minus raw | top05_f1 | 0.5372 | 0.6321 | 0.0949 | 4 | 1 | 0 |
| multimodal: M4 minus raw | top05_ndcg | 0.7831 | 0.9039 | 0.1207 | 5 | 0 | 0 |
| multimodal: M4 minus raw | top10_f1 | 0.6759 | 0.7051 | 0.0292 | 3 | 1 | 1 |
| multimodal: M4 minus raw | top10_ndcg | 0.7478 | 0.8025 | 0.0547 | 4 | 0 | 1 |
| raw: multimodal minus geochemistry | roc_auc | 0.9293 | 0.9448 | 0.0155 | 4 | 0 | 1 |
| raw: multimodal minus geochemistry | average_precision | 0.6949 | 0.7010 | 0.0061 | 2 | 0 | 3 |
| raw: multimodal minus geochemistry | f1 | 0.6511 | 0.6583 | 0.0073 | 3 | 0 | 2 |
| raw: multimodal minus geochemistry | top05_f1 | 0.5621 | 0.5372 | -0.0249 | 0 | 2 | 3 |
| raw: multimodal minus geochemistry | top05_ndcg | 0.8118 | 0.7831 | -0.0287 | 1 | 0 | 4 |
| raw: multimodal minus geochemistry | top10_f1 | 0.6442 | 0.6759 | 0.0317 | 4 | 0 | 1 |
| raw: multimodal minus geochemistry | top10_ndcg | 0.7308 | 0.7478 | 0.0170 | 3 | 0 | 2 |
| M4: multimodal minus geochemistry | roc_auc | 0.9483 | 0.9546 | 0.0063 | 3 | 1 | 1 |
| M4: multimodal minus geochemistry | average_precision | 0.7665 | 0.7868 | 0.0204 | 4 | 0 | 1 |
| M4: multimodal minus geochemistry | f1 | 0.6675 | 0.6730 | 0.0055 | 3 | 0 | 2 |
| M4: multimodal minus geochemistry | top05_f1 | 0.6117 | 0.6321 | 0.0204 | 3 | 2 | 0 |
| M4: multimodal minus geochemistry | top05_ndcg | 0.8746 | 0.9039 | 0.0293 | 4 | 0 | 1 |
| M4: multimodal minus geochemistry | top10_f1 | 0.6921 | 0.7051 | 0.0130 | 3 | 1 | 1 |
| M4: multimodal minus geochemistry | top10_ndcg | 0.7846 | 0.8025 | 0.0179 | 4 | 0 | 1 |

## 解释边界

本实验隔离了“特征模态增加”和“气候残差化”两个因素，但不等同于 Zhang et al. 或文献 [15] 的完整复现。外部方法只有在取得准确算法、代码和参数后才能以其论文名称报告。
