# Rebuttal 新实验 N4：正样本空间代理簇 Top-K 评价

## 协议

- 输入：1:10 主实验的州折 OOF 预测；
- 模型：M2 NoClimate 与 M4 Spearman；
- 代理簇归并：5 km 球面距离簇；
- 同一空间簇在 Top-K 中重复出现时只计算一次代理簇命中；
- Top-K 仍按每个测试折的样本行数计算，随后汇总代理簇召回。

## 汇总结果

| model | budget_fraction | candidate_rows | positive_rows_hit | distinct_positive_deposits | distinct_deposits_hit | distinct_deposit_recall | distinct_deposit_precision_per_candidate | duplicate_positive_hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M4_Spearman_XGB | 0.0500 | 89 | 79 | 102 | 50 | 0.4902 | 0.5618 | 29 |
| M4_Spearman_XGB | 0.1000 | 175 | 121 | 102 | 76 | 0.7451 | 0.4343 | 45 |
| NoClimate_XGB | 0.0500 | 89 | 67 | 102 | 41 | 0.4020 | 0.4607 | 26 |
| NoClimate_XGB | 0.1000 | 175 | 114 | 102 | 68 | 0.6667 | 0.3886 | 46 |

## 解释边界

该指标减少邻近正样本记录重复计数的影响，但空间簇并不等同于权威矿床或成矿带边界。输出列名中的 `deposit` 是历史字段名；对外只能称为“正样本空间代理簇”，不能称为已验证的独立矿床覆盖。
