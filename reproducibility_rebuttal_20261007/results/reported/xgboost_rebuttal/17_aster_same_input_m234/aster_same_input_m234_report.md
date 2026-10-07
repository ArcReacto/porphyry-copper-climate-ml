# Same-input ASTER M2/M3/M4 对照实验

## 1. 实验目的

在完全相同的样本、ASTER/LULC输入、XGBoost分类器和按州五折划分下，只改变气候调整方式，检验选择性气候残差化能否迁移到ASTER观测配置。

## 2. 数据

- 样本：1735（正样本 145，负样本 1590，11州）；
- ASTER：31个基础特征（VNIR 7、SWIR 11、TIR 13）；
- LULC：48个多尺度土地覆盖特征；
- 气候调节变量：24个，仅供残差器使用，不进入下游XGBoost；
- ASTER/LULC/气候平均缺失率：0.1399/0.1264/0.0000。

## 3. 实验配置

- 验证：5折 GroupKFold by state；
- 分类器：XGBoost，400棵树，learning_rate=0.03，max_depth=3，min_child_weight=3，subsample=0.85，colsample_bytree=0.85，reg_lambda=2.0；
- 类别权重：每个训练折内根据正负样本数计算；
- M4选择阈值：训练折内 ASTER 特征与24个气候变量的最大绝对 Spearman 相关系数 >= 0.30；
- 残差器：训练折内中位数填补、气候标准化、Ridge(alpha=10)；
- 测试折仅用于变换和评价，未用于填补、选择、残差器或模型拟合。

## 4. 方法

- **M2_ASTER_RAW_XGB**：31个原始ASTER特征 + 48个LULC特征；
- **M3_ASTER_BROAD_XGB**：对全部31个ASTER特征做气候残差化，LULC保持原值；
- **M4_ASTER_SPEARMAN_XGB**：仅残差化训练折内选中的ASTER特征，未选中的ASTER与全部LULC保持原值。

三种配置均向XGBoost提供79个下游特征，区别仅为ASTER列使用原值还是气候残差值。

## 5. 逐折划分与选择数量

| fold | test_group | n_train | n_test | positive_train | positive_test | m4_selected_aster_features |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Arizona | 1063 | 672 | 89 | 56 | 18 |
| 2 | Washington | 1318 | 417 | 108 | 37 | 8 |
| 3 | Nevada | 1414 | 321 | 124 | 21 | 22 |
| 4 | Colorado;New Mexico;Utah;Wyoming | 1578 | 157 | 127 | 18 | 22 |
| 5 | California;Idaho;Montana;Oregon | 1567 | 168 | 132 | 13 | 22 |

M4逐折选中特征数量：

| fold | test_group | selected | total |
| --- | --- | --- | --- |
| 1 | Arizona | 18 | 31 |
| 2 | Washington | 8 | 31 |
| 3 | Nevada | 22 | 31 |
| 4 | Colorado;New Mexico;Utah;Wyoming | 22 | 31 |
| 5 | California;Idaho;Montana;Oregon | 22 | 31 |

## 6. 主结果

| model | roc_auc_mean | roc_auc_std | average_precision_mean | average_precision_std | f1_mean | top05_f1_mean | top05_ndcg_mean | top10_f1_mean | top10_ndcg_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2_ASTER_RAW_XGB | 0.7209 | 0.1173 | 0.2863 | 0.1607 | 0.2905 | 0.2549 | 0.3823 | 0.3014 | 0.3622 |
| M3_ASTER_BROAD_XGB | 0.7268 | 0.0876 | 0.3017 | 0.1422 | 0.2932 | 0.2995 | 0.4308 | 0.2931 | 0.3647 |
| M4_ASTER_SPEARMAN_XGB | 0.6898 | 0.1672 | 0.2747 | 0.1575 | 0.2700 | 0.2416 | 0.3743 | 0.2646 | 0.3359 |

## 7. 相对M2的成对结果

| comparison | metric | m2_mean | candidate_mean | mean_difference | paired_bootstrap_ci95_low | paired_bootstrap_ci95_high | better_folds | equal_folds | worse_folds | fold_differences |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3_ASTER_BROAD_XGB minus M2_ASTER_RAW_XGB | average_precision | 0.2863 | 0.3017 | 0.0154 | -0.0117 | 0.0508 | 2 | 0 | 3 | [-0.021987, 0.07685, 0.029915, -0.00157, -0.006457] |
| M3_ASTER_BROAD_XGB minus M2_ASTER_RAW_XGB | top05_f1 | 0.2549 | 0.2995 | 0.0446 | 0.0000 | 0.1061 | 2 | 3 | 0 | [0.0, 0.068966, 0.0, 0.153846, 0.0] |
| M3_ASTER_BROAD_XGB minus M2_ASTER_RAW_XGB | top05_ndcg | 0.3823 | 0.4308 | 0.0485 | 0.0099 | 0.0870 | 4 | 1 | 0 | [0.00482, 0.09764, 0.03998, 0.099897, 0.0] |
| M4_ASTER_SPEARMAN_XGB minus M2_ASTER_RAW_XGB | average_precision | 0.2863 | 0.2747 | -0.0116 | -0.0202 | 0.0022 | 1 | 0 | 4 | [-0.019091, -0.022661, 0.015239, -0.01748, -0.013973] |
| M4_ASTER_SPEARMAN_XGB minus M2_ASTER_RAW_XGB | top05_f1 | 0.2549 | 0.2416 | -0.0133 | -0.0400 | 0.0000 | 0 | 4 | 1 | [-0.066667, 0.0, 0.0, 0.0, 0.0] |
| M4_ASTER_SPEARMAN_XGB minus M2_ASTER_RAW_XGB | top05_ndcg | 0.3823 | 0.3743 | -0.0080 | -0.0387 | 0.0128 | 2 | 1 | 2 | [-0.067261, -0.004732, 0.013027, 0.018836, 0.0] |

区间为对5个测试折的配对差值进行10,000次 bootstrap 得到的描述性95%区间。折数仅为5，区间只用于展示不确定性，不作高功效显著性检验。

## 8. 结果分析与结论

1. **M3全量残差化在平均指标上略优于M2，但不是所有地区都改善。** ROC-AUC由0.7209升至0.7268，AP由0.2863升至0.3017（差值+0.0154），Top-5% F1由0.2549升至0.2995（差值+0.0446）。AP只在2/5折提高，主要收益来自Washington和Nevada，说明效果具有明显地区异质性。
2. **M4选择性残差化未在这套基础ASTER输入上复现主实验优势。** 相对M2，M4的AP差值为-0.0116，Top-5% F1差值为-0.0133；AP仅1/5折提高。选择性调整不是跨输入模态必然增益的方法。
3. **该实验支持“气候调整效果依赖观测配置”，而不支持“所有ASTER配置上M4稳定最佳”。** 可在rebuttal中把它作为同输入、同模型、同折分的公平性补充，并如实报告M3的小幅平均收益和M4的负结果。
4. **对方法主张的影响是收缩适用范围，而非否定训练折内调整框架。** 结合既有地球化学/多模态结果，更稳妥的结论是：选择性调整在部分地学特征体系和指标上有效，但需要按观测模态验证，不能把一个选择规则直接视为普适最优。

## 9. 解释边界

本实验验证的是31维基础ASTER观测配置，不是此前611维 `ASTER_enhanced` 配置。因此，它可以回答“气候残差化是否能在另一类ASTER输入上工作”，但不能直接替代611维增强特征的同输入对照。SWIR和LULC存在缺失；处理均在训练折内完成，三种模型使用相同样本和相同输入字段。
