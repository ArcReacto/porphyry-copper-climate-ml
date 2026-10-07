# 古气候模型敏感性验证

## 固定协议

- 数据：1:10 hard-negative 论文主数据与古气候共同覆盖的 1729 个点；正样本 158 个。9 个 WNATA 范围外的负样本对所有配置统一移除。
- 交叉验证：11 州 GroupKFold 5 折；每折测试州、训练/测试样本对全部配置完全相同。
- 模型：xgboost；XGBoost 沿用主实验固定超参数，正类权重按训练折计算。Spearman 目标选择与 Ridge 残差器均只在训练折拟合。
- 古气候试算窗口：1700–1850 年；WNATA 为夏季最高气温距平均值，NASPA 为冷季降水均值。现代两变量是**年**最高气温均值和**年**降水总量，不是季节严格匹配对照。
- M2/M4 下游特征维持原有 385 个非气候字段。M4 的气候字段仅供敏感目标选择与残差化，不直接作为模型输入。

## 配置说明

| 配置 | 用法 |
|---|---|
| m2_no_climate | 385 个非气候特征，不做气候残差化 |
| m4_modern_24 | 原 24 个现代气候调节变量 |
| m4_modern_2 | 仅现代年最高气温与年降水 2 个调节变量 |
| m4_paleo_2 | 仅 2 个古气候调节变量 |
| m4_modern_plus_paleo_26 | 24 个现代变量加 2 个古气候变量共同调节 |
| m1_modern_full | 原现代气候入模的 662 特征集合 |
| m1_modern_plus_paleo | 662 特征基础上再加入 2 个古气候特征 |

## 逐折均值

| config | roc_auc_mean | average_precision_mean | f1_mean | top05_f1_mean | top05_ndcg_mean |
| --- | --- | --- | --- | --- | --- |
| m2_no_climate | 0.9471 | 0.7230 | 0.6570 | 0.5346 | 0.8059 |
| m4_modern_24 | 0.9519 | 0.7664 | 0.6520 | 0.6210 | 0.8909 |
| m4_modern_2 | 0.9409 | 0.7112 | 0.6450 | 0.5552 | 0.8031 |
| m4_paleo_2 | 0.9446 | 0.7032 | 0.6256 | 0.5388 | 0.7966 |
| m4_modern_plus_paleo_26 | 0.9513 | 0.7415 | 0.6520 | 0.5809 | 0.8281 |
| m1_modern_full | 0.9457 | 0.7041 | 0.6203 | 0.5606 | 0.7967 |
| m1_modern_plus_paleo | 0.9457 | 0.7044 | 0.6121 | 0.5517 | 0.7927 |

## 成对差值（候选配置减基线，按同折比较）

| baseline | candidate | metric | mean_delta | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- |
| m4_modern_2 | m4_paleo_2 | roc_auc | 0.0037 | 3 | 0 | 2 |
| m4_modern_2 | m4_paleo_2 | average_precision | -0.0079 | 4 | 0 | 1 |
| m4_modern_2 | m4_paleo_2 | top05_f1 | -0.0164 | 1 | 2 | 2 |
| m4_modern_2 | m4_paleo_2 | top05_ndcg | -0.0065 | 4 | 0 | 1 |
| m4_modern_24 | m4_paleo_2 | roc_auc | -0.0074 | 2 | 0 | 3 |
| m4_modern_24 | m4_paleo_2 | average_precision | -0.0631 | 2 | 0 | 3 |
| m4_modern_24 | m4_paleo_2 | top05_f1 | -0.0822 | 1 | 1 | 3 |
| m4_modern_24 | m4_paleo_2 | top05_ndcg | -0.0943 | 2 | 0 | 3 |
| m4_modern_24 | m4_modern_plus_paleo_26 | roc_auc | -0.0006 | 3 | 1 | 1 |
| m4_modern_24 | m4_modern_plus_paleo_26 | average_precision | -0.0248 | 1 | 0 | 4 |
| m4_modern_24 | m4_modern_plus_paleo_26 | top05_f1 | -0.0400 | 0 | 2 | 3 |
| m4_modern_24 | m4_modern_plus_paleo_26 | top05_ndcg | -0.0628 | 1 | 0 | 4 |
| m2_no_climate | m4_modern_24 | roc_auc | 0.0048 | 3 | 0 | 2 |
| m2_no_climate | m4_modern_24 | average_precision | 0.0434 | 3 | 0 | 2 |
| m2_no_climate | m4_modern_24 | top05_f1 | 0.0863 | 4 | 1 | 0 |
| m2_no_climate | m4_modern_24 | top05_ndcg | 0.0850 | 4 | 0 | 1 |
| m2_no_climate | m4_paleo_2 | roc_auc | -0.0026 | 3 | 1 | 1 |
| m2_no_climate | m4_paleo_2 | average_precision | -0.0198 | 2 | 0 | 3 |
| m2_no_climate | m4_paleo_2 | top05_f1 | 0.0041 | 1 | 3 | 1 |
| m2_no_climate | m4_paleo_2 | top05_ndcg | -0.0092 | 1 | 1 | 3 |
| m1_modern_full | m1_modern_plus_paleo | roc_auc | -0.0000 | 2 | 0 | 3 |
| m1_modern_full | m1_modern_plus_paleo | average_precision | 0.0003 | 2 | 0 | 3 |
| m1_modern_full | m1_modern_plus_paleo | top05_f1 | -0.0089 | 0 | 4 | 1 |
| m1_modern_full | m1_modern_plus_paleo | top05_ndcg | -0.0040 | 2 | 1 | 2 |

## 逐折 AP

| fold | test_group | m2_no_climate | m4_modern_24 | m4_modern_2 | m4_paleo_2 | m4_modern_plus_paleo_26 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Arizona | 0.7447 | 0.8117 | 0.7331 | 0.7363 | 0.8113 |
| 2 | Washington | 0.7063 | 0.8455 | 0.7034 | 0.6479 | 0.7939 |
| 3 | Nevada | 0.7298 | 0.6955 | 0.7229 | 0.7312 | 0.7079 |
| 4 | California;Colorado;New Mexico;Oregon | 0.5906 | 0.7244 | 0.5512 | 0.5541 | 0.6897 |
| 5 | Idaho;Montana;Utah;Wyoming | 0.8436 | 0.7547 | 0.8452 | 0.8467 | 0.7050 |

## 结果判断

- 古气候作为 M4 调节变量，相对 M2 的平均 AP 差为 -0.0198（2/5 折改善）；相对现代两变量 M4 的平均 AP 差为 -0.0079（4/5 折改善）。本窗口未显示稳定优势。
- 在原 24 个现代变量上叠加两个古气候变量，平均 AP 差为 -0.0248（1/5 折改善）。不能称为叠加提分。
- 将古气候特征直接加入 M1 全特征模型，平均 AP 差为 +0.0003（2/5 折改善）。这两个新增特征没有带来稳定排序收益。
- M4 的残差化目标数跨折均值：现代 24 变量 156.2，现代 2 变量 79.0，古气候 2 变量 31.0。在现代+古气候配置的 783 个折内被选目标记录里，31 个以古气候变量为最强相关调节变量；这是选择机制描述，不是因果归因。
- 初步结论限定为：在 1700–1850 年均值、两种不同季节的重建变量、当前样本与固定模型下，未观察到稳健提升。不能外推为古气候无用，也不能据此解决审稿人的地质时间尺度质疑。

## 解释边界

1. 五折均值是主口径；全体 OOF 合并指标另存于 `model_summary.json`，跨折分数未统一校准，不能直接替代逐折均值。
2. 只有 5 折，且各折州/正例数不均衡，胜折数和均值不能自动解释为统计显著。
3. 古气候与现代气候季节、时间尺度和变量定义不同。替换对照只是时间尺度敏感性检查，不能直接证明哪套气候更接近成矿时环境。
4. 古气候格点重建基于树轮和现代校准期；本实验没有真实古风化或矿床形成年代的监督标签，也不是成矿因果验证。
5. 源特征角色表沿用原标准化运行文件；新引入的古气候特征没有用全样本标签做筛选，折内仅做原方法的 Spearman 选择和训练。
