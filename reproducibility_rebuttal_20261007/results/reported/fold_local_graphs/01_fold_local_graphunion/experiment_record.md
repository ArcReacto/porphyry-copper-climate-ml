# 实验 01：无泄漏的每折 GraphUnion

## 基本信息

| 字段 | 内容 |
|---|---|
| 当前状态 | 已完成，结果需用于修订论文主张 |
| 优先级 | P0，必做 |
| 计划脚本 | `scripts/stage_16_rebuttal_experiments/01_fold_local_graphunion.py` |
| 输出目录 | `outputs/rebuttal_experiments/01_fold_local_graphunion/` |
| 主数据集 | `known_mining_neutral_ratio_1_10_supervised_all_features_v1` |

## 目的

验证 GraphUnion 在严格归纳设置下是否仍有增益。每一折只能用训练折重建概念相关、敏感率/稳定边、图目标集合和残差器，测试折不能参与任何统计量计算。

## 计划协议

| 项目 | 设定 |
|---|---|
| 验证 | 5 折 GroupKFold by state |
| 对照 | M2 NoClimate、M4 Spearman、M4 GraphUnion fold-local |
| 主指标 | AP、F1@5%、NDCG@5%、F1@10%、NDCG@10% |
| 防泄漏检查 | 每折保存训练州、测试州、图节点/边、选中特征及其数据来源 |

## 预期输出

`fold_graph_manifest.csv`、`fold_selected_targets.csv`、`fold_metrics.csv`、`oof_predictions.csv`、`paired_comparison.csv`、`run_manifest.json`、`fold_local_graphunion_report.md`。

## 最终结果

| 模型 | ROC-AUC | AP | F1 | F1@5% | NDCG@5% | F1@10% | NDCG@10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2 NoClimate RF | 0.9506 | 0.7652 | 0.6562 | 0.6023 | 0.8789 | 0.6699 | 0.7759 |
| M4 Spearman RF | 0.9555 | 0.7631 | 0.6593 | 0.6301 | 0.8858 | 0.6997 | 0.7842 |
| M4 GraphUnion fold-local RF | 0.9493 | 0.7468 | 0.5763 | 0.5949 | 0.8446 | 0.6767 | 0.7553 |

结论：M4 Spearman 的结果与原主实验一致，说明严格折内的选择性残差化结果可复现；GraphUnion 在每折重构图后 AP、普通 F1 和多数 Top-K 指标低于 M2，原固定图的优势不能作为严格无泄漏证据。后续实验将 M2 与 M4 Spearman 作为主比较，fold-local GraphUnion 仅保留为探索性对照。

质量检查：三个模型各有 1738 条 OOF 记录，以 `row_index + model` 检查共 5214 个唯一组合，无缺失预测。`sample_id` 本身存在跨州重复编号，已转交实验 06 做全局唯一键和矿区级审计。

## 运行记录

| 时间 | Git commit | 命令 | 状态 | 结论/问题 |
|---|---|---|---|---|
| 2026-09-22 | 当前工作树 | `.venv/Scripts/python.exe scripts/stage_16_rebuttal_experiments/01_fold_local_graphunion.py` | 已完成 | GraphUnion 原增益未保留；Spearman 结果稳定 |
