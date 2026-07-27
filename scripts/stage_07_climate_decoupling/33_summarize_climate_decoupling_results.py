from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from decoupling_utils import OUTPUT_DIR, ensure_output_dir


METRIC_COLUMNS = [
    "roc_auc_mean",
    "average_precision_mean",
    "balanced_accuracy_mean",
    "precision_mean",
    "recall_mean",
    "f1_mean",
]


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data generated yet._"
    display = df.head(max_rows).copy()
    headers = [str(c) for c in display.columns]
    rows = []
    for _, row in display.iterrows():
        rows.append([str(row[c]) for c in display.columns])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["cv", "test_group", "model", *METRIC_COLUMNS] if c in summary.columns]
    out = summary[cols].copy()
    for col in METRIC_COLUMNS:
        if col in out.columns:
            out[col] = out[col].round(4)
    return out


def main() -> None:
    out_dir = ensure_output_dir()
    profile_path = out_dir / "input_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}

    feature_roles = read_csv_if_exists(out_dir / "feature_role_table.csv")
    sensitive = read_csv_if_exists(out_dir / "top_climate_sensitive_elements.csv")
    cv_summary = read_csv_if_exists(out_dir / "climate_decoupling_cv_summary.csv")
    region_overall = read_csv_if_exists(out_dir / "leave_region_out_overall_summary.csv")
    region_group = read_csv_if_exists(out_dir / "leave_region_out_group_summary.csv")

    top_sensitive = sensitive.sort_values("spearman_abs_max", ascending=False).head(20) if not sensitive.empty else sensitive
    if not top_sensitive.empty:
        top_sensitive = top_sensitive[
            [
                "element_feature",
                "best_climate_feature_spearman",
                "spearman_r_at_best",
                "spearman_abs_max",
                "is_climate_sensitive",
            ]
        ].copy()
        top_sensitive["spearman_r_at_best"] = top_sensitive["spearman_r_at_best"].round(4)
        top_sensitive["spearman_abs_max"] = top_sensitive["spearman_abs_max"].round(4)

    role_counts = (
        feature_roles.groupby(["role", "is_numeric"]).size().reset_index(name="count")
        if not feature_roles.empty
        else pd.DataFrame()
    )

    report = f"""# 气候解耦阶段结果汇总

## 1. 输入数据

| 项目 | 数值 |
|---|---:|
| 总样本数 | {profile.get("n_rows", "")} |
| 正样本数 | {profile.get("n_positive", "")} |
| 负样本数 | {profile.get("n_negative", "")} |
| 总列数 | {profile.get("n_columns", "")} |
| M1 Full Climate 特征数 | {profile.get("m1_full_climate_feature_count", "")} |
| M2 No Climate 特征数 | {profile.get("m2_no_climate_feature_count", "")} |
| M3 残差化源特征数 | {profile.get("m3_climate_normalized_source_feature_count", "")} |
| 气候校正变量数 | {profile.get("climate_adjuster_count", "")} |

## 2. 变量角色统计

{markdown_table(role_counts)}

## 3. 元素-气候相关性 Top 20

{markdown_table(top_sensitive)}

说明：

- `spearman_abs_max` 越高，表示该元素观测值越可能随气候环境系统性变化。
- 当前阶段默认将 `|Spearman r| >= 0.3` 标记为气候敏感元素。

## 4. M1 / M2 / M3 / M4 全局对比

模型含义：

| 模型 | 输入 |
|---|---|
| M1_Full_Climate | 原始地球化学 + 地质/结构/地球物理/DEM + 气候变量 |
| M2_No_Climate | 原始地球化学 + 地质/结构/地球物理/DEM，不输入气候变量 |
| M3_Climate_Normalized | 气候残差化地球化学 + 地质/结构/地球物理/DEM，不输入原始气候变量 |
| M4_Sensitive_Residualized | 只对气候敏感地球化学变量做残差化，其余地球化学变量保留原始值，不输入原始气候变量 |

{markdown_table(compact_summary(cv_summary))}

## 5. 跨区域验证总体结果

{markdown_table(compact_summary(region_overall))}

## 6. 跨区域验证分组结果

{markdown_table(compact_summary(region_group), max_rows=40)}

## 7. 初步解读规则

| 结果形态 | 解释 |
|---|---|
| M3 接近 M1 且优于 M2 | 气候解耦较成功，气候主要用于校正地球化学信号 |
| M3 低于 M1 但跨区域更稳定 | 解耦有价值，但损失了一部分直接预测信息 |
| M3 明显低于 M1 和 M2 | 当前 Ridge 残差化可能过度校正或不足，需要尝试敏感元素残差化/非线性残差化 |
| M4 明显优于 M3 | 说明全量残差化过度校正，只校正气候敏感元素更合适 |
| M1 跨区域波动大而 M3 较稳 | 直接输入气候可能带来区域依赖，M3 更适合作为稳健模型 |

## 8. 相关输出文件

```text
outputs/decoupled_climate/feature_role_table.csv
outputs/decoupled_climate/input_profile.json
outputs/decoupled_climate/element_climate_correlation.csv
outputs/decoupled_climate/top_climate_sensitive_elements.csv
outputs/decoupled_climate/climate_decoupling_cv_metrics.csv
outputs/decoupled_climate/climate_decoupling_cv_summary.csv
outputs/decoupled_climate/climate_decoupling_sample_predictions.csv
outputs/decoupled_climate/leave_region_out_metrics.csv
outputs/decoupled_climate/leave_region_out_overall_summary.csv
outputs/decoupled_climate/leave_region_out_group_summary.csv
```
"""

    (out_dir / "climate_decoupling_summary_report.md").write_text(report, encoding="utf-8")
    print("Wrote climate decoupling summary report")
    print(out_dir / "climate_decoupling_summary_report.md")


if __name__ == "__main__":
    main()
