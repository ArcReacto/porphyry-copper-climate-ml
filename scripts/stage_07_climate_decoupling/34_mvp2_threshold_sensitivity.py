from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from decoupling_utils import (
    OUTPUT_DIR,
    climate_sensitive_element_columns,
    define_feature_roles,
    ensure_output_dir,
    evaluate_splitter,
    load_main_dataset,
    make_default_splitters,
    make_leave_one_group_splitter,
    summarize_metrics,
)


THRESHOLDS = [0.20, 0.30, 0.40]
OUT_DIR = OUTPUT_DIR / "threshold_sensitivity"
MODEL_KEYS = [
    "M1_Full_Climate",
    "M2_No_Climate",
    "M3_Climate_Normalized",
    "M4_Sensitive_Residualized_t0_20",
    "M4_Sensitive_Residualized_t0_30",
    "M4_Sensitive_Residualized_t0_40",
]


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No data._"
    display = df.head(max_rows).copy()
    headers = [str(c) for c in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in display.columns) + " |")
    return "\n".join(lines)


def compact_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in [
            "cv",
            "test_group",
            "model",
            "roc_auc_mean",
            "average_precision_mean",
            "balanced_accuracy_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean",
        ]
        if c in df.columns
    ]
    out = df[cols].copy()
    for col in out.columns:
        if col.endswith("_mean"):
            out[col] = out[col].round(4)
    return out


def run_standard_cv(df: pd.DataFrame, role_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_metrics = []
    all_predictions = []
    for cv_name, splitter, groups in make_default_splitters(df):
        print(f"Running {cv_name}")
        metrics, predictions = evaluate_splitter(
            df=df,
            role_table=role_table,
            splitter=splitter,
            cv_name=cv_name,
            groups=groups,
            model_keys=MODEL_KEYS,
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)
    return pd.concat(all_metrics, ignore_index=True), pd.concat(all_predictions, ignore_index=True)


def run_leave_region(df: pd.DataFrame, role_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_metrics = []
    all_predictions = []
    if "state" in df.columns:
        print("Running leave_one_state_out")
        splitter, groups = make_leave_one_group_splitter(df["state"])
        metrics, predictions = evaluate_splitter(
            df=df,
            role_table=role_table,
            splitter=splitter,
            cv_name="leave_one_state_out",
            groups=groups,
            model_keys=MODEL_KEYS,
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    if "env_causal_group_id" in df.columns:
        print("Running leave_one_environment_group_out")
        splitter, groups = make_leave_one_group_splitter(df["env_causal_group_id"])
        metrics, predictions = evaluate_splitter(
            df=df,
            role_table=role_table,
            splitter=splitter,
            cv_name="leave_one_environment_group_out",
            groups=groups,
            model_keys=MODEL_KEYS,
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    return pd.concat(all_metrics, ignore_index=True), pd.concat(all_predictions, ignore_index=True)


def write_report(
    role_table: pd.DataFrame,
    cv_summary: pd.DataFrame,
    leave_overall: pd.DataFrame,
    leave_group: pd.DataFrame,
) -> None:
    counts = []
    for threshold in THRESHOLDS:
        counts.append(
            {
                "threshold": threshold,
                "residualized_geochem_features": len(
                    climate_sensitive_element_columns(role_table, threshold=threshold)
                ),
                "raw_geochem_features": 362
                - len(climate_sensitive_element_columns(role_table, threshold=threshold)),
            }
        )
    counts_df = pd.DataFrame(counts)

    report = f"""# MVP2 气候敏感阈值敏感性实验

## 1. 实验目的

本实验比较 M4 在不同气候敏感阈值下的效果，判断 `|Spearman r| >= 0.3` 是否稳定。

比较模型：

| 模型 | 含义 |
|---|---|
| M1_Full_Climate | 原始地球化学 + 地质结构 + 气候变量 |
| M2_No_Climate | 原始地球化学 + 地质结构，不输入气候变量 |
| M3_Climate_Normalized | 全部地球化学变量做气候残差化 |
| M4_t0_20 | 只残差化 `|Spearman r| >= 0.20` 的地球化学变量 |
| M4_t0_30 | 只残差化 `|Spearman r| >= 0.30` 的地球化学变量 |
| M4_t0_40 | 只残差化 `|Spearman r| >= 0.40` 的地球化学变量 |

## 2. 不同阈值对应的残差化变量数量

{markdown_table(counts_df)}

## 3. 标准交叉验证结果

{markdown_table(compact_summary(cv_summary))}

## 4. 跨区域验证总体结果

{markdown_table(compact_summary(leave_overall))}

## 5. 跨区域验证分组结果

{markdown_table(compact_summary(leave_group), max_rows=80)}

## 6. 输出文件

```text
outputs/decoupled_climate/threshold_sensitivity/threshold_cv_metrics.csv
outputs/decoupled_climate/threshold_sensitivity/threshold_cv_summary.csv
outputs/decoupled_climate/threshold_sensitivity/threshold_leave_region_metrics.csv
outputs/decoupled_climate/threshold_sensitivity/threshold_leave_region_overall_summary.csv
outputs/decoupled_climate/threshold_sensitivity/threshold_leave_region_group_summary.csv
```
"""

    (OUT_DIR / "threshold_sensitivity_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_output_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_main_dataset()
    role_table = define_feature_roles(df)

    cv_metrics, cv_predictions = run_standard_cv(df, role_table)
    cv_summary = summarize_metrics(cv_metrics)

    leave_metrics, leave_predictions = run_leave_region(df, role_table)
    leave_overall = summarize_metrics(leave_metrics, group_cols=["cv", "model"])
    leave_group = summarize_metrics(leave_metrics, group_cols=["cv", "test_group", "model"])

    cv_metrics.to_csv(OUT_DIR / "threshold_cv_metrics.csv", index=False, encoding="utf-8-sig")
    cv_predictions.to_csv(OUT_DIR / "threshold_cv_predictions.csv", index=False, encoding="utf-8-sig")
    cv_summary.to_csv(OUT_DIR / "threshold_cv_summary.csv", index=False, encoding="utf-8-sig")
    leave_metrics.to_csv(OUT_DIR / "threshold_leave_region_metrics.csv", index=False, encoding="utf-8-sig")
    leave_predictions.to_csv(
        OUT_DIR / "threshold_leave_region_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave_overall.to_csv(
        OUT_DIR / "threshold_leave_region_overall_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave_group.to_csv(
        OUT_DIR / "threshold_leave_region_group_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "thresholds": THRESHOLDS,
        "model_keys": MODEL_KEYS,
        "note": "M4 variants residualize climate-sensitive geochemistry only; raw climate variables are not input to M4.",
    }
    (OUT_DIR / "threshold_sensitivity_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(role_table, cv_summary, leave_overall, leave_group)

    print("Wrote MVP2 threshold sensitivity results")
    print(OUT_DIR / "threshold_sensitivity_report.md")


if __name__ == "__main__":
    main()
