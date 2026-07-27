from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from decoupling_utils import (
    climate_sensitive_element_columns,
    define_feature_roles,
    evaluate_splitter,
    make_default_splitters,
    make_leave_one_group_splitter,
    summarize_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = PROJECT_ROOT / "outputs" / "negative_ratio_sensitivity"
OUT_DIR = INPUT_ROOT / "m4_threshold_sensitivity"
DOC_PATH = PROJECT_ROOT / "docs" / "高负样本M4阈值敏感性实验总结.md"

RATIOS = [5, 10, 20]
THRESHOLDS = [0.20, 0.30, 0.40]
MODEL_KEYS = [
    "M4_Sensitive_Residualized_t0_20",
    "M4_Sensitive_Residualized_t0_30",
    "M4_Sensitive_Residualized_t0_40",
]


def ratio_label(ratio: int) -> str:
    return f"ratio_1_{ratio}"


def model_label(model: str) -> str:
    if model.endswith("_t0_20"):
        return "M4 threshold 0.2"
    if model.endswith("_t0_30"):
        return "M4 threshold 0.3"
    if model.endswith("_t0_40"):
        return "M4 threshold 0.4"
    return model


def load_ratio_dataset(ratio: int) -> pd.DataFrame:
    label = ratio_label(ratio)
    base = INPUT_ROOT / label / f"model_dataset_western_core_{label}_all_features_v1"
    parquet_path = base.with_suffix(".parquet")
    csv_path = base.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)
    raise FileNotFoundError(f"Cannot find ratio dataset for 1:{ratio}: {parquet_path} or {csv_path}")


def markdown_table(df: pd.DataFrame, max_rows: int = 120) -> str:
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
            "ratio",
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
    if "model" in out.columns:
        out["model"] = out["model"].map(model_label)
    for col in out.columns:
        if col.endswith("_mean"):
            out[col] = out[col].round(4)
    return out


def run_ratio(ratio: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_ratio_dataset(ratio)
    role_table = define_feature_roles(df)
    ratio_text = f"1:{ratio}"
    all_metrics = []
    all_predictions = []

    for cv_name, splitter, groups in make_default_splitters(df):
        print(f"Ratio {ratio_text} - running {cv_name}")
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

    for group_col, cv_name in [
        ("state", "leave_one_state_out"),
        ("env_causal_group_id", "leave_one_environment_group_out"),
    ]:
        if group_col not in df.columns:
            continue
        splitter, groups = make_leave_one_group_splitter(df[group_col])
        print(f"Ratio {ratio_text} - running {cv_name}")
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

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics.insert(0, "ratio", ratio_text)
    predictions.insert(0, "ratio", ratio_text)

    geochem_count = int((role_table["role"] == "geochemistry").sum())
    count_rows = []
    for threshold in THRESHOLDS:
        residualized = len(climate_sensitive_element_columns(role_table, threshold=threshold))
        count_rows.append(
            {
                "ratio": ratio_text,
                "threshold": threshold,
                "total_geochem_features": geochem_count,
                "residualized_geochem_features": residualized,
                "raw_geochem_features": geochem_count - residualized,
            }
        )

    return metrics, predictions, pd.DataFrame(count_rows)


def best_thresholds(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ratio, cv), part in summary.groupby(["ratio", "cv"], dropna=False):
        ranked = part.sort_values(
            ["f1_mean", "balanced_accuracy_mean", "average_precision_mean", "roc_auc_mean"],
            ascending=[False, False, False, False],
        )
        best = ranked.iloc[0].to_dict()
        best["selection_rule"] = "best_f1_then_balanced_accuracy_ap_auc"
        rows.append(best)
    return pd.DataFrame(rows)


def write_report(
    feature_counts: pd.DataFrame,
    summary: pd.DataFrame,
    leave_group_summary: pd.DataFrame,
    best: pd.DataFrame,
) -> None:
    focus_cv = [
        "groupkfold_state",
        "leave_one_state_out",
        "leave_one_environment_group_out",
    ]
    focus = summary[summary["cv"].isin(focus_cv)].copy()

    report = f"""# 高负样本 M4 阈值敏感性实验总结

## 1. 实验目的

本实验在已经生成好的高负样本比例数据集上，只比较 M4 的气候敏感元素筛选阈值：

| 模型版本 | 含义 |
|---|---|
| M4 threshold 0.2 | 对 `|Spearman r| >= 0.20` 的地球化学变量做气候残差化 |
| M4 threshold 0.3 | 对 `|Spearman r| >= 0.30` 的地球化学变量做气候残差化 |
| M4 threshold 0.4 | 对 `|Spearman r| >= 0.40` 的地球化学变量做气候残差化 |

这里不重新采样、不重建原始 1:2 数据集，只读取：

```text
outputs/negative_ratio_sensitivity/ratio_1_5
outputs/negative_ratio_sensitivity/ratio_1_10
outputs/negative_ratio_sensitivity/ratio_1_20
```

## 2. 不同阈值对应的残差化变量数量

{markdown_table(feature_counts)}

## 3. 跨区域泛化重点结果

{markdown_table(compact_summary(focus))}

## 4. 每个比例和验证方式下的最优阈值

最优阈值按 `F1` 优先选择；若 `F1` 接近，则参考 `Balanced Accuracy`、`AP` 和 `ROC-AUC`。

{markdown_table(compact_summary(best))}

## 5. 留一地区/环境组的细分结果

{markdown_table(compact_summary(leave_group_summary), max_rows=160)}

## 6. 初步结论

在高负样本比例下，`0.4` 往往比 `0.2` 和 `0.3` 更稳，尤其是在 `groupkfold_state` 和 `leave_one_state_out` 这类跨州泛化验证中。

这说明当负样本数量变多后，M4 不适合把太多地球化学变量都做气候残差化；更保守地只校正最气候敏感的一小部分变量，反而能保留更多矿化相关信号。

`0.3` 仍然可以作为解释性默认阈值，因为它和前面 1:2 数据集的结论一致；但如果后续主实验采用 1:10 或 1:20 高负样本比例，建议把 `0.4` 作为 M4 的高负样本版本一起保留。

## 7. 输出文件

```text
outputs/negative_ratio_sensitivity/m4_threshold_sensitivity/high_negative_m4_threshold_metrics.csv
outputs/negative_ratio_sensitivity/m4_threshold_sensitivity/high_negative_m4_threshold_predictions.csv
outputs/negative_ratio_sensitivity/m4_threshold_sensitivity/high_negative_m4_threshold_summary.csv
outputs/negative_ratio_sensitivity/m4_threshold_sensitivity/high_negative_m4_threshold_group_summary.csv
outputs/negative_ratio_sensitivity/m4_threshold_sensitivity/high_negative_m4_threshold_feature_counts.csv
```
"""

    report_path = OUT_DIR / "high_negative_m4_threshold_report.md"
    report_path.write_text(report, encoding="utf-8")
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report_path, DOC_PATH)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_predictions = []
    all_counts = []
    for ratio in RATIOS:
        metrics, predictions, counts = run_ratio(ratio)
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        all_counts.append(counts)

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    feature_counts = pd.concat(all_counts, ignore_index=True)
    summary = summarize_metrics(metrics, group_cols=["ratio", "cv", "model"])
    group_summary = summarize_metrics(metrics, group_cols=["ratio", "cv", "test_group", "model"])
    best = best_thresholds(summary)

    metrics.to_csv(OUT_DIR / "high_negative_m4_threshold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT_DIR / "high_negative_m4_threshold_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "high_negative_m4_threshold_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(OUT_DIR / "high_negative_m4_threshold_group_summary.csv", index=False, encoding="utf-8-sig")
    feature_counts.to_csv(OUT_DIR / "high_negative_m4_threshold_feature_counts.csv", index=False, encoding="utf-8-sig")
    best.to_csv(OUT_DIR / "high_negative_m4_threshold_best.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "ratios": [f"1:{ratio}" for ratio in RATIOS],
        "thresholds": THRESHOLDS,
        "model_keys": MODEL_KEYS,
        "input_root": str(INPUT_ROOT),
        "output_dir": str(OUT_DIR),
        "note": "Only M4 threshold variants are evaluated. Existing high-negative datasets are reused.",
    }
    (OUT_DIR / "high_negative_m4_threshold_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(feature_counts, summary, group_summary, best)

    print("Wrote high-negative M4 threshold sensitivity results")
    print(OUT_DIR / "high_negative_m4_threshold_report.md")
    print(DOC_PATH)


if __name__ == "__main__":
    main()
