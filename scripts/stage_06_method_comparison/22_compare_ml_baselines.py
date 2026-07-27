from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json


OUTPUT_DIR = "method_comparison"
FULL_BASELINE_SUMMARY = "baseline_results/baseline_cv_summary_western_core_all_features_v1.csv"
CONCEPT_BASELINE_SUMMARY = "causal_graph/concept_baseline_cv_summary_western_core_concept_features_v1.csv"

DATASET_META = {
    "western_core_all_features_v1": {
        "feature_space": "all_features",
        "feature_space_cn": "全特征",
        "rows": 474,
        "positive": 158,
        "negative": 316,
        "features": 703,
        "description": "原始多源工程特征，作为预测性能上限参考。",
    },
    "western_core_concept_features_v1": {
        "feature_space": "concept_features",
        "feature_space_cn": "概念级特征",
        "rows": 474,
        "positive": 158,
        "negative": 316,
        "features": 49,
        "description": "由 703 个原始字段压缩得到的概念级特征，作为可解释版本参考。",
    },
}

METRIC_COLUMNS = [
    "roc_auc_mean",
    "average_precision_mean",
    "balanced_accuracy_mean",
    "precision_mean",
    "recall_mean",
    "f1_mean",
]


def markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    if df.empty:
        return "(empty)"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    cols = list(display.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_summary(path: Path, dataset_key: str) -> pd.DataFrame:
    df = read_table(path)
    if "index" in df.columns:
        df = df.drop(columns=["index"])
    meta = DATASET_META[dataset_key]
    df["dataset_key"] = dataset_key
    df["feature_space"] = meta["feature_space"]
    df["feature_space_cn"] = meta["feature_space_cn"]
    df["n_rows"] = meta["rows"]
    df["n_positive"] = meta["positive"]
    df["n_negative"] = meta["negative"]
    df["n_features"] = meta["features"]
    df["description"] = meta["description"]
    df["preferred_validation"] = df["cv"].eq("groupkfold_state")
    return df


def make_comparison(full: pd.DataFrame, concept: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([full, concept], ignore_index=True)
    front = [
        "feature_space",
        "feature_space_cn",
        "dataset_key",
        "cv",
        "model",
        "n_rows",
        "n_positive",
        "n_negative",
        "n_features",
        "preferred_validation",
    ]
    return combined[front + [c for c in combined.columns if c not in front]]


def make_delta_table(comparison: pd.DataFrame) -> pd.DataFrame:
    full = comparison[comparison["feature_space"] == "all_features"].copy()
    concept = comparison[comparison["feature_space"] == "concept_features"].copy()

    keep = ["cv", "model"] + METRIC_COLUMNS
    full = full[keep].rename(columns={c: f"all_features_{c}" for c in METRIC_COLUMNS})
    concept = concept[keep].rename(columns={c: f"concept_features_{c}" for c in METRIC_COLUMNS})
    out = full.merge(concept, on=["cv", "model"], how="inner")
    for metric in METRIC_COLUMNS:
        out[f"delta_concept_minus_all_{metric}"] = out[f"concept_features_{metric}"] - out[f"all_features_{metric}"]
        out[f"relative_retention_{metric}"] = out[f"concept_features_{metric}"] / out[f"all_features_{metric}"]
    return out.sort_values(["cv", "model"]).reset_index(drop=True)


def make_best_model_table(comparison: pd.DataFrame) -> pd.DataFrame:
    preferred = comparison[comparison["cv"] == "groupkfold_state"].copy()
    rows = []
    for feature_space, part in preferred.groupby("feature_space", sort=False):
        ranked = part.sort_values("roc_auc_mean", ascending=False)
        best = ranked.iloc[0]
        rows.append(
            {
                "feature_space": feature_space,
                "feature_space_cn": best["feature_space_cn"],
                "best_model_by_roc_auc": best["model"],
                "roc_auc_mean": best["roc_auc_mean"],
                "average_precision_mean": best["average_precision_mean"],
                "balanced_accuracy_mean": best["balanced_accuracy_mean"],
                "n_features": int(best["n_features"]),
            }
        )
    return pd.DataFrame(rows)


def write_report(path: Path, comparison: pd.DataFrame, delta: pd.DataFrame, best: pd.DataFrame) -> None:
    preferred = comparison[comparison["cv"] == "groupkfold_state"].copy()
    preferred_table = preferred[
        [
            "feature_space_cn",
            "model",
            "n_features",
            "roc_auc_mean",
            "average_precision_mean",
            "balanced_accuracy_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean",
        ]
    ].sort_values(["feature_space_cn", "roc_auc_mean"], ascending=[True, False])

    delta_preferred = delta[delta["cv"] == "groupkfold_state"].copy()
    delta_table = delta_preferred[
        [
            "model",
            "all_features_roc_auc_mean",
            "concept_features_roc_auc_mean",
            "delta_concept_minus_all_roc_auc_mean",
            "all_features_average_precision_mean",
            "concept_features_average_precision_mean",
            "delta_concept_minus_all_average_precision_mean",
            "all_features_balanced_accuracy_mean",
            "concept_features_balanced_accuracy_mean",
            "delta_concept_minus_all_balanced_accuracy_mean",
        ]
    ].sort_values("all_features_roc_auc_mean", ascending=False)

    lines = [
        "# 方法对比实验一：普通机器学习 baseline 对比",
        "",
        "## 1. 实验目的",
        "",
        "本实验对比两种特征表达方式在普通机器学习 baseline 下的预测表现：",
        "",
        "| 特征空间 | 特征数 | 含义 |",
        "|---|---:|---|",
        "| 全特征 | 703 | 原始多源工程特征，作为预测性能上限参考 |",
        "| 概念级特征 | 49 | 由原始字段压缩得到的可解释概念变量 |",
        "",
        "优先参考 `groupkfold_state`，因为它按州分组验证，更接近跨地区泛化。",
        "",
        "## 2. groupkfold_state 结果",
        "",
        markdown_table(preferred_table),
        "",
        "## 3. 最优模型对比",
        "",
        markdown_table(best),
        "",
        "## 4. 概念级特征相对全特征的性能变化",
        "",
        markdown_table(delta_table),
        "",
        "## 5. 初步结论",
        "",
        "1. 全特征模型仍然是预测性能上限参考，其中 Random Forest 在 `groupkfold_state` 下表现最好。",
        "2. 概念级特征模型有一定性能下降，但没有崩塌，说明 49 个概念变量仍保留了主要预测信号。",
        "3. Logistic Regression 的 ROC-AUC 几乎保持，说明概念级压缩对线性可分性的影响较小。",
        "4. 树模型在概念级特征上下降更明显，说明原始 703 个特征中的细尺度信息对非线性模型仍有额外帮助。",
        "5. 后续可以继续构建 `causal_core_features`，检验少量核心因果图特征是否仍有预测能力。",
        "",
        "## 6. 解释边界",
        "",
        "baseline 分数只说明当前特征对正负样本有预测可分性，不等同于因果证据。因果解释仍应优先参考分环境稳定性、候选因果图和知识图谱式解释结构。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config = load_config()
    ensure_project_dirs(config)

    full_path = output_path(config, "outputs", FULL_BASELINE_SUMMARY)
    concept_path = output_path(config, "outputs", CONCEPT_BASELINE_SUMMARY)
    if not full_path.exists():
        raise FileNotFoundError(f"Missing full-feature baseline summary: {full_path}")
    if not concept_path.exists():
        raise FileNotFoundError(f"Missing concept baseline summary: {concept_path}")

    full = load_summary(full_path, "western_core_all_features_v1")
    concept = load_summary(concept_path, "western_core_concept_features_v1")
    comparison = make_comparison(full, concept)
    delta = make_delta_table(comparison)
    best = make_best_model_table(comparison)

    comparison_path = output_path(config, "outputs", f"{OUTPUT_DIR}/ml_baseline_comparison.csv")
    delta_path = output_path(config, "outputs", f"{OUTPUT_DIR}/ml_baseline_feature_space_delta.csv")
    best_path = output_path(config, "outputs", f"{OUTPUT_DIR}/ml_baseline_best_models.csv")
    report_path = output_path(config, "outputs", f"{OUTPUT_DIR}/ml_baseline_comparison_report.md")

    write_dataframe(comparison, comparison_path)
    write_dataframe(delta, delta_path)
    write_dataframe(best, best_path)
    write_report(report_path, comparison, delta, best)

    summary = {
        "experiment": "method_comparison_01_ml_baseline",
        "inputs": {
            "full_features": str(full_path),
            "concept_features": str(concept_path),
        },
        "rows": {
            "comparison": int(len(comparison)),
            "delta": int(len(delta)),
            "best_models": int(len(best)),
        },
        "outputs": {
            "comparison": str(comparison_path),
            "delta": str(delta_path),
            "best_models": str(best_path),
            "report": str(report_path),
        },
    }
    write_json(output_path(config, "logs", "22_compare_ml_baselines_summary.json"), summary)

    print("Wrote method comparison experiment 1 outputs")
    print(comparison_path)
    print(delta_path)
    print(best_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

