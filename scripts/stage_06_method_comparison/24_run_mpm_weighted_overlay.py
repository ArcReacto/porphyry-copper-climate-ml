from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json


INPUT_DATASET = "causal_graph/concept_features_western_core.parquet"
OUTPUT_DIR = "method_comparison"
METHOD_NAME = "mpm_weighted_overlay"
RANDOM_STATE = 20260622

WEIGHTS = {
    "concept_Cu_anomaly": 0.30,
    "concept_Mo_anomaly": 0.15,
    "concept_fault_density": 0.15,
    "concept_fault_proximity": 0.10,
    "concept_terrain_relief": 0.10,
    "concept_As_Sb_Bi_pathfinder": 0.10,
    "concept_Pb_Zn_background": 0.10,
}

FEATURE_DESCRIPTIONS = {
    "concept_Cu_anomaly": "主矿化信号",
    "concept_Mo_anomaly": "伴生斑岩铜矿化信号",
    "concept_fault_density": "构造背景强度",
    "concept_fault_proximity": "构造接近性",
    "concept_terrain_relief": "暴露、剥蚀或样本可见性条件",
    "concept_As_Sb_Bi_pathfinder": "找矿指示元素组合",
    "concept_Pb_Zn_background": "外围或背景异常",
}

META_COLUMNS = [
    "sample_id",
    "Y_label",
    "sample_type",
    "negative_type",
    "state",
    "latitude",
    "longitude",
    "dep_id",
    "mrds_id",
    "site_name",
    "env_aridity_class_3",
    "env_snow_influence",
    "env_relief_class",
    "env_water_deficit_class",
    "env_causal_group",
    "env_causal_group_id",
    "env_weathering_regime",
]

METRIC_COLUMNS = ["roc_auc", "average_precision", "balanced_accuracy", "precision", "recall", "f1"]


def robust_minmax(series: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> tuple[pd.Series, float, float]:
    x = pd.to_numeric(series, errors="coerce")
    lo = float(x.quantile(lower_q))
    hi = float(x.quantile(upper_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        filled = x.fillna(x.median())
        return pd.Series(np.zeros(len(filled)), index=series.index), lo, hi
    filled = x.fillna(x.median())
    scaled = ((filled - lo) / (hi - lo)).clip(0, 1)
    return scaled, lo, hi


def build_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [feature for feature in WEIGHTS if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing MPM input features: {missing}")

    out = df[[c for c in META_COLUMNS if c in df.columns]].copy()
    weight_rows = []
    score = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    for feature, weight in WEIGHTS.items():
        normalized, q05, q95 = robust_minmax(df[feature])
        norm_col = f"mpm_norm_{feature.replace('concept_', '')}"
        contrib_col = f"mpm_contrib_{feature.replace('concept_', '')}"
        out[feature] = df[feature]
        out[norm_col] = normalized
        out[contrib_col] = normalized * weight
        score = score + out[contrib_col]
        weight_rows.append(
            {
                "feature": feature,
                "description": FEATURE_DESCRIPTIONS.get(feature, ""),
                "weight": weight,
                "normalization": "robust_minmax_q05_q95",
                "q05": q05,
                "q95": q95,
                "direction": "higher_is_more_prospective",
            }
        )
    out["mpm_score"] = score
    weights = pd.DataFrame(weight_rows)
    return out, weights


def safe_score(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> float:
    if metric_name in {"roc_auc", "average_precision"} and len(np.unique(y_true)) < 2:
        return math.nan
    if metric_name == "roc_auc":
        return float(roc_auc_score(y_true, y_score))
    if metric_name == "average_precision":
        return float(average_precision_score(y_true, y_score))
    if metric_name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric_name == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric_name == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric_name == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    raise ValueError(metric_name)


def threshold_for_prevalence(scores: pd.Series, prevalence: float) -> float:
    positive_rate = min(max(float(prevalence), 0.0), 1.0)
    if positive_rate <= 0:
        return float("inf")
    if positive_rate >= 1:
        return float("-inf")
    return float(scores.quantile(1.0 - positive_rate))


def metric_row(
    cv_name: str,
    fold: int,
    y_true: pd.Series,
    y_score: pd.Series,
    threshold: float,
    n_train: int | None = None,
) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    row = {
        "method": METHOD_NAME,
        "cv": cv_name,
        "fold": fold,
        "n_train": int(n_train) if n_train is not None else np.nan,
        "n_test": int(len(y_true)),
        "positive_test": int(y_true.sum()),
        "negative_test": int((y_true == 0).sum()),
        "threshold": threshold,
        "roc_auc": safe_score("roc_auc", y_true.to_numpy(), y_pred.to_numpy(), y_score.to_numpy()),
        "average_precision": safe_score("average_precision", y_true.to_numpy(), y_pred.to_numpy(), y_score.to_numpy()),
        "balanced_accuracy": safe_score("balanced_accuracy", y_true.to_numpy(), y_pred.to_numpy(), y_score.to_numpy()),
        "precision": safe_score("precision", y_true.to_numpy(), y_pred.to_numpy(), y_score.to_numpy()),
        "recall": safe_score("recall", y_true.to_numpy(), y_pred.to_numpy(), y_score.to_numpy()),
        "f1": safe_score("f1", y_true.to_numpy(), y_pred.to_numpy(), y_score.to_numpy()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return row


def evaluate_scores(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = scores["Y_label"].astype(int)
    score = scores["mpm_score"]
    rows = []
    oof_parts = []

    global_threshold = threshold_for_prevalence(score, float(y.mean()))
    rows.append(metric_row("full_dataset_prevalence_threshold", 1, y, score, global_threshold, n_train=None))

    n_splits = min(5, int(y.sum()), int((y == 0).sum()))
    splitters = [("stratified_kfold", StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), None)]
    if "state" in scores.columns and scores["state"].nunique(dropna=True) >= 2:
        group_splits = min(5, scores["state"].nunique(dropna=True))
        splitters.append(("groupkfold_state", GroupKFold(n_splits=group_splits), scores["state"]))

    for cv_name, splitter, groups in splitters:
        split_iter = splitter.split(scores, y, groups) if groups is not None else splitter.split(scores, y)
        for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
            y_train = y.iloc[train_idx]
            train_scores = score.iloc[train_idx]
            threshold = threshold_for_prevalence(train_scores, float(y_train.mean()))
            y_test = y.iloc[test_idx]
            test_score = score.iloc[test_idx]
            rows.append(metric_row(cv_name, fold, y_test, test_score, threshold, n_train=len(train_idx)))
            oof_parts.append(
                pd.DataFrame(
                    {
                        "row_index": test_idx,
                        "cv": cv_name,
                        "fold": fold,
                        "Y_label": y_test.to_numpy(),
                        "mpm_score": test_score.to_numpy(),
                        "threshold": threshold,
                        "y_pred": (test_score >= threshold).astype(int).to_numpy(),
                    }
                )
            )
    metrics = pd.DataFrame(rows)
    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    return metrics, oof


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    fold_metrics = metrics[metrics["cv"] != "full_dataset_prevalence_threshold"].copy()
    summary = (
        fold_metrics.groupby(["method", "cv"], as_index=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else col for col in summary.columns]
    return summary


def group_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_col in ["Y_label", "state", "env_causal_group", "negative_type"]:
        if group_col not in scores.columns:
            continue
        for value, part in scores.groupby(group_col, dropna=False):
            if group_col not in {"Y_label", "negative_type"} and len(part) < 5:
                continue
            rows.append(
                {
                    "group_type": group_col,
                    "group_value": "(blank_positive)" if str(value).strip() == "" else value,
                    "n_rows": int(len(part)),
                    "positive": int(part["Y_label"].sum()),
                    "negative": int((part["Y_label"] == 0).sum()),
                    "mpm_score_mean": float(part["mpm_score"].mean()),
                    "mpm_score_median": float(part["mpm_score"].median()),
                    "mpm_score_q75": float(part["mpm_score"].quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def compare_with_ml(config: dict, mpm_summary: pd.DataFrame) -> pd.DataFrame:
    comparison_path = output_path(config, "outputs", f"{OUTPUT_DIR}/feature_space_comparison_with_causal_core.csv")
    if comparison_path.exists():
        ml = read_table(comparison_path)
    else:
        ml_path = output_path(config, "outputs", f"{OUTPUT_DIR}/ml_baseline_comparison.csv")
        ml = read_table(ml_path) if ml_path.exists() else pd.DataFrame()

    mpm_rows = mpm_summary.copy()
    mpm_rows = mpm_rows.rename(columns={"method": "model"})
    mpm_rows["dataset"] = "western_core_mpm_weighted_overlay_v1"
    mpm_rows["feature_space"] = "mpm_weighted_overlay"
    mpm_rows["feature_space_cn"] = "传统MPM加权叠加"
    shared_cols = sorted(set(ml.columns).union(mpm_rows.columns))
    combined = pd.concat([ml.reindex(columns=shared_cols), mpm_rows.reindex(columns=shared_cols)], ignore_index=True)
    return combined


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
        lines.append("| " + " | ".join([str(row[col]).replace("\n", " ") for col in cols]) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    scores: pd.DataFrame,
    weights: pd.DataFrame,
    summary: pd.DataFrame,
    group_stats: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    mpm_preferred = summary[summary["cv"] == "groupkfold_state"].copy()
    comparison_preferred = comparison[comparison["cv"] == "groupkfold_state"].copy()
    comparison_cols = [
        "feature_space_cn",
        "model",
        "roc_auc_mean",
        "average_precision_mean",
        "balanced_accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
    ]
    comparison_table = comparison_preferred[[c for c in comparison_cols if c in comparison_preferred.columns]].sort_values(
        "roc_auc_mean", ascending=False
    )
    group_table = group_stats[group_stats["group_type"].eq("Y_label")].copy()
    lines = [
        "# 方法对比实验三：传统 MPM 加权叠加",
        "",
        "## 1. 实验目的",
        "",
        "本实验用传统 mineral prospectivity mapping 思路构造一个人工规则评分，不训练模型，只按找矿概念变量加权叠加，检验传统规则评分与机器学习 baseline 的差异。",
        "",
        "## 2. 权重设置",
        "",
        markdown_table(weights[["feature", "description", "weight", "normalization", "direction"]]),
        "",
        "## 3. MPM 分数分布",
        "",
        f"- 样本数：{len(scores)}",
        f"- 正样本：{int(scores['Y_label'].sum())}",
        f"- 负样本：{int((scores['Y_label'] == 0).sum())}",
        "",
        markdown_table(group_table[["group_value", "n_rows", "mpm_score_mean", "mpm_score_median", "mpm_score_q75"]]),
        "",
        "## 4. groupkfold_state 评价结果",
        "",
        markdown_table(
            mpm_preferred[
                [
                    "method",
                    "roc_auc_mean",
                    "average_precision_mean",
                    "balanced_accuracy_mean",
                    "precision_mean",
                    "recall_mean",
                    "f1_mean",
                ]
            ]
        ),
        "",
        "## 5. 与 ML baseline 对比",
        "",
        markdown_table(comparison_table),
        "",
        "## 6. 解释边界",
        "",
        "MPM 加权叠加是人工规则评分，优点是透明、容易解释；缺点是权重来自经验设定，不能自动学习变量间非线性关系。它适合作为传统找矿远景预测基线，而不是因果证据。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    config = load_config()
    ensure_project_dirs(config)
    dataset_path = output_path(config, "outputs", INPUT_DATASET)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing concept feature dataset: {dataset_path}. "
            "Run scripts/stage_05_causal_graph/16_build_concept_features.py first."
        )

    df = read_table(dataset_path)
    scores, weights = build_scores(df)
    metrics, oof = evaluate_scores(scores)
    summary = summarize_metrics(metrics)
    group_stats = group_summary(scores)
    comparison = compare_with_ml(config, summary)

    scores_path = output_path(config, "outputs", f"{OUTPUT_DIR}/mpm_weighted_overlay_scores.csv")
    weights_path = output_path(config, "outputs", f"{OUTPUT_DIR}/mpm_weighted_overlay_feature_weights.csv")
    metrics_path = output_path(config, "outputs", f"{OUTPUT_DIR}/mpm_weighted_overlay_metrics.csv")
    summary_path = output_path(config, "outputs", f"{OUTPUT_DIR}/mpm_weighted_overlay_cv_summary.csv")
    oof_path = output_path(config, "outputs", f"{OUTPUT_DIR}/mpm_weighted_overlay_oof_predictions.csv")
    group_path = output_path(config, "outputs", f"{OUTPUT_DIR}/mpm_weighted_overlay_group_summary.csv")
    comparison_path = output_path(config, "outputs", f"{OUTPUT_DIR}/mpm_vs_ml_baseline_metrics.csv")
    report_path = output_path(config, "outputs", f"{OUTPUT_DIR}/mpm_weighted_overlay_report.md")

    write_dataframe(scores, scores_path)
    write_dataframe(weights, weights_path)
    write_dataframe(metrics, metrics_path)
    write_dataframe(summary, summary_path)
    write_dataframe(oof, oof_path)
    write_dataframe(group_stats, group_path)
    write_dataframe(comparison, comparison_path)
    write_report(report_path, scores, weights, summary, group_stats, comparison)

    run_summary = {
        "input_dataset": str(dataset_path),
        "method": METHOD_NAME,
        "rows": int(len(scores)),
        "positive": int(scores["Y_label"].sum()),
        "negative": int((scores["Y_label"] == 0).sum()),
        "weights": WEIGHTS,
        "outputs": {
            "scores": str(scores_path),
            "weights": str(weights_path),
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "oof_predictions": str(oof_path),
            "group_summary": str(group_path),
            "comparison": str(comparison_path),
            "report": str(report_path),
        },
    }
    write_json(output_path(config, "logs", "24_run_mpm_weighted_overlay_summary.json"), run_summary)

    print(f"Ran MPM weighted overlay: {len(scores)} rows, {len(WEIGHTS)} weighted features")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
