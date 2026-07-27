"""
脚本 28：对比"全量变量" vs "剔除气候变量"的找矿效果（普通随机森林，无欠采样）

与脚本 27 的区别：
- 使用 sklearn 的 RandomForestClassifier，不再使用 BalancedRandomForestClassifier
- 模型内部不再对负样本做 Bootstrap 欠采样
- 其余流程（特征分组、5 折分层 CV、样本级分类、气候诊断列）保持一致

输出：
outputs/method_comparison/climate_ablation/random_forest_no_undersampling/
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "outputs" / "model_datasets" / "model_dataset_western_core_all_features_v1.csv"
OUT_DIR = ROOT / "outputs" / "method_comparison" / "climate_ablation" / "random_forest_no_undersampling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 20260622
N_SPLITS = 5
DECISION_THRESHOLD = 0.5

CONFIG = {
    "target": "Y_label",
    "model_name": "RandomForestClassifier",
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "bootstrap": True,
    "oob_score": True,
    "balanced_data": False,
    "exclude_cols": [
        "sample_id", "Y_label", "sample_type", "negative_type", "state",
        "latitude", "longitude", "dep_id", "mrds_id", "site_name",
    ],
    "climate_prefixes": [
        "climate_",
        "env_climate_",
        "env_aridity",
        "env_water_deficit",
        "env_snow_influence",
        "env_relief_class",
        "env_weathering_regime",
        "env_causal_group_id",
    ],
}


# --------------------------------------------------------------------------- #
# 1. 数据读取与变量分组
# --------------------------------------------------------------------------- #
def load_dataset(path: Path) -> pd.DataFrame:
    logger.info("读取数据: %s", path)
    df = pd.read_csv(path)
    logger.info("数据 shape: %s, 正例=%d, 负例=%d",
                df.shape, int(df["Y_label"].sum()), int((df["Y_label"] == 0).sum()))
    return df


def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """分出全量特征与剔除气候变量后的特征。"""
    exclude = CONFIG["exclude_cols"] + [CONFIG["target"]]
    all_features = [c for c in df.columns if c not in exclude and df[c].dtype.kind in "fi"]

    climate_prefixes = CONFIG["climate_prefixes"]
    no_climate_features = [
        c for c in all_features
        if not any(c.startswith(p) or c.startswith(p.replace("_", "")) for p in climate_prefixes)
    ]
    # 额外排除 env_causal_group 等非数值/分组列
    no_climate_features = [
        c for c in no_climate_features
        if not c.startswith("env_causal_group")
    ]

    return {
        "full": all_features,
        "no_climate": no_climate_features,
    }


# --------------------------------------------------------------------------- #
# 2. 模型
# --------------------------------------------------------------------------- #
def make_model() -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=CONFIG["n_estimators"],
        max_depth=CONFIG["max_depth"],
        min_samples_leaf=CONFIG["min_samples_leaf"],
        max_features=CONFIG["max_features"],
        bootstrap=CONFIG["bootstrap"],
        oob_score=CONFIG["oob_score"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


# --------------------------------------------------------------------------- #
# 3. 评估
# --------------------------------------------------------------------------- #
def safe_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
    if metric_name in {"roc_auc", "average_precision"} and len(np.unique(y_true)) < 2:
        return float("nan")
    if metric_name == "roc_auc":
        return float(roc_auc_score(y_true, y_prob))
    if metric_name == "average_precision":
        return float(average_precision_score(y_true, y_prob))
    if metric_name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric_name == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric_name == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric_name == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    raise ValueError(metric_name)


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    oof_parts = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        logger.info("  Fold %d: train=%d (pos=%d, neg=%d), test=%d (pos=%d, neg=%d)",
                    fold,
                    len(train_idx), int(y_train.sum()), int((y_train == 0).sum()),
                    len(test_idx), int(y_test.sum()), int((y_test == 0).sum()))
        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        row = {
            "fold": fold,
            "roc_auc": safe_metric("roc_auc", y_test.to_numpy(), y_pred, y_prob),
            "average_precision": safe_metric("average_precision", y_test.to_numpy(), y_pred, y_prob),
            "balanced_accuracy": safe_metric("balanced_accuracy", y_test.to_numpy(), y_pred, y_pred),
            "precision": safe_metric("precision", y_test.to_numpy(), y_pred, y_prob),
            "recall": safe_metric("recall", y_test.to_numpy(), y_pred, y_prob),
            "f1": safe_metric("f1", y_test.to_numpy(), y_pred, y_prob),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        }
        logger.info("    -> AUC=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f",
                    row["roc_auc"], row["precision"], row["recall"], row["f1"])
        rows.append(row)
        oof_parts.append(pd.DataFrame({
            "row_index": test_idx,
            "Y_label": y_test.to_numpy(),
            "y_prob": y_prob,
            "y_pred": y_pred,
        }))

    metrics = pd.DataFrame(rows)
    oof = pd.concat(oof_parts, ignore_index=True).sort_values("row_index").reset_index(drop=True)

    # OOB on full data
    logger.info("  全数据训练获取 OOB AUC")
    model.fit(X, y)
    oob_proba = model.named_steps["model"].oob_decision_function_[:, 1]
    # OOB decision function 可能包含 NaN（对未被采样的样本），安全处理
    valid_mask = ~np.isnan(oob_proba)
    if valid_mask.sum() < 2 or len(np.unique(y[valid_mask])) < 2:
        oob_auc = float("nan")
    else:
        oob_auc = roc_auc_score(y[valid_mask], oob_proba[valid_mask])
    logger.info("    OOB AUC = %.4f", oob_auc)

    return metrics, oof, oob_auc


# --------------------------------------------------------------------------- #
# 4. 主流程
# --------------------------------------------------------------------------- #
def main():
    logger.info("===== 对比：全量变量 vs 剔除气候变量（普通随机森林，无欠采样） =====")
    df = load_dataset(INPUT_CSV)
    groups = classify_columns(df)
    logger.info("全量特征: %d, 剔除气候后特征: %d", len(groups["full"]), len(groups["no_climate"]))

    y = df[CONFIG["target"]].astype(int)
    meta = df[["sample_id", "latitude", "longitude", "state"]].copy()

    results = {}

    # Model Full
    logger.info("\n[Model Full] 全量变量，特征数=%d", len(groups["full"]))
    model_full = make_model()
    metrics_full, oof_full, oob_auc_full = evaluate_model(model_full, df[groups["full"]], y)
    results["Full"] = {
        "cv_metrics_mean": metrics_full.mean().round(4).to_dict(),
        "cv_metrics_std": metrics_full.std().round(4).to_dict(),
        "oob_auc": round(oob_auc_full, 4) if not np.isnan(oob_auc_full) else None,
    }

    # Model No_Climate
    logger.info("\n[Model No_Climate] 剔除气候变量，特征数=%d", len(groups["no_climate"]))
    model_no_climate = make_model()
    metrics_no_climate, oof_no_climate, oob_auc_no_climate = evaluate_model(
        model_no_climate, df[groups["no_climate"]], y
    )
    results["No_Climate"] = {
        "cv_metrics_mean": metrics_no_climate.mean().round(4).to_dict(),
        "cv_metrics_std": metrics_no_climate.std().round(4).to_dict(),
        "oob_auc": round(oob_auc_no_climate, 4) if not np.isnan(oob_auc_no_climate) else None,
    }

    # 样本级对比
    comparison = pd.DataFrame({
        "sample_id": meta["sample_id"].values,
        "Y_label": y.values,
        "prob_full": oof_full["y_prob"].values,
        "prob_no_climate": oof_no_climate["y_prob"].values,
    })
    comparison["pred_full"] = (comparison["prob_full"] >= DECISION_THRESHOLD).astype(int)
    comparison["pred_no_climate"] = (comparison["prob_no_climate"] >= DECISION_THRESHOLD).astype(int)

    def classify(row):
        full = row["pred_full"]
        no_clim = row["pred_no_climate"]
        y_true = row["Y_label"]
        if full == 1 and no_clim == 0 and y_true == 1:
            return "climate_reveals_new_mine"      # 气候变量帮助发现的新矿点
        if full == 0 and no_clim == 1 and y_true == 1:
            return "climate_hides_true_mine"       # 气候变量导致漏掉的矿点
        if full == 1 and no_clim == 0 and y_true == 0:
            return "climate_introduces_fp"         # 气候变量新增的假阳性
        if full == 0 and no_clim == 1 and y_true == 0:
            return "climate_corrects_fp"           # 气候变量纠正的假阳性
        if full == 1 and no_clim == 1 and y_true == 1:
            return "both_correct_mine"
        if full == 0 and no_clim == 0 and y_true == 1:
            return "both_miss_mine"
        if full == 1 and no_clim == 1 and y_true == 0:
            return "both_false_mine"
        return "both_correct_non_mine"

    comparison["classification"] = comparison.apply(classify, axis=1)
    comparison = comparison.merge(meta, on="sample_id", how="left")

    # 附加关键气候变量用于诊断
    climate_diag_cols = [c for c in df.columns if c.startswith(("env_aridity", "env_climate", "climate_"))]
    key_diag = [c for c in climate_diag_cols if "annual" in c or "aridity_index" in c or "deficit_annual" in c]
    diag = df[["sample_id"] + key_diag].copy()
    comparison = comparison.merge(diag, on="sample_id", how="left")

    # 保存
    metrics_full["model"] = "Full"
    metrics_no_climate["model"] = "No_Climate"
    pd.concat([metrics_full, metrics_no_climate], ignore_index=True).to_csv(
        OUT_DIR / "cv_metrics.csv", index=False
    )

    comparison.to_csv(OUT_DIR / "sample_classification.csv", index=False)

    summary = comparison["classification"].value_counts().reset_index()
    summary.columns = ["classification", "count"]
    summary.to_csv(OUT_DIR / "classification_summary.csv", index=False)

    report = {
        "config": CONFIG,
        "variable_counts": {
            "full": len(groups["full"]),
            "no_climate": len(groups["no_climate"]),
        },
        "model_comparison": results,
        "classification_counts": comparison["classification"].value_counts().to_dict(),
    }
    with open(OUT_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("\n===== 完成 =====")
    logger.info("输出目录: %s", OUT_DIR)
    logger.info("\n性能对比（5折 CV 平均）：")
    logger.info("  Model Full:        AUC=%.4f (±%.4f), Prec=%.4f, Rec=%.4f, F1=%.4f, OOB=%.4f",
                metrics_full["roc_auc"].mean(), metrics_full["roc_auc"].std(),
                metrics_full["precision"].mean(), metrics_full["recall"].mean(),
                metrics_full["f1"].mean(), oob_auc_full)
    logger.info("  Model No_Climate:  AUC=%.4f (±%.4f), Prec=%.4f, Rec=%.4f, F1=%.4f, OOB=%.4f",
                metrics_no_climate["roc_auc"].mean(), metrics_no_climate["roc_auc"].std(),
                metrics_no_climate["precision"].mean(), metrics_no_climate["recall"].mean(),
                metrics_no_climate["f1"].mean(), oob_auc_no_climate)

    logger.info("\n样本分类统计：")
    for cls, n in comparison["classification"].value_counts().items():
        logger.info("  %-30s: %d", cls, n)


if __name__ == "__main__":
    main()
