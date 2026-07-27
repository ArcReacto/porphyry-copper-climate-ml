from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json


INPUT_DATASET = "causal_graph/concept_features_western_core.parquet"
OUTPUT_DIR = "causal_graph"
DATASET_NAME = "western_core_concept_features_v1"
RANDOM_STATE = 20260622


def make_models(pos_weight: float) -> dict[str, Pipeline]:
    return {
        "dummy_stratified": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=200,
                        max_leaf_nodes=15,
                        l2_regularization=0.1,
                        class_weight={0: 1.0, 1: pos_weight},
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def safe_score(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
    if metric_name in {"roc_auc", "average_precision"} and len(np.unique(y_true)) < 2:
        return math.nan
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


def predict_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model[-1], "decision_function"):
        score = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-score))
    return model.predict(x).astype(float)


def evaluate_cv(
    model_name: str,
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    splitter,
    cv_name: str,
    groups: pd.Series | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    rows = []
    oof_parts = []
    split_iter = splitter.split(x, y, groups) if groups is not None else splitter.split(x, y)
    for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = predict_probability(model, x_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "dataset": DATASET_NAME,
                "cv": cv_name,
                "model": model_name,
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "positive_test": int(y_test.sum()),
                "negative_test": int((y_test == 0).sum()),
                "roc_auc": safe_score("roc_auc", y_test.to_numpy(), y_pred, y_prob),
                "average_precision": safe_score("average_precision", y_test.to_numpy(), y_pred, y_prob),
                "balanced_accuracy": safe_score("balanced_accuracy", y_test.to_numpy(), y_pred, y_prob),
                "precision": safe_score("precision", y_test.to_numpy(), y_pred, y_prob),
                "recall": safe_score("recall", y_test.to_numpy(), y_pred, y_prob),
                "f1": safe_score("f1", y_test.to_numpy(), y_pred, y_prob),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
        oof_parts.append(
            pd.DataFrame(
                {
                    "row_index": test_idx,
                    "dataset": DATASET_NAME,
                    "cv": cv_name,
                    "model": model_name,
                    "Y_label": y_test.to_numpy(),
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                }
            )
        )
    return rows, pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else col for col in out.columns]
    return out


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["roc_auc", "average_precision", "balanced_accuracy", "precision", "recall", "f1"]
    return flatten_columns(
        metrics.groupby(["dataset", "cv", "model"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )


def group_error_report(oof: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if oof.empty:
        return pd.DataFrame()
    merged = oof.merge(meta.reset_index().rename(columns={"index": "row_index"}), on="row_index", how="left")
    rows = []
    group_cols = [c for c in ["negative_type", "state", "env_causal_group"] if c in merged.columns]
    for group_col in group_cols:
        for (dataset, cv_name, model_name, group_value), part in merged.groupby(
            ["dataset", "cv", "model", group_col], dropna=False
        ):
            if group_col != "negative_type" and len(part) < 5:
                continue
            negatives = int((part["Y_label"] == 0).sum())
            rows.append(
                {
                    "dataset": dataset,
                    "cv": cv_name,
                    "model": model_name,
                    "group_type": group_col,
                    "group_value": "(blank_positive)" if str(group_value).strip() == "" else group_value,
                    "n_rows": int(len(part)),
                    "n_negative": negatives,
                    "false_positive_rate": float(((part["Y_label"] == 0) & (part["y_pred"] == 1)).sum())
                    / max(1, negatives),
                    "mean_predicted_probability": float(part["y_prob"].mean()),
                }
            )
    return pd.DataFrame(rows)


def feature_importance(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    permutation_repeats: int,
) -> pd.DataFrame:
    model.fit(x, y)
    rf = model.named_steps["model"]
    out = pd.DataFrame({"feature": x.columns, "rf_gini_importance": rf.feature_importances_})
    if permutation_repeats > 0:
        perm = permutation_importance(
            model,
            x,
            y,
            scoring="roc_auc",
            n_repeats=permutation_repeats,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        out["permutation_importance_mean"] = perm.importances_mean
        out["permutation_importance_std"] = perm.importances_std
    else:
        out["permutation_importance_mean"] = np.nan
        out["permutation_importance_std"] = np.nan
    out["dataset"] = DATASET_NAME
    sort_cols = ["rf_gini_importance"] if permutation_repeats <= 0 else ["permutation_importance_mean", "rf_gini_importance"]
    return out.sort_values(sort_cols, ascending=False)


def write_report(
    path: Path,
    df: pd.DataFrame,
    feature_cols: list[str],
    summary: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    display_cols = [
        "dataset",
        "cv",
        "model",
        "roc_auc_mean",
        "average_precision_mean",
        "balanced_accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
    ]
    available = [c for c in display_cols if c in summary.columns]
    top_cols = ["feature", "rf_gini_importance", "permutation_importance_mean"]
    lines = [
        f"Concept baseline report: {DATASET_NAME}",
        f"Rows: {len(df)}",
        f"Concept features: {len(feature_cols)}",
        f"Positive labels: {int(df['Y_label'].sum())}",
        f"Negative labels: {int((df['Y_label'] == 0).sum())}",
        "",
        "Important safeguards:",
        "- Only concept_* columns are used as model inputs.",
        "- Coordinates, state, sample IDs, negative_type, and env_* fields are metadata only.",
        "- Scores are diagnostics for concept compression, not causal evidence.",
        "",
        "Mean cross-validation metrics:",
        summary[available].round(4).to_string(index=False),
        "",
        "Top concept features by Random Forest importance:",
        importance[[c for c in top_cols if c in importance.columns]].head(20).round(5).to_string(index=False),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline models on concept-level causal graph features.")
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=3,
        help="Permutation importance repeats. Use 0 for faster runs with only Random Forest impurity importance.",
    )
    parser.add_argument("--skip-permutation", action="store_true", help="Shortcut for --permutation-repeats 0.")
    args = parser.parse_args()

    config = load_config()
    ensure_project_dirs(config)
    dataset_path = output_path(config, "outputs", INPUT_DATASET)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing concept feature dataset: {dataset_path}. Run scripts/stage_05_causal_graph/16_build_concept_features.py first.")

    df = read_table(dataset_path)
    feature_cols = sorted([c for c in df.columns if c.startswith("concept_")])
    if not feature_cols:
        raise ValueError("No concept_* feature columns found.")
    x = df[feature_cols]
    y = df["Y_label"].astype(int)
    meta_cols = [c for c in ["negative_type", "state", "env_causal_group"] if c in df.columns]
    meta = df[meta_cols].copy()

    pos = int(y.sum())
    neg = int((y == 0).sum())
    pos_weight = neg / max(pos, 1)
    n_splits = min(5, pos, neg)
    if n_splits < 2:
        raise ValueError("Not enough positive/negative samples for CV.")

    splitters = [("stratified_kfold", StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), None)]
    if "state" in df.columns and df["state"].nunique(dropna=True) >= 2:
        group_splits = min(5, df["state"].nunique(dropna=True))
        splitters.append(("groupkfold_state", GroupKFold(n_splits=group_splits), df["state"]))

    models = make_models(pos_weight)
    all_metric_rows = []
    all_oof = []
    for model_name, model in models.items():
        for cv_name, splitter, groups in splitters:
            rows, oof = evaluate_cv(model_name, model, x, y, splitter, cv_name, groups)
            all_metric_rows.extend(rows)
            all_oof.append(oof)

    metrics = pd.DataFrame(all_metric_rows)
    oof_all = pd.concat(all_oof, ignore_index=True)
    summary = summarize_metrics(metrics)
    group_errors = group_error_report(oof_all, meta)
    permutation_repeats = 0 if args.skip_permutation else max(0, args.permutation_repeats)
    importance = feature_importance(models["random_forest"], x, y, permutation_repeats)

    metrics_path = output_path(config, "outputs", f"{OUTPUT_DIR}/concept_baseline_cv_metrics_{DATASET_NAME}.csv")
    summary_path = output_path(config, "outputs", f"{OUTPUT_DIR}/concept_baseline_cv_summary_{DATASET_NAME}.csv")
    group_path = output_path(config, "outputs", f"{OUTPUT_DIR}/concept_baseline_group_errors_{DATASET_NAME}.csv")
    importance_path = output_path(config, "outputs", f"{OUTPUT_DIR}/concept_baseline_feature_importance_{DATASET_NAME}.csv")
    report_path = output_path(config, "outputs", f"{OUTPUT_DIR}/concept_baseline_report_{DATASET_NAME}.txt")

    write_dataframe(metrics, metrics_path)
    write_dataframe(summary, summary_path)
    write_dataframe(group_errors, group_path)
    write_dataframe(importance, importance_path)
    write_report(report_path, df, feature_cols, summary, importance)

    run_summary = {
        "input_dataset": str(dataset_path),
        "dataset": DATASET_NAME,
        "rows": int(len(df)),
        "concept_features": int(len(feature_cols)),
        "positive": pos,
        "negative": neg,
        "permutation_repeats": int(permutation_repeats),
        "outputs": {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "group_errors": str(group_path),
            "feature_importance": str(importance_path),
            "report": str(report_path),
        },
    }
    write_json(output_path(config, "logs", "17_train_concept_baseline_summary.json"), run_summary)

    print(f"Trained concept baselines: {len(df)} rows, {len(feature_cols)} concept features")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


