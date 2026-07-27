from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = "Y_label"
RANDOM_STATE = 20260622
TOP_K_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "outputs"
    / "model_datasets"
    / "by_sample_scheme"
    / "known_mining_neutral"
    / "known_mining_neutral_ratio_1_10"
    / "model_dataset_known_mining_neutral_ratio_1_10_supervised_all_features_v1.parquet"
)
DEFAULT_ROLE_TABLE = (
    PROJECT_ROOT
    / "outputs"
    / "standardized_runs"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
    / "00_dataset_profile"
    / "feature_roles.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "paper_optimization"
    / "direct_initial_models_xgboost"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def feature_columns(df: pd.DataFrame, role_table_path: Path | None) -> list[str]:
    if role_table_path and role_table_path.exists():
        role_table = pd.read_csv(role_table_path)
        if "used_in_m1_full_climate" in role_table.columns:
            cols = role_table.loc[role_table["used_in_m1_full_climate"], "column"].tolist()
            return [c for c in cols if c in df.columns]
        if {"column", "is_numeric", "role"}.issubset(role_table.columns):
            excluded_roles = {"metadata", "target", "environment_group", "other_non_numeric"}
            cols = role_table.loc[
                (role_table["is_numeric"]) & (~role_table["role"].isin(excluded_roles)), "column"
            ].tolist()
            return [c for c in cols if c in df.columns]
    return [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c != TARGET and not c.lower().endswith("_id")
    ]


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
                        max_iter=300,
                        max_leaf_nodes=15,
                        l2_regularization=0.1,
                        class_weight={0: 1.0, 1: pos_weight},
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.03,
                        max_depth=3,
                        min_child_weight=3,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=pos_weight,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(roc_auc_score(y_true, y_prob))


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(average_precision_score(y_true, y_prob))


def ndcg_at_k(y_true_sorted: np.ndarray, k: int) -> float:
    selected = y_true_sorted[:k].astype(float)
    if len(selected) == 0:
        return math.nan
    discounts = 1.0 / np.log2(np.arange(2, len(selected) + 2))
    dcg = float(np.sum(selected * discounts))
    ideal = np.sort(y_true_sorted)[::-1][:k].astype(float)
    ideal_dcg = float(np.sum(ideal * discounts))
    return dcg / ideal_dcg if ideal_dcg > 0 else math.nan


def top_k_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    order = np.argsort(-y_prob)
    ranked_true = y_true[order]
    total_pos = max(int(y_true.sum()), 1)
    base_rate = float(y_true.mean()) if len(y_true) else math.nan
    out: dict[str, float] = {}
    for frac in TOP_K_FRACTIONS:
        k = max(1, int(math.ceil(len(y_true) * frac)))
        selected = ranked_true[:k]
        precision_at_k = float(selected.mean()) if k else math.nan
        recall_at_k = float(selected.sum() / total_pos)
        f1_at_k = (
            2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
            if precision_at_k + recall_at_k > 0
            else 0.0
        )
        label = str(int(frac * 100)).zfill(2)
        out[f"top{label}_k"] = int(k)
        out[f"top{label}_precision"] = precision_at_k
        out[f"top{label}_recall"] = recall_at_k
        out[f"top{label}_f1"] = f1_at_k
        out[f"top{label}_lift"] = precision_at_k / base_rate if base_rate and not pd.isna(base_rate) else math.nan
        out[f"top{label}_ndcg"] = ndcg_at_k(ranked_true, k)
    return out


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((y_true == 0).sum()),
        "roc_auc": safe_auc(y_true, y_prob),
        "average_precision": safe_ap(y_true, y_prob),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else math.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else math.nan,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else math.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else math.nan,
    }
    out.update(top_k_metrics(y_true, y_prob))
    return out


def make_splitters(df: pd.DataFrame, y: pd.Series) -> list[tuple[str, object, pd.Series | None]]:
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    n_splits = min(5, positives, negatives)
    if n_splits < 2:
        raise ValueError("Not enough positive/negative samples for cross validation.")
    splitters: list[tuple[str, object, pd.Series | None]] = [
        ("stratified_kfold", StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), None)
    ]
    if "state" in df.columns and df["state"].nunique(dropna=True) >= 2:
        groups = df["state"].fillna("unknown").astype(str)
        splitters.append(("groupkfold_state", GroupKFold(n_splits=min(5, groups.nunique())), groups))
    return splitters


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    for keys, group in predictions.groupby(["dataset", "cv", "model", "fold"], dropna=False):
        row = dict(zip(["dataset", "cv", "model", "fold"], keys))
        row.update(binary_metrics(group[TARGET].astype(int).to_numpy(), group["y_prob"].astype(float).to_numpy()))
        fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)

    model_rows = []
    for keys, group in predictions.groupby(["dataset", "cv", "model"], dropna=False):
        row = dict(zip(["dataset", "cv", "model"], keys))
        row.update(binary_metrics(group[TARGET].astype(int).to_numpy(), group["y_prob"].astype(float).to_numpy()))
        model_rows.append(row)
    model_metrics = pd.DataFrame(model_rows)
    return fold_metrics, model_metrics


def compact_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    lines = [
        "| " + " | ".join(map(str, show.columns)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in show.columns) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    dataset_path: Path,
    dataset_name: str,
    sample_counts: dict[str, int],
    n_features: int,
    model_metrics: pd.DataFrame,
) -> None:
    metric_cols = [
        "dataset",
        "cv",
        "model",
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "top05_precision",
        "top05_recall",
        "top05_f1",
        "top05_lift",
        "top05_ndcg",
        "top10_precision",
        "top10_recall",
        "top10_f1",
        "top10_lift",
        "top10_ndcg",
    ]
    display = model_metrics[[c for c in metric_cols if c in model_metrics.columns]].sort_values(
        ["cv", "top05_f1"], ascending=[True, False]
    )
    report = [
        "# Direct Initial Models + XGBoost Baseline",
        "",
        "## Experiment Setting",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Input file: `{dataset_path}`",
        "- Samples used: positive and negative only; neutral samples are not included.",
        "- Feature processing: full numeric feature set; no climate residualization, no graph guidance, no concept aggregation.",
        "- Validation: StratifiedKFold and GroupKFold by state.",
        "",
        "## Sample Counts",
        "",
        compact_markdown_table(pd.DataFrame([sample_counts])),
        "",
        f"Feature count: `{n_features}`",
        "",
        "## Models",
        "",
        "| Model | Meaning |",
        "|---|---|",
        "| `dummy_stratified` | Random baseline preserving class frequency. |",
        "| `logistic_regression` | Linear baseline with balanced class weights. |",
        "| `random_forest` | Tree ensemble baseline. |",
        "| `hist_gradient_boosting` | Gradient boosting baseline from scikit-learn. |",
        "| `xgboost` | External gradient boosting model. |",
        "",
        "## Results",
        "",
        compact_markdown_table(display),
        "",
        "## Notes",
        "",
        "- This experiment is a pure direct-training baseline.",
        "- It should be used to compare against later climate-decoupled, graph-guided, or counterfactual variants.",
        "- For paper reporting, `groupkfold_state` is the stricter spatial generalization setting.",
    ]
    (output_dir / "direct_initial_models_xgboost_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direct initial baselines plus XGBoost without residualization.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--role-table", default=str(DEFAULT_ROLE_TABLE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    role_table_path = Path(args.role_table) if args.role_table else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(dataset_path)
    df = df[df[TARGET].isin([0, 1])].reset_index(drop=True)
    dataset_name = dataset_path.stem.replace("model_dataset_", "")
    y = df[TARGET].astype(int)
    sample_counts = {
        "rows": int(len(df)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "neutral": int((~df[TARGET].isin([0, 1])).sum()),
    }
    cols = feature_columns(df, role_table_path)
    x = df[cols]
    pos_weight = sample_counts["negative"] / max(sample_counts["positive"], 1)
    models = make_models(pos_weight)

    pred_rows: list[pd.DataFrame] = []
    for cv_name, splitter, groups in make_splitters(df, y):
        split_iter = splitter.split(x, y, groups) if groups is not None else splitter.split(x, y)
        for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            test_group = ""
            if groups is not None:
                test_group = ";".join(sorted(pd.Series(groups).iloc[test_idx].astype(str).unique().tolist()))
            for model_name, model in models.items():
                model.fit(x_train, y_train)
                y_prob = model.predict_proba(x_test)[:, 1]
                pred_rows.append(
                    pd.DataFrame(
                        {
                            "dataset": dataset_name,
                            "cv": cv_name,
                            "fold": fold,
                            "test_group": test_group,
                            "model": model_name,
                            "row_index": test_idx,
                            TARGET: y_test.to_numpy(),
                            "y_prob": y_prob,
                            "sample_id": df.iloc[test_idx].get("sample_id", pd.Series([""] * len(test_idx))).to_numpy(),
                            "state": df.iloc[test_idx].get("state", pd.Series([""] * len(test_idx))).to_numpy(),
                            "sample_type": df.iloc[test_idx].get("sample_type", pd.Series([""] * len(test_idx))).to_numpy(),
                        }
                    )
                )

    predictions = pd.concat(pred_rows, ignore_index=True)
    fold_metrics, model_metrics = summarize_predictions(predictions)
    write_table(predictions, output_dir / "direct_initial_models_xgboost_predictions.csv")
    write_table(fold_metrics, output_dir / "direct_initial_models_xgboost_fold_metrics.csv")
    write_table(model_metrics, output_dir / "direct_initial_models_xgboost_model_metrics.csv")
    manifest = {
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "role_table_path": str(role_table_path) if role_table_path else "",
        "output_dir": str(output_dir),
        "samples_used": "positive_negative_only",
        "neutral_samples_used": False,
        "feature_processing": "direct_full_numeric_features_no_residualization",
        "feature_count": len(cols),
        "models": list(models.keys()),
        "validation": ["stratified_kfold", "groupkfold_state"],
        "random_state": RANDOM_STATE,
        "sample_counts": sample_counts,
    }
    (output_dir / "direct_initial_models_xgboost_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output_dir, dataset_path, dataset_name, sample_counts, len(cols), model_metrics)
    print(f"Wrote direct model results to: {output_dir}")
    print((output_dir / "direct_initial_models_xgboost_report.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
