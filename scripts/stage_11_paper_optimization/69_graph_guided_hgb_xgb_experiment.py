from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = PROJECT_ROOT / "scripts" / "stage_11_paper_optimization" / "63_full_feature_graph_guided_m4_and_perturbation.py"
RANDOM_STATE = 20260622
DECISION_THRESHOLD = 0.5

MODEL_KEYS = [
    "Full_RF",
    "NoClimate_RF",
    "M4_Spearman_RF",
    "M4_GraphGuided_RF",
    "M4_GraphUnion_RF",
]
CLASSIFIER_FAMILIES = ["hist_gradient_boosting", "xgboost"]


def load_base_module():
    spec = importlib.util.spec_from_file_location("full_feature_graph_guided_m4", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def make_classifier(family: str, pos_weight: float) -> Pipeline:
    if family == "hist_gradient_boosting":
        return Pipeline(
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
        )
    if family == "xgboost":
        return Pipeline(
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
        )
    raise ValueError(family)


def display_model_name(model_key: str, family: str) -> str:
    stem = model_key.removesuffix("_RF")
    suffix = "HGB" if family == "hist_gradient_boosting" else "XGB"
    return f"{stem}_{suffix}"


def add_metric_row(
    rows: list[dict],
    preds: list[dict],
    *,
    dataset: str,
    cv_name: str,
    fold: int,
    test_group: str | None,
    model_key: str,
    classifier_family: str,
    used_cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    df: pd.DataFrame,
    y_prob: np.ndarray,
) -> None:
    y_true = df.iloc[test_idx][base.TARGET].astype(int).to_numpy()
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    model_name = display_model_name(model_key, classifier_family)
    row = {
        "dataset": dataset,
        "experiment": "graph_guided_classifier_replacement",
        "cv": cv_name,
        "fold": int(fold),
        "test_group": test_group,
        "model": model_key,
        "classifier_family": classifier_family,
        "model_name": model_name,
        "n_features": int(len(used_cols)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "positive_test": int(y_true.sum()),
        "negative_test": int((y_true == 0).sum()),
        "roc_auc": base.safe_metric("roc_auc", y_true, y_pred, y_prob),
        "average_precision": base.safe_metric("average_precision", y_true, y_pred, y_prob),
        "balanced_accuracy": base.safe_metric("balanced_accuracy", y_true, y_pred, y_prob),
        "precision": base.safe_metric("precision", y_true, y_pred, y_prob),
        "recall": base.safe_metric("recall", y_true, y_pred, y_prob),
        "f1": base.safe_metric("f1", y_true, y_pred, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    row.update(base.top_k_metrics(y_true, y_prob))
    rows.append(row)
    for row_index, true_value, pred_value, prob_value in zip(test_idx, y_true, y_pred, y_prob):
        preds.append(
            {
                "dataset": dataset,
                "experiment": "graph_guided_classifier_replacement",
                "row_index": int(row_index),
                "sample_id": df.iloc[row_index].get("sample_id", ""),
                "cv": cv_name,
                "fold": int(fold),
                "test_group": test_group,
                "model": model_key,
                "classifier_family": classifier_family,
                "model_name": model_name,
                "Y_label": int(true_value),
                "y_pred": int(pred_value),
                "y_prob": float(prob_value),
                "state": df.iloc[row_index].get("state", ""),
                "negative_type": df.iloc[row_index].get("negative_type", ""),
                "env_causal_group_id": df.iloc[row_index].get("env_causal_group_id", ""),
                "env_weathering_regime": df.iloc[row_index].get("env_weathering_regime", ""),
            }
        )


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
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
    summary = metrics.groupby(["dataset", "cv", "model", "classifier_family", "model_name"], dropna=False)[
        metric_cols
    ].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    return summary


def markdown_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return "_No data._"
    show = df.head(max_rows).copy()
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


def write_report(out_dir: Path, dataset: str, summary: pd.DataFrame) -> None:
    cols = [
        "cv",
        "model",
        "classifier_family",
        "roc_auc_mean",
        "average_precision_mean",
        "f1_mean",
        "top05_precision_mean",
        "top05_recall_mean",
        "top05_f1_mean",
        "top05_lift_mean",
        "top05_ndcg_mean",
        "top10_precision_mean",
        "top10_recall_mean",
        "top10_f1_mean",
        "top10_lift_mean",
        "top10_ndcg_mean",
    ]
    group = summary[summary["cv"] == "groupkfold_state"].copy()
    report = [
        f"# Graph-Guided M4 with HGB/XGBoost: {dataset}",
        "",
        "## Setting",
        "",
        "- Dataset: `known_mining_neutral_ratio_1_10_supervised_all_features_v1`.",
        "- Samples: positive + negative only; neutral samples are not used.",
        "- Graph inputs: Climate Sensitivity Graph target concepts and mapped raw features.",
        "- Replaced classifiers: HistGradientBoosting and XGBoost.",
        "- Validation: StratifiedKFold and GroupKFold by state.",
        "",
        "## Model Definitions",
        "",
        "| Model | Meaning |",
        "|---|---|",
        "| Full | All raw numeric features including climate variables. |",
        "| NoClimate | All numeric non-climate features. |",
        "| M4_Spearman | Fold-local Spearman-selected geochemical residualization. |",
        "| M4_GraphGuided | Residualize raw features mapped from Climate Sensitivity Graph target concepts. |",
        "| M4_GraphUnion | Union of Spearman-selected and graph-selected residualization targets. |",
        "",
        "## GroupKFold Main Results",
        "",
        markdown_table(group[[c for c in cols if c in group.columns]].sort_values("top05_f1_mean", ascending=False)),
        "",
        "## All CV Results",
        "",
        markdown_table(
            summary[[c for c in cols if c in summary.columns]].sort_values(
                ["cv", "top05_f1_mean"], ascending=[True, False]
            ),
            max_rows=200,
        ),
    ]
    (out_dir / "graph_guided_hgb_xgb_report.md").write_text("\n".join(report), encoding="utf-8")


def run_experiment(
    run_dir: Path,
    graph_dir: Path,
    output_dir: Path,
    spearman_threshold: float,
    model_keys: list[str],
    classifier_families: list[str],
    cv_filter: set[str] | None,
):
    dataset = run_dir.name
    df, dataset_path = base.load_dataset_from_run(run_dir)
    df = df[df[base.TARGET].isin([0, 1])].reset_index(drop=True)
    role_table = pd.read_csv(run_dir / "00_dataset_profile" / "feature_roles.csv")
    mapping_path = run_dir / "03_causal_graph" / "02_feature_to_concept_resolved.csv"
    graph_targets_path = graph_dir / "climate_sensitive_target_concepts.csv"
    graph_cols = base.selected_graph_target_columns(mapping_path, graph_targets_path, role_table)
    graph_residual_cols = set(graph_cols.loc[graph_cols["residualizable"], "column"].tolist())
    fsets = base.feature_sets(role_table)
    y = df[base.TARGET].astype(int).reset_index(drop=True)
    pos_weight = int((y == 0).sum()) / max(int(y.sum()), 1)

    metrics_rows: list[dict] = []
    pred_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []

    for cv_name, splitter, groups in base.make_splitters(df):
        if cv_filter and cv_name not in cv_filter:
            continue
        split_iter = splitter.split(df, y, groups) if groups is not None else splitter.split(df, y)
        for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            train_df = df.iloc[train_idx].copy()
            y_train = y.iloc[train_idx]
            test_group = None
            if groups is not None:
                test_group = ";".join(sorted(pd.Series(groups).iloc[test_idx].astype(str).unique().tolist()))
            for model_key in model_keys:
                recipe, selection = base.make_recipe(
                    model_key,
                    train_df,
                    fsets,
                    graph_residual_cols,
                    spearman_threshold,
                )
                if not selection.empty:
                    temp = selection.copy()
                    temp.insert(0, "dataset", dataset)
                    temp.insert(1, "cv", cv_name)
                    temp.insert(2, "fold", fold)
                    temp.insert(3, "test_group", test_group)
                    temp["base_model"] = model_key
                    selection_rows.append(temp)
                x_train = recipe.transform(train_df)
                x_test = recipe.transform(df.iloc[test_idx].copy())
                for family in classifier_families:
                    clf = make_classifier(family, pos_weight)
                    clf.fit(x_train, y_train)
                    y_prob = clf.predict_proba(x_test)[:, 1]
                    add_metric_row(
                        metrics_rows,
                        pred_rows,
                        dataset=dataset,
                        cv_name=cv_name,
                        fold=fold,
                        test_group=test_group,
                        model_key=model_key,
                        classifier_family=family,
                        used_cols=recipe.output_columns(),
                        train_idx=train_idx,
                        test_idx=test_idx,
                        df=df,
                        y_prob=y_prob,
                    )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(pred_rows)
    selections = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    summary = summarize_metrics(metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    base.write_table(metrics, output_dir / "graph_guided_hgb_xgb_fold_metrics.csv")
    base.write_table(predictions, output_dir / "graph_guided_hgb_xgb_predictions.csv")
    base.write_table(summary, output_dir / "graph_guided_hgb_xgb_summary.csv")
    base.write_table(graph_cols, output_dir / "graph_guided_raw_feature_targets.csv")
    base.write_table(selections, output_dir / "graph_guided_hgb_xgb_spearman_selection_by_fold.csv")
    write_report(output_dir, dataset, summary)
    manifest = {
        "dataset": dataset,
        "input_dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "graph_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "spearman_threshold": spearman_threshold,
        "samples_used": "positive_negative_only",
        "neutral_samples_used": False,
        "graph_mapped_feature_count": int(len(graph_cols)),
        "graph_residual_feature_count": int(len(graph_residual_cols)),
        "model_keys": model_keys,
        "classifier_families": classifier_families,
        "cv_filter": sorted(cv_filter) if cv_filter else [],
    }
    (output_dir / "graph_guided_hgb_xgb_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Graph-guided M4 with HistGradientBoosting and XGBoost.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    parser.add_argument(
        "--model-keys",
        nargs="+",
        default=MODEL_KEYS,
        choices=MODEL_KEYS,
        help="Subset of graph-guided model keys to run. Default runs all.",
    )
    parser.add_argument(
        "--classifier-families",
        nargs="+",
        default=CLASSIFIER_FAMILIES,
        choices=CLASSIFIER_FAMILIES,
        help="Classifier families to run. Default runs both HGB and XGBoost.",
    )
    parser.add_argument(
        "--cv",
        nargs="*",
        default=None,
        choices=["stratified_kfold", "groupkfold_state"],
        help="Optional CV names to run. Default runs all CV settings.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    graph_dir = Path(args.graph_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "paper_optimization" / "graph_guided_model_family_replacement" / run_dir.name
    )
    cv_filter = set(args.cv) if args.cv else None
    run_experiment(
        run_dir,
        graph_dir,
        output_dir,
        args.spearman_threshold,
        args.model_keys,
        args.classifier_families,
        cv_filter,
    )
    print(f"Wrote Graph-guided HGB/XGB experiment outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
