from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = PROJECT_ROOT / "scripts" / "stage_11_paper_optimization" / "63_full_feature_graph_guided_m4_and_perturbation.py"
METHOD_SCRIPT = PROJECT_ROOT / "scripts" / "stage_12_paper_comparison" / "71_reproduce_recent_mpm_methods.py"
RUN_DIR = PROJECT_ROOT / "outputs" / "standardized_runs" / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
GRAPH_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "paper_optimization"
    / "climate_sensitivity_graph"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "paper_comparison_methods"
    / "graph_guided_m4_external_models"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)
RANDOM_STATE = 20260622
DECISION_THRESHOLD = 0.5

MODEL_KEYS = [
    "Full_RF",
    "NoClimate_RF",
    "M4_Spearman_RF",
    "M4_GraphGuided_RF",
    "M4_GraphUnion_RF",
]
CLASSIFIER_FAMILIES = ["paper_stacking_ensemble", "supervised_catboost"]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT, "graph_guided_base")
methods = load_module(METHOD_SCRIPT, "recent_mpm_methods_for_graph")


def make_splitters(df: pd.DataFrame, validation_mode: str):
    y = df[base.TARGET].astype(int)
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    n_splits = min(5, positives, negatives)
    splitters = []
    if validation_mode in {"all", "stratified"}:
        splitters.append(
            ("stratified_kfold", StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), None)
        )
    if validation_mode in {"all", "group"} and "state" in df.columns and df["state"].nunique(dropna=True) >= 2:
        groups = df["state"].fillna("unknown").astype(str)
        splitters.append(("groupkfold_state", GroupKFold(n_splits=min(5, groups.nunique())), groups))
    return splitters


def make_external_classifier(family: str, pos_weight: float):
    if family == "paper_stacking_ensemble":
        return methods.make_stacking_model(pos_weight)
    if family == "supervised_catboost":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", methods.make_catboost(pos_weight)),
            ]
        )
    raise ValueError(family)


def display_model_name(model_key: str, family: str) -> str:
    stem = model_key.removesuffix("_RF")
    suffix = "Stacking" if family == "paper_stacking_ensemble" else "CatBoost"
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
        "experiment": "graph_guided_m4_external_models",
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
                "experiment": "graph_guided_m4_external_models",
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


def write_report(output_dir: Path, dataset: str, summary: pd.DataFrame, validation_mode: str) -> None:
    cols = [
        "cv",
        "model",
        "classifier_family",
        "roc_auc_mean",
        "average_precision_mean",
        "balanced_accuracy_mean",
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
    display = summary[[c for c in cols if c in summary.columns]].sort_values(
        ["cv", "classifier_family", "top05_f1_mean"], ascending=[True, True, False]
    )
    report = [
        "# Graph-Guided M4 on External MPM Models",
        "",
        "## Experiment Setting",
        "",
        f"- Dataset: `{dataset}`.",
        "- Samples: positive + negative only; neutral samples are not used.",
        "- Validation: `{}`.".format(validation_mode),
        "- External classifiers: `paper_stacking_ensemble`, `supervised_catboost`.",
        "- Graph source: Climate Sensitivity Graph.",
        "",
        "## Model Definitions",
        "",
        "| Model | Meaning |",
        "|---|---|",
        "| Full | All numeric features including climate variables. |",
        "| NoClimate | All numeric non-climate features. |",
        "| M4_Spearman | Fold-local Spearman-selected geochemical residualization. |",
        "| M4_GraphGuided | Residualize raw features mapped from selected Climate Sensitivity Graph target concepts. |",
        "| M4_GraphUnion | Union of Spearman-selected and graph-selected residualization targets. |",
        "",
        "## Results",
        "",
        base.markdown_table(display, max_rows=200),
    ]
    (output_dir / "graph_guided_m4_external_models_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph-guided M4 variants with external MPM models.")
    parser.add_argument("--run-dir", default=str(RUN_DIR))
    parser.add_argument("--graph-dir", default=str(GRAPH_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    parser.add_argument("--validation-mode", choices=["group", "stratified", "all"], default="group")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []

    for cv_name, splitter, groups in make_splitters(df, args.validation_mode):
        split_iter = splitter.split(df, y, groups) if groups is not None else splitter.split(df, y)
        for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            train_df = df.iloc[train_idx].copy()
            y_train = y.iloc[train_idx]
            test_group = None
            if groups is not None:
                test_group = ";".join(sorted(pd.Series(groups).iloc[test_idx].astype(str).unique().tolist()))

            for model_key in MODEL_KEYS:
                recipe, selection = base.make_recipe(
                    model_key,
                    train_df,
                    fsets,
                    graph_residual_cols,
                    args.spearman_threshold,
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

                for family in CLASSIFIER_FAMILIES:
                    clf = make_external_classifier(family, pos_weight)
                    clf.fit(x_train, y_train)
                    y_prob = clf.predict_proba(x_test)[:, 1]
                    add_metric_row(
                        metric_rows,
                        prediction_rows,
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

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    summary = summarize_metrics(metrics)

    base.write_table(metrics, output_dir / "graph_guided_m4_external_models_fold_metrics.csv")
    base.write_table(predictions, output_dir / "graph_guided_m4_external_models_predictions.csv")
    base.write_table(summary, output_dir / "graph_guided_m4_external_models_summary.csv")
    base.write_table(graph_cols, output_dir / "graph_guided_m4_external_raw_feature_targets.csv")
    base.write_table(selections, output_dir / "graph_guided_m4_external_spearman_selection_by_fold.csv")
    manifest = {
        "dataset": dataset,
        "dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "graph_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "spearman_threshold": args.spearman_threshold,
        "validation_mode": args.validation_mode,
        "samples_used": "positive_negative_only",
        "neutral_samples_used": False,
        "model_keys": MODEL_KEYS,
        "classifier_families": CLASSIFIER_FAMILIES,
        "graph_mapped_feature_count": int(len(graph_cols)),
        "graph_residual_feature_count": int(len(graph_residual_cols)),
    }
    (output_dir / "graph_guided_m4_external_models_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir, dataset, summary, args.validation_mode)

    print(f"Wrote graph-guided external model results to: {output_dir}")
    print((output_dir / "graph_guided_m4_external_models_report.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
