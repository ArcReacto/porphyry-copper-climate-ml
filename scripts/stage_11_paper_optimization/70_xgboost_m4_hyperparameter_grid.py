from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = PROJECT_ROOT / "scripts" / "stage_11_paper_optimization" / "63_full_feature_graph_guided_m4_and_perturbation.py"
RANDOM_STATE = 20260622
DECISION_THRESHOLD = 0.5


def load_base_module():
    spec = importlib.util.spec_from_file_location("full_feature_graph_guided_m4", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def make_xgb_classifier(pos_weight: float) -> Pipeline:
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


def graph_target_columns_by_tau(
    mapping_path: Path,
    graph_targets_path: Path,
    role_table: pd.DataFrame,
    tau: float,
    residual_role_policy: str,
) -> pd.DataFrame:
    mapping = pd.read_csv(mapping_path)
    targets = pd.read_csv(graph_targets_path)
    score_col = "max_combined_climate_sensitivity_score"
    if score_col not in targets.columns:
        raise KeyError(f"Missing graph score column: {score_col}")

    selected = targets[pd.to_numeric(targets[score_col], errors="coerce") >= tau].copy()
    selected_concepts = set(selected["target_concept"].astype(str))
    selected_score = selected.set_index("target_concept")[score_col].to_dict()
    selected_best_climate = selected.set_index("target_concept").get("best_climate_concept", pd.Series()).to_dict()
    role_lookup = role_table.set_index("column").to_dict(orient="index")

    rows: list[dict] = []
    mapped = mapping[
        mapping["mapped"].astype(bool)
        & mapping["include_in_concept_features"].astype(bool)
        & mapping["concept"].astype(str).isin(selected_concepts)
    ].copy()
    for _, row in mapped.iterrows():
        column = str(row["feature_name"])
        role = role_lookup.get(column)
        if not role or not bool(role.get("is_numeric")):
            continue
        n_unique = int(role.get("n_unique", 0))
        raw_role = str(role.get("role"))
        concept_group = str(row["concept_group"])
        if residual_role_policy == "surface":
            allowed = raw_role == "geochemistry" or concept_group == "terrain"
        else:
            allowed = raw_role in {"geochemistry", "geo_structure"}
        residualizable = n_unique > 2 and allowed
        rows.append(
            {
                "column": column,
                "concept": str(row["concept"]),
                "concept_group": concept_group,
                "raw_role": raw_role,
                "n_unique": n_unique,
                "tau_threshold": tau,
                "graph_score": float(selected_score.get(str(row["concept"]), math.nan)),
                "best_climate_concept": str(selected_best_climate.get(str(row["concept"]), "")),
                "residual_role_policy": residual_role_policy,
                "residualizable": residualizable,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "column",
                "concept",
                "concept_group",
                "raw_role",
                "n_unique",
                "tau_threshold",
                "graph_score",
                "best_climate_concept",
                "residual_role_policy",
                "residualizable",
            ]
        )
    return pd.DataFrame(rows).drop_duplicates("column").sort_values(["concept_group", "concept", "column"])


def make_grid_recipe(
    train_df: pd.DataFrame,
    fsets: dict[str, list[str]],
    graph_residual_cols: set[str],
    rho_threshold: float,
) -> tuple[object, pd.DataFrame]:
    climate = fsets["climate_adjusters"]
    geochem = fsets["geochemistry"]
    no_climate = fsets["no_climate"]

    sensitive = base.local_spearman_sensitive(train_df, geochem, climate, rho_threshold)
    spearman_cols = set(sensitive.loc[sensitive["is_spearman_sensitive"], "target_column"].tolist())
    residual_cols = graph_residual_cols | spearman_cols
    residual_cols_ordered = [c for c in no_climate if c in residual_cols]
    raw_cols = [c for c in no_climate if c not in set(residual_cols_ordered)]

    selection = sensitive.copy()
    selection["selected_by_spearman"] = selection["target_column"].isin(spearman_cols)
    selection["selected_by_graph"] = selection["target_column"].isin(graph_residual_cols)
    selection["selected_by_model"] = selection["target_column"].isin(residual_cols_ordered)
    return base.FeatureRecipe("M4_GraphUnion_XGB_grid", raw_cols, residual_cols_ordered, climate).fit(train_df), selection


def add_metric_row(
    rows: list[dict],
    preds: list[dict],
    *,
    dataset: str,
    cv_name: str,
    fold: int,
    test_group: str | None,
    model_name: str,
    tau_threshold: float | None,
    rho_threshold: float | None,
    n_graph_residual_cols: int,
    n_spearman_selected_cols: int,
    used_cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    df: pd.DataFrame,
    y_prob: np.ndarray,
) -> None:
    y_true = df.iloc[test_idx][base.TARGET].astype(int).to_numpy()
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    row = {
        "dataset": dataset,
        "experiment": "xgboost_m4_hyperparameter_grid",
        "cv": cv_name,
        "fold": int(fold),
        "test_group": test_group,
        "model": model_name,
        "classifier_family": "xgboost",
        "tau_threshold": tau_threshold,
        "rho_threshold": rho_threshold,
        "n_graph_residual_cols": int(n_graph_residual_cols),
        "n_spearman_selected_cols": int(n_spearman_selected_cols),
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
                "experiment": "xgboost_m4_hyperparameter_grid",
                "row_index": int(row_index),
                "sample_id": df.iloc[row_index].get("sample_id", ""),
                "cv": cv_name,
                "fold": int(fold),
                "test_group": test_group,
                "model": model_name,
                "tau_threshold": tau_threshold,
                "rho_threshold": rho_threshold,
                "Y_label": int(true_value),
                "y_pred": int(pred_value),
                "y_prob": float(prob_value),
                "state": df.iloc[row_index].get("state", ""),
                "negative_type": df.iloc[row_index].get("negative_type", ""),
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
        "top01_precision",
        "top01_recall",
        "top01_f1",
        "top01_lift",
        "top01_ndcg",
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
        "n_graph_residual_cols",
        "n_spearman_selected_cols",
        "n_features",
    ]
    groups = ["dataset", "experiment", "cv", "model", "tau_threshold", "rho_threshold"]
    summary = metrics.groupby(groups, dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    return summary.sort_values(["cv", "top05_f1_mean", "top05_precision_mean"], ascending=[True, False, False])


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
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


def write_report(output_dir: Path, dataset: str, summary: pd.DataFrame, graph_counts: pd.DataFrame) -> None:
    preferred = [
        "cv",
        "model",
        "tau_threshold",
        "rho_threshold",
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
        "top10_ndcg_mean",
        "n_graph_residual_cols_mean",
        "n_spearman_selected_cols_mean",
    ]
    report = [
        f"# XGBoost M4 Hyperparameter Grid: {dataset}",
        "",
        "This experiment uses only supervised positive and hard-negative samples (`Y_label` in {0, 1}).",
        "",
        "## Parameter Meaning",
        "",
        "- `tau_threshold`: minimum Climate Sensitivity Graph combined score for target concepts.",
        "- `rho_threshold`: fold-local absolute Spearman threshold between geochemistry features and climate adjusters.",
        "- `NoClimate_XGB`: baseline without climate variables and without residualization.",
        "- `M4_GraphUnion_XGB`: graph-selected residual targets union fold-local Spearman-selected geochemical targets.",
        "",
        "## Summary",
        "",
        markdown_table(summary[[c for c in preferred if c in summary.columns]].sort_values(["cv", "top05_f1_mean"], ascending=[True, False])),
        "",
        "## Graph Feature Counts by Tau",
        "",
        markdown_table(graph_counts),
    ]
    (output_dir / "xgboost_m4_hyperparameter_grid_report.md").write_text("\n".join(report), encoding="utf-8")


def run_experiment(
    run_dir: Path,
    graph_dir: Path,
    output_dir: Path,
    tau_values: list[float],
    rho_values: list[float],
    cv_filter: set[str] | None,
    residual_role_policy: str,
) -> pd.DataFrame:
    dataset = run_dir.name
    df, dataset_path = base.load_dataset_from_run(run_dir)
    df = df[df[base.TARGET].isin([0, 1])].reset_index(drop=True)
    role_table = pd.read_csv(run_dir / "00_dataset_profile" / "feature_roles.csv")
    mapping_path = run_dir / "03_causal_graph" / "02_feature_to_concept_resolved.csv"
    graph_targets_path = graph_dir / "climate_sensitive_target_concepts.csv"
    fsets = base.feature_sets(role_table)
    y = df[base.TARGET].astype(int).reset_index(drop=True)

    graph_by_tau: dict[float, pd.DataFrame] = {}
    graph_counts_rows: list[dict] = []
    for tau in tau_values:
        graph_cols = graph_target_columns_by_tau(
            mapping_path,
            graph_targets_path,
            role_table,
            tau,
            residual_role_policy,
        )
        graph_by_tau[tau] = graph_cols
        graph_counts_rows.append(
            {
                "tau_threshold": tau,
                "mapped_feature_count": int(len(graph_cols)),
                "residualizable_feature_count": int(graph_cols["residualizable"].sum()) if not graph_cols.empty else 0,
                "selected_concept_count": int(graph_cols["concept"].nunique()) if not graph_cols.empty else 0,
            }
        )

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
            test_df = df.iloc[test_idx].copy()
            y_train = y.iloc[train_idx]
            pos_weight = int((y_train == 0).sum()) / max(int(y_train.sum()), 1)
            test_group = None
            if groups is not None:
                test_group = ";".join(sorted(pd.Series(groups).iloc[test_idx].astype(str).unique().tolist()))

            no_climate_recipe = base.FeatureRecipe("NoClimate_XGB", fsets["no_climate"], [], fsets["climate_adjusters"]).fit(train_df)
            x_train = no_climate_recipe.transform(train_df)
            x_test = no_climate_recipe.transform(test_df)
            clf = make_xgb_classifier(pos_weight)
            clf.fit(x_train, y_train)
            y_prob = clf.predict_proba(x_test)[:, 1]
            add_metric_row(
                metrics_rows,
                pred_rows,
                dataset=dataset,
                cv_name=cv_name,
                fold=fold,
                test_group=test_group,
                model_name="NoClimate_XGB",
                tau_threshold=None,
                rho_threshold=None,
                n_graph_residual_cols=0,
                n_spearman_selected_cols=0,
                used_cols=no_climate_recipe.output_columns(),
                train_idx=train_idx,
                test_idx=test_idx,
                df=df,
                y_prob=y_prob,
            )

            for tau in tau_values:
                graph_cols = graph_by_tau[tau]
                graph_residual_cols = set(graph_cols.loc[graph_cols["residualizable"], "column"].tolist())
                for rho in rho_values:
                    recipe, selection = make_grid_recipe(train_df, fsets, graph_residual_cols, rho)
                    if not selection.empty:
                        temp = selection.copy()
                        temp.insert(0, "dataset", dataset)
                        temp.insert(1, "cv", cv_name)
                        temp.insert(2, "fold", fold)
                        temp.insert(3, "test_group", test_group)
                        temp["tau_threshold"] = tau
                        temp["rho_threshold"] = rho
                        temp["model"] = "M4_GraphUnion_XGB"
                        selection_rows.append(temp)
                    x_train = recipe.transform(train_df)
                    x_test = recipe.transform(test_df)
                    clf = make_xgb_classifier(pos_weight)
                    clf.fit(x_train, y_train)
                    y_prob = clf.predict_proba(x_test)[:, 1]
                    n_spearman = int(selection["selected_by_spearman"].sum()) if not selection.empty else 0
                    add_metric_row(
                        metrics_rows,
                        pred_rows,
                        dataset=dataset,
                        cv_name=cv_name,
                        fold=fold,
                        test_group=test_group,
                        model_name=f"M4_GraphUnion_XGB_tau{tau:.1f}_rho{rho:.1f}",
                        tau_threshold=tau,
                        rho_threshold=rho,
                        n_graph_residual_cols=len(graph_residual_cols),
                        n_spearman_selected_cols=n_spearman,
                        used_cols=recipe.output_columns(),
                        train_idx=train_idx,
                        test_idx=test_idx,
                        df=df,
                        y_prob=y_prob,
                    )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(pred_rows)
    selections = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    graph_counts = pd.DataFrame(graph_counts_rows)
    summary = summarize_metrics(metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    base.write_table(metrics, output_dir / "xgboost_m4_hyperparameter_grid_fold_metrics.csv")
    base.write_table(predictions, output_dir / "xgboost_m4_hyperparameter_grid_predictions.csv")
    base.write_table(summary, output_dir / "xgboost_m4_hyperparameter_grid_summary.csv")
    base.write_table(selections, output_dir / "xgboost_m4_hyperparameter_grid_spearman_selection_by_fold.csv")
    base.write_table(graph_counts, output_dir / "xgboost_m4_hyperparameter_grid_graph_counts.csv")
    for tau, graph_cols in graph_by_tau.items():
        base.write_table(graph_cols, output_dir / f"graph_targets_tau{tau:.1f}.csv")
    write_report(output_dir, dataset, summary, graph_counts)
    manifest = {
        "dataset": dataset,
        "input_dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "graph_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "tau_values": tau_values,
        "rho_values": rho_values,
        "cv_filter": sorted(cv_filter) if cv_filter else None,
        "residual_role_policy": residual_role_policy,
        "classifier": "XGBoost",
        "xgboost_params": {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 2.0,
        },
    }
    (output_dir / "xgboost_m4_hyperparameter_grid_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    default_run_dir = (
        PROJECT_ROOT
        / "outputs"
        / "standardized_runs"
        / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
    )
    default_graph_dir = (
        PROJECT_ROOT
        / "outputs"
        / "paper_optimization"
        / "climate_sensitivity_graph"
        / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
    )
    parser = argparse.ArgumentParser(description="Run XGBoost grid over Graph-guided M4 thresholds.")
    parser.add_argument("--run-dir", default=str(default_run_dir))
    parser.add_argument("--graph-dir", default=str(default_graph_dir))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tau-values", default="0.3,0.5,0.7")
    parser.add_argument("--rho-values", default="0.2,0.3,0.4")
    parser.add_argument(
        "--cv",
        nargs="*",
        default=["groupkfold_state"],
        help="CV names to run. Use stratified_kfold groupkfold_state for both.",
    )
    parser.add_argument(
        "--residual-role-policy",
        choices=["all", "surface"],
        default="all",
        help="all matches the previous graph-guided M4; surface restricts graph residualization to geochemistry and terrain concepts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    graph_dir = Path(args.graph_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "paper_optimization" / "xgboost_m4_hyperparameter_grid" / run_dir.name
    )
    summary = run_experiment(
        run_dir=run_dir,
        graph_dir=graph_dir,
        output_dir=output_dir,
        tau_values=parse_float_list(args.tau_values),
        rho_values=parse_float_list(args.rho_values),
        cv_filter=set(args.cv) if args.cv else None,
        residual_role_policy=args.residual_role_policy,
    )
    best = summary.sort_values("top05_f1_mean", ascending=False).head(5)
    print(f"Wrote XGBoost M4 hyperparameter grid outputs to: {output_dir}")
    print(best[["cv", "model", "tau_threshold", "rho_threshold", "top05_precision_mean", "top05_f1_mean", "top10_f1_mean", "average_precision_mean"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
