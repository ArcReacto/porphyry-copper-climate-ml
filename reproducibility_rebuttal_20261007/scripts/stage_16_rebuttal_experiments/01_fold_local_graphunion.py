from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "data"
    / "run_inputs"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rebuttal_experiments" / "01_fold_local_graphunion"
MODEL_KEYS = ["NoClimate_RF", "M4_Spearman_RF", "M4_GraphUnion_RF"]
REPORT_METRICS = [
    "roc_auc",
    "average_precision",
    "f1",
    "top05_precision",
    "top05_recall",
    "top05_f1",
    "top05_ndcg",
    "top10_precision",
    "top10_recall",
    "top10_f1",
    "top10_ndcg",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage40 = load_module(
    "rebuttal_stage40",
    PROJECT_ROOT / "scripts" / "stage_05_causal_graph" / "40_run_standard_causal_graph_workflow.py",
)
stage41 = load_module(
    "rebuttal_stage41",
    PROJECT_ROOT / "scripts" / "stage_05_causal_graph" / "41_run_concept_climate_decoupling.py",
)
graph61 = load_module(
    "rebuttal_graph61",
    PROJECT_ROOT / "scripts" / "stage_11_paper_optimization" / "61_build_climate_sensitivity_graph.py",
)
base63 = load_module(
    "rebuttal_base63",
    PROJECT_ROOT
    / "scripts"
    / "stage_11_paper_optimization"
    / "63_full_feature_graph_guided_m4_and_perturbation.py",
)


def align_concept_rows(df: pd.DataFrame, concept_df: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" in df.columns and "sample_id" in concept_df.columns:
        left_ids = df["sample_id"].astype(str)
        right_ids = concept_df["sample_id"].astype(str)
        if left_ids.is_unique and right_ids.is_unique and set(left_ids) == set(right_ids):
            aligned = concept_df.assign(_sample_key=right_ids).set_index("_sample_key").loc[left_ids].reset_index(drop=True)
            if not np.array_equal(aligned["sample_id"].astype(str).to_numpy(), left_ids.to_numpy()):
                raise ValueError("Concept feature sample alignment failed.")
            return aligned
    if len(df) != len(concept_df):
        raise ValueError(f"Cannot align dataset ({len(df)}) and concept table ({len(concept_df)}).")
    return concept_df.reset_index(drop=True)


def local_m4_summary(
    train_concept: pd.DataFrame,
    concept_roles: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensitive = stage41.local_climate_sensitive_concepts(train_concept, concept_roles, threshold)
    if sensitive.empty:
        return sensitive, pd.DataFrame()
    out = sensitive.copy()
    out["target_concept"] = out["concept_feature"].map(graph61.concept_name)
    out["climate_concept"] = out["best_climate_concept_spearman"].map(graph61.concept_name)
    out["m4_sensitive_rate"] = out["is_climate_sensitive"].astype(float)
    out["m4_mean_abs_spearman"] = out["spearman_abs_max"]
    out["m4_max_abs_spearman"] = out["spearman_abs_max"]
    out["m4_fold_count"] = 1
    summary = out[
        [
            "climate_concept",
            "target_concept",
            "m4_sensitive_rate",
            "m4_mean_abs_spearman",
            "m4_max_abs_spearman",
            "m4_fold_count",
        ]
    ].copy()
    return sensitive, summary


def build_fold_graph(
    train_concept: pd.DataFrame,
    dictionary: pd.DataFrame,
    m4_summary: pd.DataFrame,
    edge_path: Path,
    min_combined_score: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = graph61.global_climate_correlations(train_concept, dictionary)
    edges = graph61.causal_edge_summary(edge_path, dictionary)
    graph = corr.merge(m4_summary, on=["climate_concept", "target_concept"], how="left")
    graph = graph.merge(edges, on=["climate_concept", "target_concept"], how="left")
    numeric_fill = [
        "m4_sensitive_rate",
        "m4_mean_abs_spearman",
        "m4_max_abs_spearman",
        "m4_fold_count",
        "causal_environment_count",
        "causal_stable_edge_score",
        "causal_mean_edge_weight",
    ]
    for column in numeric_fill:
        if column not in graph.columns:
            graph[column] = 0.0
        graph[column] = pd.to_numeric(graph[column], errors="coerce").fillna(0.0)
    if "causal_edge_present" not in graph.columns:
        graph["causal_edge_present"] = False
    graph["causal_edge_present"] = graph["causal_edge_present"].fillna(False).astype(bool)
    if "causal_edge_class" not in graph.columns:
        graph["causal_edge_class"] = ""
    graph["causal_edge_class"] = graph["causal_edge_class"].fillna("")

    graph["global_score_norm"] = graph61.min_max(graph["global_abs_spearman"])
    graph["m4_score_norm"] = graph61.min_max(graph["m4_sensitive_rate"])
    graph["causal_score_norm"] = graph61.min_max(graph["causal_stable_edge_score"])
    graph["combined_climate_sensitivity_score"] = (
        0.45 * graph["global_score_norm"]
        + 0.35 * graph["m4_score_norm"]
        + 0.20 * graph["causal_score_norm"]
    )
    graph["selected_for_graph_guided_m4"] = (
        graph["combined_climate_sensitivity_score"] >= min_combined_score
    )
    graph = graph.sort_values("combined_climate_sensitivity_score", ascending=False).reset_index(drop=True)

    selected = graph.groupby(["target_concept", "target_group"], dropna=False)[
        "combined_climate_sensitivity_score"
    ].idxmax()
    target_summary = graph.loc[selected].copy()
    target_summary = target_summary.rename(
        columns={
            "climate_concept": "best_climate_concept",
            "global_abs_spearman": "max_global_abs_spearman",
            "m4_sensitive_rate": "max_m4_sensitive_rate",
            "causal_stable_edge_score": "max_causal_stable_edge_score",
            "combined_climate_sensitivity_score": "max_combined_climate_sensitivity_score",
        }
    )
    target_summary = target_summary[
        [
            "target_concept",
            "target_group",
            "best_climate_concept",
            "max_global_abs_spearman",
            "max_m4_sensitive_rate",
            "max_causal_stable_edge_score",
            "max_combined_climate_sensitivity_score",
            "selected_for_graph_guided_m4",
        ]
    ].sort_values("max_combined_climate_sensitivity_score", ascending=False)
    return graph, target_summary.reset_index(drop=True)


def paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics[metrics["model"] == "NoClimate_RF"].set_index(["fold", "test_group"])
    rows: list[dict] = []
    for candidate in ["M4_Spearman_RF", "M4_GraphUnion_RF"]:
        current = metrics[metrics["model"] == candidate].set_index(["fold", "test_group"])
        joined = baseline[REPORT_METRICS].join(current[REPORT_METRICS], lsuffix="_base", rsuffix="_candidate")
        for metric in REPORT_METRICS:
            diff = joined[f"{metric}_candidate"] - joined[f"{metric}_base"]
            rows.append(
                {
                    "candidate": candidate,
                    "baseline": "NoClimate_RF",
                    "metric": metric,
                    "baseline_mean": float(joined[f"{metric}_base"].mean()),
                    "candidate_mean": float(joined[f"{metric}_candidate"].mean()),
                    "mean_difference": float(diff.mean()),
                    "better_folds": int((diff > 0).sum()),
                    "equal_folds": int(np.isclose(diff, 0).sum()),
                    "worse_folds": int((diff < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if df.empty:
        return "_No data._"
    show = df[columns].copy() if columns else df.copy()
    for column in show.columns:
        if pd.api.types.is_float_dtype(show[column]):
            show[column] = show[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    lines = [
        "| " + " | ".join(map(str, show.columns)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("|", "\\|") for column in show.columns) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    fold_manifest: pd.DataFrame,
) -> None:
    group = summary[summary["cv"] == "groupkfold_state"].copy()
    summary_columns = [
        "model",
        "roc_auc_mean",
        "average_precision_mean",
        "f1_mean",
        "top05_f1_mean",
        "top05_ndcg_mean",
        "top10_f1_mean",
        "top10_ndcg_mean",
    ]
    paired_show = paired[paired["metric"].isin(["average_precision", "top05_f1", "top05_ndcg", "top10_f1", "top10_ndcg"])]
    report = [
        "# Rebuttal 实验 01：Fold-local GraphUnion 结果",
        "",
        "## 实验边界",
        "",
        "本实验在每个 GroupKFold 训练折内重新计算概念相关、环境稳定信号、候选边、图目标集合、Spearman 目标和残差器。测试折只用于最终变换与预测。",
        "",
        "## 主结果",
        "",
        markdown_table(group, summary_columns),
        "",
        "## 相对 M2 的逐折差值",
        "",
        markdown_table(paired_show),
        "",
        "## 每折构图规模",
        "",
        markdown_table(fold_manifest),
        "",
        "## 解释要求",
        "",
        "- 如果 fold-local GraphUnion 仍优于 M2，可将其作为无泄漏图引导结果。",
        "- 如果增益明显收缩，应以 M4 Spearman 作为主方法，并把 GraphUnion 降为探索性知识增强扩展。",
        "- 无论结果方向如何，均不得用 StratifiedKFold 替代本表的跨州结果。",
    ]
    (output_dir / "fold_local_graphunion_report.md").write_text("\n".join(report), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_graph_root = output_dir / "fold_graphs"
    fold_graph_root.mkdir(parents=True, exist_ok=True)

    df, dataset_path = base63.load_dataset_from_run(run_dir)
    df = df[df[base63.TARGET].isin([0, 1])].reset_index(drop=True)
    dataset = run_dir.name
    role_table = pd.read_csv(run_dir / "00_dataset_profile" / "feature_roles.csv")
    mapping_path = run_dir / "03_causal_graph" / "02_feature_to_concept_resolved.csv"
    concept_df = graph61.read_table(run_dir / "03_causal_graph" / "04_concept_features.parquet")
    concept_df = align_concept_rows(df, concept_df)
    dictionary = graph61.load_dictionary(run_dir / "03_causal_graph" / "05_concept_feature_dictionary.csv")
    concept_roles = pd.read_csv(run_dir / "04_concept_climate_decoupling" / "00_concept_feature_roles.csv")
    fsets = base63.feature_sets(role_table)

    groups = df["state"].fillna("unknown").astype(str)
    splitter = GroupKFold(n_splits=min(5, groups.nunique()))
    y = df[base63.TARGET].astype(int).reset_index(drop=True)
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []
    graph_target_rows: list[pd.DataFrame] = []
    fold_manifest_rows: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, y, groups), start=1):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        train_concept = concept_df.iloc[train_idx].copy()
        fold_dir = fold_graph_root / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        test_group = ";".join(sorted(groups.iloc[test_idx].unique().tolist()))

        stable, candidates = stage40.run_environment_stability(
            train_concept,
            dictionary,
            fold_dir,
            bootstrap=args.bootstrap,
            top_k=args.top_k,
            selection_rate_threshold=args.selection_rate_threshold,
            auc_threshold=args.auc_threshold,
            rf_top_k=args.rf_top_k,
        )
        _, stable_edges, _ = stage40.run_causal_discovery(
            train_concept,
            candidates,
            fold_dir,
            partial_threshold=args.partial_threshold,
            p_threshold=args.p_threshold,
            max_edges=args.max_edges_per_environment,
        )
        local_sensitive, m4_summary = local_m4_summary(
            train_concept,
            concept_roles,
            args.spearman_threshold,
        )
        graph, targets = build_fold_graph(
            train_concept,
            dictionary,
            m4_summary,
            fold_dir / "20_cross_environment_edge_all_summary.csv",
            args.min_combined_score,
        )
        graph61.write_table(local_sensitive, fold_dir / "fold_local_sensitive_concepts.csv")
        graph61.write_table(graph, fold_dir / "climate_sensitivity_edges.csv")
        graph61.write_table(targets, fold_dir / "climate_sensitive_target_concepts.csv")

        graph_cols = base63.selected_graph_target_columns(
            mapping_path,
            fold_dir / "climate_sensitive_target_concepts.csv",
            role_table,
        )
        graph_residual_cols = set(graph_cols.loc[graph_cols["residualizable"], "column"].tolist())
        if not graph_cols.empty:
            temp_targets = graph_cols.copy()
            temp_targets.insert(0, "fold", fold)
            temp_targets.insert(1, "test_group", test_group)
            graph_target_rows.append(temp_targets)

        for model_key in MODEL_KEYS:
            recipe, selection = base63.make_recipe(
                model_key,
                train_df,
                fsets,
                graph_residual_cols,
                args.spearman_threshold,
            )
            if not selection.empty:
                temp = selection.copy()
                temp.insert(0, "fold", fold)
                temp.insert(1, "test_group", test_group)
                selection_rows.append(temp)
            x_train = recipe.transform(train_df)
            x_test = recipe.transform(test_df)
            model = base63.make_rf_model()
            model.fit(x_train, y.iloc[train_idx])
            y_prob = model.predict_proba(x_test)[:, 1]
            base63.add_metric_row(
                metric_rows,
                prediction_rows,
                dataset=dataset,
                cv_name="groupkfold_state",
                fold=fold,
                test_group=test_group,
                model_key=model_key,
                used_cols=recipe.output_columns(),
                train_idx=train_idx,
                test_idx=test_idx,
                df=df,
                y_prob=y_prob,
            )

        fold_manifest_rows.append(
            {
                "fold": fold,
                "test_group": test_group,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "positive_train": int(y.iloc[train_idx].sum()),
                "positive_test": int(y.iloc[test_idx].sum()),
                "stable_concepts": int(len(stable)),
                "candidate_concepts": int(len(candidates)),
                "stable_edges": int(len(stable_edges)),
                "selected_graph_concepts": int(targets["selected_for_graph_guided_m4"].sum()),
                "graph_mapped_features": int(len(graph_cols)),
                "graph_residual_features": int(len(graph_residual_cols)),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    graph_targets = pd.concat(graph_target_rows, ignore_index=True) if graph_target_rows else pd.DataFrame()
    fold_manifest = pd.DataFrame(fold_manifest_rows)
    summary = base63.summarize_metrics(metrics)
    paired = paired_comparisons(metrics)

    base63.write_table(metrics, output_dir / "fold_metrics.csv")
    base63.write_table(predictions, output_dir / "oof_predictions.csv")
    base63.write_table(summary, output_dir / "model_summary.csv")
    base63.write_table(paired, output_dir / "paired_comparison.csv")
    base63.write_table(selections, output_dir / "fold_spearman_selection.csv")
    base63.write_table(graph_targets, output_dir / "fold_graph_raw_feature_targets.csv")
    base63.write_table(fold_manifest, output_dir / "fold_graph_manifest.csv")
    write_report(output_dir, summary, paired, fold_manifest)

    manifest = {
        "experiment": "fold_local_graphunion",
        "dataset": dataset,
        "input_dataset": str(dataset_path),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "validation": "5-fold GroupKFold by state",
        "random_state": base63.RANDOM_STATE,
        "models": MODEL_KEYS,
        "parameters": {
            "bootstrap": args.bootstrap,
            "top_k": args.top_k,
            "selection_rate_threshold": args.selection_rate_threshold,
            "auc_threshold": args.auc_threshold,
            "rf_top_k": args.rf_top_k,
            "partial_threshold": args.partial_threshold,
            "p_threshold": args.p_threshold,
            "max_edges_per_environment": args.max_edges_per_environment,
            "spearman_threshold": args.spearman_threshold,
            "min_combined_score": args.min_combined_score,
        },
        "leakage_control": {
            "concept_correlation": "training fold only",
            "environment_stability": "training fold only",
            "causal_edge_discovery": "training fold only",
            "graph_target_selection": "training fold only",
            "residualizer": "training fold only",
            "test_fold_usage": "transform and evaluation only",
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-controlled fold-local GraphUnion evaluation.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--selection-rate-threshold", type=float, default=0.50)
    parser.add_argument("--auc-threshold", type=float, default=0.30)
    parser.add_argument("--rf-top-k", type=int, default=12)
    parser.add_argument("--partial-threshold", type=float, default=0.18)
    parser.add_argument("--p-threshold", type=float, default=0.20)
    parser.add_argument("--max-edges-per-environment", type=int, default=45)
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    parser.add_argument("--min-combined-score", type=float, default=0.50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args)
    print(f"Wrote rebuttal experiment 01 outputs to: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
