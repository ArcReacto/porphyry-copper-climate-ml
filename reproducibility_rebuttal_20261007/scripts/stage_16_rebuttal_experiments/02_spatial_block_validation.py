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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rebuttal_experiments" / "02_spatial_block_validation"
MODEL_KEYS = ["NoClimate_RF", "M4_Spearman_RF", "M4_GraphUnion_RF"]
REPORT_METRICS = ["average_precision", "f1", "top05_f1", "top05_ndcg", "top10_f1", "top10_ndcg"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp01 = load_module(
    "rebuttal_exp01",
    PROJECT_ROOT / "scripts" / "stage_16_rebuttal_experiments" / "01_fold_local_graphunion.py",
)


def spatial_block_ids(df: pd.DataFrame, grid_degrees: float) -> pd.Series:
    longitude = pd.to_numeric(df["longitude"], errors="coerce")
    latitude = pd.to_numeric(df["latitude"], errors="coerce")
    if longitude.isna().any() or latitude.isna().any():
        raise ValueError("Spatial validation requires non-missing longitude and latitude.")
    x = np.floor((longitude + 180.0) / grid_degrees).astype(int)
    y = np.floor((latitude + 90.0) / grid_degrees).astype(int)
    return pd.Series([f"{grid_degrees:g}deg_{ix}_{iy}" for ix, iy in zip(x, y)], index=df.index)


def paired_by_grid(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for grid_degrees, grid_metrics in metrics.groupby("grid_degrees"):
        baseline = grid_metrics[grid_metrics["model"] == "NoClimate_RF"].set_index(["fold", "test_group"])
        for candidate in ["M4_Spearman_RF", "M4_GraphUnion_RF"]:
            current = grid_metrics[grid_metrics["model"] == candidate].set_index(["fold", "test_group"])
            joined = baseline[REPORT_METRICS].join(
                current[REPORT_METRICS], lsuffix="_base", rsuffix="_candidate"
            )
            for metric in REPORT_METRICS:
                diff = joined[f"{metric}_candidate"] - joined[f"{metric}_base"]
                rows.append(
                    {
                        "grid_degrees": grid_degrees,
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


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
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
    out = metrics.groupby(["grid_degrees", "model"], dropna=False)[metric_cols].agg(["mean", "std", "count"])
    out = out.reset_index()
    out.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in out.columns
    ]
    return out


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, dataset_path = exp01.base63.load_dataset_from_run(run_dir)
    df = df[df[exp01.base63.TARGET].isin([0, 1])].reset_index(drop=True)
    y = df[exp01.base63.TARGET].astype(int).reset_index(drop=True)
    role_table = pd.read_csv(run_dir / "00_dataset_profile" / "feature_roles.csv")
    mapping_path = run_dir / "03_causal_graph" / "02_feature_to_concept_resolved.csv"
    concept_df = exp01.graph61.read_table(run_dir / "03_causal_graph" / "04_concept_features.parquet")
    concept_df = exp01.align_concept_rows(df, concept_df)
    dictionary = exp01.graph61.load_dictionary(run_dir / "03_causal_graph" / "05_concept_feature_dictionary.csv")
    concept_roles = pd.read_csv(run_dir / "04_concept_climate_decoupling" / "00_concept_feature_roles.csv")
    fsets = exp01.base63.feature_sets(role_table)

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    fold_rows: list[dict] = []
    target_rows: list[pd.DataFrame] = []

    for grid_degrees in args.grid_degrees:
        block_ids = spatial_block_ids(df, grid_degrees)
        block_counts = block_ids.value_counts()
        if block_counts.size < 5:
            raise ValueError(f"Grid {grid_degrees:g} has fewer than five spatial blocks.")
        splitter = GroupKFold(n_splits=5)
        grid_root = output_dir / f"grid_{grid_degrees:g}deg" / "fold_graphs"
        grid_root.mkdir(parents=True, exist_ok=True)

        for fold, (train_idx, test_idx) in enumerate(splitter.split(df, y, block_ids), start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            train_concept = concept_df.iloc[train_idx].copy()
            test_blocks = sorted(block_ids.iloc[test_idx].unique().tolist())
            test_group = ";".join(test_blocks)
            fold_dir = grid_root / f"fold_{fold:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            stable, candidates = exp01.stage40.run_environment_stability(
                train_concept,
                dictionary,
                fold_dir,
                bootstrap=args.bootstrap,
                top_k=args.top_k,
                selection_rate_threshold=args.selection_rate_threshold,
                auc_threshold=args.auc_threshold,
                rf_top_k=args.rf_top_k,
            )
            _, stable_edges, _ = exp01.stage40.run_causal_discovery(
                train_concept,
                candidates,
                fold_dir,
                partial_threshold=args.partial_threshold,
                p_threshold=args.p_threshold,
                max_edges=args.max_edges_per_environment,
            )
            local_sensitive, m4_summary = exp01.local_m4_summary(
                train_concept, concept_roles, args.spearman_threshold
            )
            graph, targets = exp01.build_fold_graph(
                train_concept,
                dictionary,
                m4_summary,
                fold_dir / "20_cross_environment_edge_all_summary.csv",
                args.min_combined_score,
            )
            exp01.graph61.write_table(local_sensitive, fold_dir / "fold_local_sensitive_concepts.csv")
            exp01.graph61.write_table(graph, fold_dir / "climate_sensitivity_edges.csv")
            exp01.graph61.write_table(targets, fold_dir / "climate_sensitive_target_concepts.csv")
            graph_cols = exp01.base63.selected_graph_target_columns(
                mapping_path,
                fold_dir / "climate_sensitive_target_concepts.csv",
                role_table,
            )
            graph_residual_cols = set(graph_cols.loc[graph_cols["residualizable"], "column"].tolist())
            if not graph_cols.empty:
                temp = graph_cols.copy()
                temp.insert(0, "grid_degrees", grid_degrees)
                temp.insert(1, "fold", fold)
                target_rows.append(temp)

            for model_key in MODEL_KEYS:
                recipe, _ = exp01.base63.make_recipe(
                    model_key,
                    train_df,
                    fsets,
                    graph_residual_cols,
                    args.spearman_threshold,
                )
                x_train = recipe.transform(train_df)
                x_test = recipe.transform(test_df)
                model = exp01.base63.make_rf_model()
                model.fit(x_train, y.iloc[train_idx])
                y_prob = model.predict_proba(x_test)[:, 1]
                before_metrics = len(metric_rows)
                before_predictions = len(prediction_rows)
                exp01.base63.add_metric_row(
                    metric_rows,
                    prediction_rows,
                    dataset=run_dir.name,
                    cv_name="spatial_block_kfold",
                    fold=fold,
                    test_group=test_group,
                    model_key=model_key,
                    used_cols=recipe.output_columns(),
                    train_idx=train_idx,
                    test_idx=test_idx,
                    df=df,
                    y_prob=y_prob,
                )
                metric_rows[before_metrics]["grid_degrees"] = grid_degrees
                for row in prediction_rows[before_predictions:]:
                    row["grid_degrees"] = grid_degrees
                    row["spatial_block"] = block_ids.iloc[row["row_index"]]

            fold_rows.append(
                {
                    "grid_degrees": grid_degrees,
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "train_blocks": int(block_ids.iloc[train_idx].nunique()),
                    "test_blocks": len(test_blocks),
                    "positive_train": int(y.iloc[train_idx].sum()),
                    "positive_test": int(y.iloc[test_idx].sum()),
                    "stable_edges": int(len(stable_edges)),
                    "selected_graph_concepts": int(targets["selected_for_graph_guided_m4"].sum()),
                    "graph_residual_features": int(len(graph_residual_cols)),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    fold_manifest = pd.DataFrame(fold_rows)
    graph_targets = pd.concat(target_rows, ignore_index=True) if target_rows else pd.DataFrame()
    model_summary = summarize(metrics)
    paired = paired_by_grid(metrics)

    exp01.base63.write_table(metrics, output_dir / "spatial_fold_metrics.csv")
    exp01.base63.write_table(predictions, output_dir / "spatial_oof_predictions.csv")
    exp01.base63.write_table(fold_manifest, output_dir / "spatial_fold_manifest.csv")
    exp01.base63.write_table(graph_targets, output_dir / "spatial_fold_graph_targets.csv")
    exp01.base63.write_table(model_summary, output_dir / "spatial_model_summary.csv")
    exp01.base63.write_table(paired, output_dir / "spatial_paired_comparison.csv")

    show_columns = [
        "grid_degrees",
        "model",
        "roc_auc_mean",
        "average_precision_mean",
        "f1_mean",
        "top05_f1_mean",
        "top05_ndcg_mean",
        "top10_f1_mean",
        "top10_ndcg_mean",
    ]
    report = [
        "# Rebuttal 实验 02：最终模型空间块验证",
        "",
        "每个空间网格只属于一个测试折；每折内部重新构图、选择特征并拟合残差器。2° 为主尺度，3° 为尺度敏感性对照。",
        "",
        "## 主结果",
        "",
        exp01.markdown_table(model_summary, show_columns),
        "",
        "## 相对 M2 的逐折差值",
        "",
        exp01.markdown_table(paired),
        "",
        "## 每折样本与图规模",
        "",
        exp01.markdown_table(fold_manifest),
    ]
    (output_dir / "spatial_block_validation_report.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "experiment": "final_model_spatial_block_validation",
        "dataset": run_dir.name,
        "input_dataset": str(dataset_path),
        "grid_degrees": args.grid_degrees,
        "validation": "5-fold GroupKFold by spatial grid",
        "models": MODEL_KEYS,
        "random_state": exp01.base63.RANDOM_STATE,
        "fold_local_steps": [
            "environment stability",
            "causal edge discovery",
            "graph construction",
            "Spearman selection",
            "residualizer fit",
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final-model spatial block validation for rebuttal.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--grid-degrees", type=float, nargs="+", default=[2.0, 3.0])
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
    print(f"Wrote rebuttal experiment 02 outputs to: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
