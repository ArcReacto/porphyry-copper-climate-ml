from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
MAIN_MODELS = ["Full_RF", "NoClimate_RF", "M4_Spearman_RF", "M4_GraphUnion_RF"]
MODEL_LABELS = {
    "Full_RF": "M1 Full climate",
    "NoClimate_RF": "M2 No climate",
    "M4_Spearman_RF": "M4 Spearman residual",
    "M4_GraphUnion_RF": "M4 GraphUnion residual",
}
MODEL_BASELINE_STATUS = {
    "Full_RF": "control",
    "NoClimate_RF": "baseline",
    "M4_Spearman_RF": "method",
    "M4_GraphUnion_RF": "main method candidate",
}
MODEL_ROLES = {
    "Full_RF": "Direct climate-input control using all raw features, including climate variables.",
    "NoClimate_RF": "Primary baseline using all non-climate numeric features.",
    "M4_Spearman_RF": "Data-driven climate decoupling model using fold-local Spearman-selected residualization targets.",
    "M4_GraphUnion_RF": "Graph-guided plus data-driven climate decoupling model using the union of Climate Sensitivity Graph and Spearman targets.",
}


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def round_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    return out


def markdown_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return "_No data._"
    show = round_metrics(df.head(max_rows))
    lines = [
        "| " + " | ".join(map(str, show.columns)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in show.columns) + " |")
    return "\n".join(lines)


def load_inputs(dataset_name: str, input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = input_root / dataset_name
    summary_path = run_dir / "full_feature_graph_guided_m4_summary.csv"
    perturb_path = run_dir / "climate_perturbation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not perturb_path.exists():
        raise FileNotFoundError(perturb_path)
    return pd.read_csv(summary_path), pd.read_csv(perturb_path)


def prepare_main_performance(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "cv",
        "model",
        "roc_auc_mean",
        "average_precision_mean",
        "balanced_accuracy_mean",
        "precision_mean",
        "recall_mean",
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
    out = summary[summary["model"].isin(MAIN_MODELS)][cols].copy()
    out["model_label"] = out["model"].map(MODEL_LABELS)
    out["baseline_status"] = out["model"].map(MODEL_BASELINE_STATUS)
    out["model_role"] = out["model"].map(MODEL_ROLES)
    out = out[
        [
            "cv",
            "model",
            "model_label",
            "baseline_status",
            "model_role",
            "roc_auc_mean",
            "average_precision_mean",
            "balanced_accuracy_mean",
            "precision_mean",
            "recall_mean",
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
    ]
    order = {m: i for i, m in enumerate(MAIN_MODELS)}
    out["_order"] = out["model"].map(order)
    return out.sort_values(["cv", "_order"]).drop(columns="_order").reset_index(drop=True)


def prepare_best_by_metric(main: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
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
    rows = []
    for cv, group in main.groupby("cv", dropna=False):
        for metric in metric_cols:
            best = group.sort_values(metric, ascending=False).iloc[0]
            rows.append(
                {
                    "cv": cv,
                    "metric": metric,
                    "best_model": best["model"],
                    "best_model_label": best["model_label"],
                    "baseline_status": best["baseline_status"],
                    "model_role": best["model_role"],
                    "best_value": best[metric],
                }
            )
    return pd.DataFrame(rows)


def prepare_perturbation(main: pd.DataFrame, perturb: pd.DataFrame) -> pd.DataFrame:
    base = main[
        [
            "cv",
            "model",
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
    ].rename(
        columns={
            "average_precision_mean": "original_average_precision",
            "f1_mean": "original_f1",
            "top05_recall_mean": "original_top05_recall",
            "top05_precision_mean": "original_top05_precision",
            "top05_f1_mean": "original_top05_f1",
            "top05_lift_mean": "original_top05_lift",
            "top05_ndcg_mean": "original_top05_ndcg",
            "top10_recall_mean": "original_top10_recall",
            "top10_precision_mean": "original_top10_precision",
            "top10_f1_mean": "original_top10_f1",
            "top10_lift_mean": "original_top10_lift",
            "top10_ndcg_mean": "original_top10_ndcg",
        }
    )
    cols = [
        "cv",
        "model",
        "scenario",
        "mean_abs_delta_mean",
        "p95_abs_delta_mean",
        "frac_abs_delta_gt_0_05_mean",
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
    available_cols = [c for c in cols if c in perturb.columns]
    out = perturb[perturb["model"].isin(MAIN_MODELS)][available_cols].copy()
    out = out.merge(base, on=["cv", "model"], how="left")
    out["model_label"] = out["model"].map(MODEL_LABELS)
    out["baseline_status"] = out["model"].map(MODEL_BASELINE_STATUS)
    out["model_role"] = out["model"].map(MODEL_ROLES)
    out["delta_average_precision_vs_original"] = out["average_precision_mean"] - out["original_average_precision"]
    out["delta_f1_vs_original"] = out["f1_mean"] - out["original_f1"]
    if "top05_precision_mean" in out.columns:
        out["delta_top05_precision_vs_original"] = out["top05_precision_mean"] - out["original_top05_precision"]
    if "top05_recall_mean" in out.columns:
        out["delta_top05_recall_vs_original"] = out["top05_recall_mean"] - out["original_top05_recall"]
    if "top05_f1_mean" in out.columns:
        out["delta_top05_f1_vs_original"] = out["top05_f1_mean"] - out["original_top05_f1"]
    if "top05_ndcg_mean" in out.columns:
        out["delta_top05_ndcg_vs_original"] = out["top05_ndcg_mean"] - out["original_top05_ndcg"]
    if "top10_recall_mean" in out.columns:
        out["delta_top10_recall_vs_original"] = out["top10_recall_mean"] - out["original_top10_recall"]
    if "top10_ndcg_mean" in out.columns:
        out["delta_top10_ndcg_vs_original"] = out["top10_ndcg_mean"] - out["original_top10_ndcg"]
    order = {m: i for i, m in enumerate(MAIN_MODELS)}
    out["_order"] = out["model"].map(order)
    return out.sort_values(["cv", "_order", "scenario"]).drop(columns="_order").reset_index(drop=True)


def prepare_robustness_overview(perturbation: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "mean_abs_delta_mean",
        "p95_abs_delta_mean",
        "frac_abs_delta_gt_0_05_mean",
        "delta_average_precision_vs_original",
        "delta_f1_vs_original",
        "delta_top05_precision_vs_original",
        "delta_top05_recall_vs_original",
        "delta_top05_f1_vs_original",
        "delta_top05_ndcg_vs_original",
        "delta_top10_recall_vs_original",
        "delta_top10_ndcg_vs_original",
    ]
    available_cols = [c for c in cols if c in perturbation.columns]
    out = perturbation.groupby(
        ["cv", "model", "model_label", "baseline_status", "model_role"],
        dropna=False,
    )[available_cols].mean().reset_index()
    order = {m: i for i, m in enumerate(MAIN_MODELS)}
    out["_order"] = out["model"].map(order)
    return out.sort_values(["cv", "_order"]).drop(columns="_order").reset_index(drop=True)


def write_report(
    out_dir: Path,
    dataset_name: str,
    main: pd.DataFrame,
    best: pd.DataFrame,
    perturb: pd.DataFrame,
    overview: pd.DataFrame,
) -> None:
    group_main = main[main["cv"] == "groupkfold_state"].copy()
    group_best = best[best["cv"] == "groupkfold_state"].copy()
    group_overview = overview[overview["cv"] == "groupkfold_state"].copy()
    report = [
        f"# Final Main Experiment Tables: {dataset_name}",
        "",
        "## Purpose",
        "",
        "This report treats climate perturbation as a robustness evaluation, not as training augmentation.",
        "",
        "Main model comparison:",
        "",
        "- `Full_RF`: direct climate-input control using all features including climate.",
        "- `NoClimate_RF`: primary baseline after removing climate variables.",
        "- `M4_Spearman_RF`: data-driven climate-sensitive residualization using fold-local Spearman screening.",
        "- `M4_GraphUnion_RF`: main method candidate using the union of Climate Sensitivity Graph targets and Spearman-selected targets.",
        "",
        "## Main Performance: GroupKFold by State",
        "",
        markdown_table(group_main),
        "",
        "## Best Model by Metric: GroupKFold by State",
        "",
        markdown_table(group_best),
        "",
        "## Perturbation Robustness Overview: GroupKFold by State",
        "",
        markdown_table(group_overview),
        "",
        "## Detailed Perturbation Table: GroupKFold by State",
        "",
        markdown_table(perturb[perturb["cv"] == "groupkfold_state"], max_rows=80),
        "",
        "## Interpretation",
        "",
        "- `NoClimate_RF` is naturally insensitive to climate perturbation because climate inputs are absent.",
        "- `M4_Spearman_RF` gives the strongest GroupKFold ROC-AUC, F1, and Top-5% recall in the no-augmentation setting.",
        "- `M4_GraphUnion_RF` gives the strongest GroupKFold AP/AUPRC and Top-10% recall, making it the best candidate for the paper's graph-guided decoupling method.",
        "- Climate perturbation is kept as a robustness audit because naive counterfactual augmentation reduced original-test discrimination.",
    ]
    (out_dir / "final_main_experiment_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare final paper-ready tables for main no-augmentation robustness setup.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument(
        "--input-root",
        default=str(PROJECT_ROOT / "outputs" / "paper_optimization" / "full_feature_graph_guided_m4"),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "paper_optimization" / "final_main_experiment" / args.dataset_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, perturb = load_inputs(args.dataset_name, input_root)
    main_perf = prepare_main_performance(summary)
    best = prepare_best_by_metric(main_perf)
    perturb_table = prepare_perturbation(main_perf, perturb)
    overview = prepare_robustness_overview(perturb_table)

    write_table(round_metrics(main_perf), out_dir / "table_main_model_performance.csv")
    write_table(round_metrics(best), out_dir / "table_best_model_by_metric.csv")
    write_table(round_metrics(perturb_table), out_dir / "table_climate_perturbation_detail.csv")
    write_table(round_metrics(overview), out_dir / "table_climate_perturbation_overview.csv")
    write_report(out_dir, args.dataset_name, main_perf, best, perturb_table, overview)

    manifest = {
        "dataset_name": args.dataset_name,
        "input_root": str(input_root),
        "output_dir": str(out_dir),
        "models": MAIN_MODELS,
        "outputs": [
            "table_main_model_performance.csv",
            "table_best_model_by_metric.csv",
            "table_climate_perturbation_detail.csv",
            "table_climate_perturbation_overview.csv",
            "final_main_experiment_report.md",
        ],
    }
    (out_dir / "final_main_experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote final main experiment tables to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
