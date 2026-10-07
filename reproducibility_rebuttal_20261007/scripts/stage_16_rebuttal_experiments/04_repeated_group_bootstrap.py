from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_PREDICTIONS = (
    PROJECT_ROOT / "outputs" / "rebuttal_experiments" / "xgboost" / "01_fold_local_graphunion" / "oof_predictions.csv"
)
DEFAULT_SPATIAL_PREDICTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "rebuttal_experiments"
    / "xgboost"
    / "02_spatial_block_validation"
    / "spatial_oof_predictions.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rebuttal_experiments" / "xgboost" / "04_repeated_group_bootstrap"
BASELINE = "M2_NoClimate_XGB"
CANDIDATE = "M4_Spearman_XGB"
METRICS = ["roc_auc", "average_precision", "f1", "top05_f1", "top05_ndcg", "top10_f1", "top10_ndcg"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(
    "bootstrap_base",
    PROJECT_ROOT
    / "scripts"
    / "stage_11_paper_optimization"
    / "63_full_feature_graph_guided_m4_and_perturbation.py",
)


def metric_values(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    values = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    values.update(base.top_k_metrics(y_true, y_prob))
    return {metric: values[metric] for metric in METRICS}


def paired_frame(predictions: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    selected = predictions[predictions["model"].isin([BASELINE, CANDIDATE])].copy()
    keys = ["row_index", cluster_col, "Y_label"]
    if "grid_degrees" in selected.columns and cluster_col == "spatial_block":
        keys.insert(0, "grid_degrees")
    wide = selected.pivot_table(index=keys, columns="model", values="y_prob", aggfunc="first").reset_index()
    if wide[[BASELINE, CANDIDATE]].isna().any().any():
        raise ValueError("OOF predictions are not paired for all rows.")
    return wide


def bootstrap_one_protocol(
    frame: pd.DataFrame,
    protocol: str,
    cluster_col: str,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    clusters = np.asarray(sorted(frame[cluster_col].astype(str).unique().tolist()))
    cluster_indices = {
        cluster: frame.index[frame[cluster_col].astype(str) == cluster].to_numpy() for cluster in clusters
    }
    original_rows = []
    y_original = frame["Y_label"].astype(int).to_numpy()
    original_metrics: dict[str, dict[str, float]] = {}
    for model in [BASELINE, CANDIDATE]:
        original_metrics[model] = metric_values(y_original, frame[model].astype(float).to_numpy())
    for metric in METRICS:
        original_rows.append(
            {
                "protocol": protocol,
                "metric": metric,
                "baseline_value": original_metrics[BASELINE][metric],
                "candidate_value": original_metrics[CANDIDATE][metric],
                "candidate_minus_baseline": original_metrics[CANDIDATE][metric] - original_metrics[BASELINE][metric],
                "cluster_count": len(clusters),
                "row_count": len(frame),
            }
        )

    bootstrap_rows = []
    for iteration in range(iterations):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_indices = np.concatenate([cluster_indices[cluster] for cluster in sampled_clusters])
        sampled = frame.loc[sampled_indices]
        y_true = sampled["Y_label"].astype(int).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        base_values = metric_values(y_true, sampled[BASELINE].astype(float).to_numpy())
        candidate_values = metric_values(y_true, sampled[CANDIDATE].astype(float).to_numpy())
        for metric in METRICS:
            bootstrap_rows.append(
                {
                    "protocol": protocol,
                    "iteration": iteration,
                    "metric": metric,
                    "baseline_value": base_values[metric],
                    "candidate_value": candidate_values[metric],
                    "candidate_minus_baseline": candidate_values[metric] - base_values[metric],
                    "sampled_cluster_count": len(sampled_clusters),
                    "sampled_unique_clusters": len(set(sampled_clusters)),
                    "sampled_rows": len(sampled),
                }
            )
    return pd.DataFrame(original_rows), pd.DataFrame(bootstrap_rows)


def summarize(original: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (protocol, metric), group in bootstrap.groupby(["protocol", "metric"]):
        point = original[(original["protocol"] == protocol) & (original["metric"] == metric)].iloc[0]
        delta = group["candidate_minus_baseline"].astype(float).to_numpy()
        rows.append(
            {
                "protocol": protocol,
                "metric": metric,
                "baseline_oof": float(point["baseline_value"]),
                "candidate_oof": float(point["candidate_value"]),
                "delta_oof": float(point["candidate_minus_baseline"]),
                "bootstrap_delta_mean": float(np.mean(delta)),
                "bootstrap_ci95_low": float(np.quantile(delta, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(delta, 0.975)),
                "probability_delta_gt_zero": float(np.mean(delta > 0)),
                "valid_iterations": int(group["iteration"].nunique()),
                "cluster_count": int(point["cluster_count"]),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    show = df.copy()
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


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_predictions = pd.read_csv(args.state_predictions, low_memory=False)
    spatial_predictions = pd.read_csv(args.spatial_predictions, low_memory=False)

    originals = []
    bootstraps = []
    state_frame = paired_frame(state_predictions, "state")
    original, bootstrap = bootstrap_one_protocol(
        state_frame, "state_grouped_oof", "state", args.iterations, args.seed
    )
    originals.append(original)
    bootstraps.append(bootstrap)

    for grid_degrees in sorted(spatial_predictions["grid_degrees"].unique()):
        current = spatial_predictions[spatial_predictions["grid_degrees"] == grid_degrees].copy()
        spatial_frame = paired_frame(current, "spatial_block")
        protocol = f"spatial_{grid_degrees:g}deg_oof"
        original, bootstrap = bootstrap_one_protocol(
            spatial_frame,
            protocol,
            "spatial_block",
            args.iterations,
            args.seed + int(grid_degrees * 100),
        )
        originals.append(original)
        bootstraps.append(bootstrap)

    original_table = pd.concat(originals, ignore_index=True)
    bootstrap_table = pd.concat(bootstraps, ignore_index=True)
    summary = summarize(original_table, bootstrap_table)
    base.write_table(original_table, output_dir / "oof_point_estimates.csv")
    base.write_table(bootstrap_table, output_dir / "cluster_bootstrap_iterations.csv")
    base.write_table(summary, output_dir / "cluster_bootstrap_summary.csv")

    report = [
        "# Rebuttal 实验 04：聚类 Bootstrap 不确定性",
        "",
        "本实验不重新训练模型。它对现有 OOF 预测按州或空间网格整体重采样，用于估计 M4 Spearman 相对 M2 的差值不确定性。CI 包含 0 表示现有样本不足以支持方向稳定的提升。",
        "",
        markdown_table(summary),
    ]
    (output_dir / "repeated_group_bootstrap_report.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "experiment": "cluster_bootstrap_on_oof_predictions",
        "state_predictions": str(Path(args.state_predictions)),
        "spatial_predictions": str(Path(args.spatial_predictions)),
        "iterations": args.iterations,
        "seed": args.seed,
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "scope": "uncertainty of fixed OOF predictions; models are not retrained per bootstrap sample",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster bootstrap M4 versus M2 OOF differences.")
    parser.add_argument("--state-predictions", default=str(DEFAULT_STATE_PREDICTIONS))
    parser.add_argument("--spatial-predictions", default=str(DEFAULT_SPATIAL_PREDICTIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260622)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args)
    print(f"Wrote rebuttal experiment 04 outputs to: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
