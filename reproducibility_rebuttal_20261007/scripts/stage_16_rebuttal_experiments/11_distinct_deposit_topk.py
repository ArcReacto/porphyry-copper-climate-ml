from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "rebuttal_experiments"
    / "xgboost"
    / "05_final_ratio_sensitivity"
    / "ratio_oof_predictions.csv"
)
DEFAULT_CLUSTERS = (
    PROJECT_ROOT
    / "outputs"
    / "rebuttal_experiments"
    / "06_deposit_dedup_buffer_audit"
    / "deposit_clusters.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rebuttal_experiments" / "xgboost" / "11_distinct_deposit_topk"


def evaluate_group(part: pd.DataFrame, budget: float) -> dict:
    ranked = part.sort_values(["y_prob", "row_index"], ascending=[False, True]).reset_index(drop=True)
    k = max(1, int(math.ceil(len(ranked) * budget)))
    selected = ranked.head(k)
    positive = part[part["Y_label"].eq(1)]
    positive_groups = set(positive["deposit_group_id"].astype(str))
    hit_rows = selected[selected["Y_label"].eq(1)]
    hit_groups = set(hit_rows["deposit_group_id"].astype(str))
    unique_hits = len(hit_groups)
    total_groups = len(positive_groups)
    return {
        "budget_fraction": budget,
        "n_test_rows": len(part),
        "k_rows": k,
        "positive_rows": int(part["Y_label"].sum()),
        "positive_rows_hit": int(hit_rows.shape[0]),
        "distinct_positive_deposits": total_groups,
        "distinct_deposits_hit": unique_hits,
        "distinct_deposit_recall": unique_hits / total_groups if total_groups else float("nan"),
        "distinct_deposit_precision_per_candidate": unique_hits / k,
        "duplicate_positive_hits": int(hit_rows.shape[0] - unique_hits),
        "duplicate_fraction_among_positive_hits": (
            (hit_rows.shape[0] - unique_hits) / hit_rows.shape[0] if hit_rows.shape[0] else 0.0
        ),
    }


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    show = frame.copy()
    for column in show.select_dtypes(include=["float"]).columns:
        show[column] = show[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    lines = [
        "| " + " | ".join(show.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for row in show.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions, low_memory=False)
    predictions = predictions[
        predictions["negative_ratio"].astype(str).eq("1:10")
        & predictions["model"].isin(["NoClimate_XGB", "M4_Spearman_XGB"])
    ].copy()
    clusters = pd.read_csv(args.clusters, low_memory=False)
    clusters = clusters[clusters["threshold_km"].eq(args.cluster_threshold_km)][
        ["row_index", "cluster_id", "cluster_size", "states", "labels"]
    ].drop_duplicates("row_index")
    if clusters["row_index"].duplicated().any():
        raise ValueError("Cluster table has duplicate row_index entries.")
    predictions = predictions.merge(clusters, on="row_index", how="left", validate="many_to_one")
    if predictions["cluster_id"].isna().any():
        raise ValueError("Some prediction rows could not be mapped to a deposit cluster.")
    predictions["deposit_group_id"] = predictions["cluster_id"].astype(int).map(lambda value: f"cluster_{value}")

    rows: list[dict] = []
    selected_rows: list[pd.DataFrame] = []
    for (model, fold, test_group), part in predictions.groupby(["model", "fold", "test_group"], sort=True):
        for budget in args.budgets:
            result = evaluate_group(part, budget)
            result.update({"model": model, "fold": fold, "test_group": test_group})
            rows.append(result)
            k = result["k_rows"]
            selected = part.sort_values(["y_prob", "row_index"], ascending=[False, True]).head(k).copy()
            selected.insert(0, "budget_fraction", budget)
            selected_rows.append(selected)

    fold_metrics = pd.DataFrame(rows)
    selected = pd.concat(selected_rows, ignore_index=True)
    summary = (
        fold_metrics.groupby(["model", "budget_fraction"])[
            [
                "k_rows",
                "positive_rows_hit",
                "distinct_positive_deposits",
                "distinct_deposits_hit",
                "distinct_deposit_recall",
                "distinct_deposit_precision_per_candidate",
                "duplicate_positive_hits",
                "duplicate_fraction_among_positive_hits",
            ]
        ]
        .agg(["sum", "mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(piece) for piece in column if str(piece)) if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    pooled_rows: list[dict] = []
    for (model, budget), part in fold_metrics.groupby(["model", "budget_fraction"]):
        total_deposits = int(part["distinct_positive_deposits"].sum())
        hits = int(part["distinct_deposits_hit"].sum())
        k_rows = int(part["k_rows"].sum())
        positive_hits = int(part["positive_rows_hit"].sum())
        pooled_rows.append(
            {
                "model": model,
                "budget_fraction": budget,
                "candidate_rows": k_rows,
                "positive_rows_hit": positive_hits,
                "distinct_positive_deposits": total_deposits,
                "distinct_deposits_hit": hits,
                "distinct_deposit_recall": hits / total_deposits if total_deposits else float("nan"),
                "distinct_deposit_precision_per_candidate": hits / k_rows,
                "duplicate_positive_hits": positive_hits - hits,
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    fold_metrics.to_csv(output_dir / "distinct_deposit_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "distinct_deposit_summary.csv", index=False)
    pooled.to_csv(output_dir / "distinct_deposit_pooled_foldwise.csv", index=False)
    selected.to_csv(output_dir / "distinct_deposit_topk_rows.csv", index=False)

    report = f"""# Rebuttal 新实验 N4：正样本空间代理簇 Top-K 评价

## 协议

- 输入：1:10 主实验的州折 OOF 预测；
- 模型：M2 NoClimate 与 M4 Spearman；
- 代理簇归并：{args.cluster_threshold_km:g} km 球面距离簇；
- 同一空间簇在 Top-K 中重复出现时只计算一次代理簇命中；
- Top-K 仍按每个测试折的样本行数计算，随后汇总代理簇召回。

## 汇总结果

{markdown_table(pooled)}

## 解释边界

该指标减少邻近正样本记录重复计数的影响，但空间簇并不等同于权威矿床或成矿带边界。输出列名中的 `deposit` 是历史字段名；对外只能称为“正样本空间代理簇”，不能称为已验证的独立矿床覆盖。
"""
    (output_dir / "distinct_deposit_topk_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "predictions": str(Path(args.predictions)),
        "clusters": str(Path(args.clusters)),
        "cluster_threshold_km": args.cluster_threshold_km,
        "budgets": args.budgets,
        "models": ["NoClimate_XGB", "M4_Spearman_XGB"],
        "unit_warning": "Distance-based positive-sample proxy clusters, not verified independent deposits",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote N4 outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distinct deposit Top-K evaluation from OOF predictions.")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--clusters", default=str(DEFAULT_CLUSTERS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cluster-threshold-km", type=float, default=5.0)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.05, 0.10])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
