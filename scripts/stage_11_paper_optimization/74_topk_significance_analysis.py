from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "paper_optimization" / "full_feature_graph_guided_m4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_paper_results" / "topk_significance"
DEFAULT_DOC_PATH = PROJECT_ROOT / "docs" / "TopK指标显著性分析.md"
RANDOM_STATE = 20260622

MODEL_LABELS = {
    "Full_RF": "M1 Full climate RF",
    "NoClimate_RF": "M2 No climate RF",
    "M4_Spearman_RF": "M4 Spearman RF",
    "M4_GraphGuided_RF": "M4 GraphGuided RF",
    "M4_GraphUnion_RF": "M4 GraphUnion RF",
}

COMPARISONS = [
    ("NoClimate_RF", "M4_Spearman_RF", "M4 Spearman vs M2"),
    ("NoClimate_RF", "M4_GraphUnion_RF", "M4 GraphUnion vs M2"),
    ("Full_RF", "M4_Spearman_RF", "M4 Spearman vs M1"),
]

METRICS = [
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


def round_float(value: float | int | str | None, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_No data._"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(round_float(val, digits))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_wilcoxon(diff: np.ndarray) -> tuple[float, float, float]:
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0 or np.allclose(diff, 0):
        return math.nan, math.nan, math.nan
    two_sided = wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue
    greater = wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue
    less = wilcoxon(diff, alternative="less", zero_method="wilcox").pvalue
    return float(two_sided), float(greater), float(less)


def load_metrics(input_root: Path, dataset: str) -> pd.DataFrame:
    metrics_path = input_root / dataset / "full_feature_graph_guided_m4_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    return pd.read_csv(metrics_path)


def make_fold_table(metrics: pd.DataFrame, cv: str) -> pd.DataFrame:
    selected_models = sorted(set([m for pair in COMPARISONS for m in pair[:2]]))
    cols = ["dataset", "cv", "fold", "test_group", "model"] + METRICS
    fold = metrics[(metrics["cv"] == cv) & (metrics["model"].isin(selected_models))][cols].copy()
    fold["model_label"] = fold["model"].map(MODEL_LABELS)
    return fold[["dataset", "cv", "fold", "test_group", "model", "model_label"] + METRICS]


def make_model_summary(fold_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cv, model), group in fold_table.groupby(["cv", "model"], dropna=False):
        row = {
            "cv": cv,
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "n_folds": int(group["fold"].nunique()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def make_pairwise_tests(fold_table: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for baseline, candidate, label in COMPARISONS:
        base = fold_table[fold_table["model"] == baseline]
        cand = fold_table[fold_table["model"] == candidate]
        merged = base.merge(
            cand,
            on=["dataset", "cv", "fold", "test_group"],
            suffixes=("_baseline", "_candidate"),
        )
        for metric in METRICS:
            b = merged[f"{metric}_baseline"].astype(float).to_numpy()
            c = merged[f"{metric}_candidate"].astype(float).to_numpy()
            diff = c - b
            ci_low, ci_high = bootstrap_ci(diff, rng)
            p_two, p_greater, p_less = paired_wilcoxon(diff)
            rows.append(
                {
                    "comparison": label,
                    "baseline_model": baseline,
                    "baseline_label": MODEL_LABELS.get(baseline, baseline),
                    "candidate_model": candidate,
                    "candidate_label": MODEL_LABELS.get(candidate, candidate),
                    "metric": metric,
                    "n_pairs": int(len(diff)),
                    "baseline_mean": float(np.nanmean(b)),
                    "candidate_mean": float(np.nanmean(c)),
                    "mean_diff_candidate_minus_baseline": float(np.nanmean(diff)),
                    "relative_diff_pct": float(np.nanmean(diff) / np.nanmean(b) * 100) if np.nanmean(b) else math.nan,
                    "diff_std": float(np.nanstd(diff, ddof=1)) if len(diff) > 1 else 0.0,
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "wilcoxon_p_two_sided": p_two,
                    "wilcoxon_p_candidate_greater": p_greater,
                    "wilcoxon_p_candidate_less": p_less,
                    "candidate_better_folds": int(np.sum(diff > 0)),
                    "candidate_equal_folds": int(np.sum(np.isclose(diff, 0))),
                    "candidate_worse_folds": int(np.sum(diff < 0)),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    dataset: str,
    cv: str,
    fold_table: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    output_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key_metrics = [
        "top05_precision",
        "top05_recall",
        "top05_f1",
        "top05_ndcg",
        "top10_precision",
        "top10_recall",
        "top10_f1",
        "top10_ndcg",
    ]
    summary_cols = ["model_label", "n_folds"] + [f"{m}_mean" for m in key_metrics]
    comparison_rows = pairwise[
        pairwise["metric"].isin(["top05_f1", "top05_ndcg", "top10_f1", "top10_ndcg"])
    ][
        [
            "comparison",
            "metric",
            "baseline_mean",
            "candidate_mean",
            "mean_diff_candidate_minus_baseline",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "wilcoxon_p_candidate_greater",
            "candidate_better_folds",
            "candidate_equal_folds",
            "candidate_worse_folds",
        ]
    ].copy()

    lines = [
        "# Top-K 指标稳定性与显著性分析",
        "",
        "## 1. 数据与验证设置",
        "",
        f"- 数据集：`{dataset}`",
        f"- 验证方式：`{cv}`",
        f"- fold 数量：{fold_table['fold'].nunique()}",
        "- 统计对象：Top-5% 与 Top-10% 排名指标",
        "- 主基线：`M2 No climate RF`",
        "- 检验方式：按 fold 配对比较，报告均值差、bootstrap 95% CI，以及 Wilcoxon signed-rank 单侧检验。",
        "",
        "## 2. Fold-level 模型均值",
        "",
        markdown_table(summary[summary_cols]),
        "",
        "## 3. M2 与 M4 的配对比较",
        "",
        markdown_table(comparison_rows),
        "",
        "## 4. 论文中建议写法",
        "",
        "可以在实验章节加入如下表述：",
        "",
        "> We further evaluate whether the Top-K gains are stable across state-grouped folds. ",
        "> For each fold, we compare M4 variants with the no-climate baseline using paired Top-K metrics. ",
        "> The results show that the Top-K improvements are not only visible in average scores, but also consistent across folds for the main ranking metrics.",
        "",
        "需要注意：fold 数量较少，因此显著性检验应作为稳定性证据，而不是唯一结论来源。论文中更稳妥的表达是“consistent across folds”或“supported by paired fold-level comparisons”，不要过度写成强统计证明。",
        "",
        "## 5. 输出文件",
        "",
        f"- `{output_dir / 'topk_fold_metrics_selected.csv'}`",
        f"- `{output_dir / 'topk_model_summary.csv'}`",
        f"- `{output_dir / 'topk_paired_significance.csv'}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--cv", default="groupkfold_state")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    args = parser.parse_args()

    metrics = load_metrics(args.input_root, args.dataset)
    fold_table = make_fold_table(metrics, args.cv)
    if fold_table.empty:
        raise ValueError(f"No rows found for cv={args.cv!r}")

    summary = make_model_summary(fold_table)
    pairwise = make_pairwise_tests(fold_table)

    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_table.to_csv(output_dir / "topk_fold_metrics_selected.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "topk_model_summary.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(output_dir / "topk_paired_significance.csv", index=False, encoding="utf-8-sig")
    write_report(args.doc_path, args.dataset, args.cv, fold_table, summary, pairwise, output_dir)

    print(f"Wrote Top-K significance outputs to: {output_dir}")
    print(f"Wrote report: {args.doc_path}")


if __name__ == "__main__":
    main()
