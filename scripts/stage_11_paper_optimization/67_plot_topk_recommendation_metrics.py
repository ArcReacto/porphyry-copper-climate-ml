from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
MAIN_MODELS = ["Full_RF", "NoClimate_RF", "M4_Spearman_RF", "M4_GraphUnion_RF"]
MODEL_LABELS = {
    "Full_RF": "M1 直接加气候",
    "NoClimate_RF": "M2 去气候基线",
    "M4_Spearman_RF": "M4 数据驱动解耦",
    "M4_GraphUnion_RF": "M4 图+数据解耦",
}
MODEL_COLORS = {
    "Full_RF": "#9CA3AF",
    "NoClimate_RF": "#4B5563",
    "M4_Spearman_RF": "#2563EB",
    "M4_GraphUnion_RF": "#F97316",
}
TOP_KS = [1, 5, 10, 20]
METRIC_LABELS = {
    "precision": "Precision@K",
    "recall": "Recall@K",
    "f1": "F1@K",
    "lift": "Lift@K",
    "ndcg": "NDCG@K",
}


def configure_matplotlib() -> None:
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.labelcolor"] = "#111827"
    plt.rcParams["xtick.color"] = "#374151"
    plt.rcParams["ytick.color"] = "#374151"


def read_metrics(dataset_name: str, input_root: Path) -> pd.DataFrame:
    metrics_path = input_root / dataset_name / "full_feature_graph_guided_m4_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    df = pd.read_csv(metrics_path)
    df = df[(df["cv"] == "groupkfold_state") & (df["model"].isin(MAIN_MODELS))].copy()
    if df.empty:
        raise ValueError("No GroupKFold rows found for main models.")
    return df


def summarize_topk(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in metrics.groupby("model", sort=False):
        for k in TOP_KS:
            prefix = f"top{str(k).zfill(2)}"
            for metric in ["precision", "recall", "f1", "lift", "ndcg"]:
                col = f"{prefix}_{metric}"
                if col not in group.columns:
                    continue
                rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "k_percent": k,
                        "metric": metric,
                        "mean": group[col].mean(),
                        "std": group[col].std(),
                        "count": group[col].count(),
                    }
                )
    return pd.DataFrame(rows)


def summarize_main(metrics: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "roc_auc",
        "average_precision",
        "f1",
        "top05_precision",
        "top05_recall",
        "top05_f1",
        "top05_lift",
        "top05_ndcg",
        "top10_recall",
        "top10_ndcg",
    ]
    rows = []
    for model, group in metrics.groupby("model", sort=False):
        row = {"model": model, "model_label": MODEL_LABELS[model]}
        for col in wanted:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std()
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_order"] = out["model"].map({m: i for i, m in enumerate(MAIN_MODELS)})
    return out.sort_values("_order").drop(columns="_order")


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_topk_curves(topk: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["precision", "recall", "f1", "ndcg"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metrics):
        sub = topk[topk["metric"] == metric]
        for model in MAIN_MODELS:
            model_sub = sub[sub["model"] == model].sort_values("k_percent")
            ax.plot(
                model_sub["k_percent"],
                model_sub["mean"],
                marker="o",
                linewidth=2.4,
                markersize=5.5,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )
        ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight="bold", color="#111827")
        ax.set_xlabel("候选区比例 K（%）")
        ax.set_ylabel("指标值")
        ax.set_xticks(TOP_KS)
        ax.set_ylim(0, 1.02)
        style_axis(ax)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Top-K 找矿候选区排序指标曲线（GroupKFold）", fontsize=17, fontweight="bold", y=1.10)
    save_figure(fig, out_dir / "topk_precision_recall_f1_ndcg_curves.png")


def plot_lift_curve(topk: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.8), constrained_layout=True)
    sub = topk[topk["metric"] == "lift"]
    for model in MAIN_MODELS:
        model_sub = sub[sub["model"] == model].sort_values("k_percent")
        ax.plot(
            model_sub["k_percent"],
            model_sub["mean"],
            marker="o",
            linewidth=2.8,
            markersize=6,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
        if model in {"M4_Spearman_RF", "M4_GraphUnion_RF"}:
            for _, row in model_sub.iterrows():
                if int(row["k_percent"]) in {5, 10}:
                    ax.annotate(
                        f"{row['mean']:.2f}x",
                        (row["k_percent"], row["mean"]),
                        textcoords="offset points",
                        xytext=(0, 9),
                        ha="center",
                        fontsize=9,
                        color=MODEL_COLORS[model],
                    )
    ax.set_title("Lift 曲线：高分候选区相对随机选点的富集倍数", fontsize=15, fontweight="bold")
    ax.set_xlabel("候选区比例 K（%）")
    ax.set_ylabel("Lift@K（倍）")
    ax.set_xticks(TOP_KS)
    style_axis(ax)
    ax.legend(loc="upper right", frameon=False)
    save_figure(fig, out_dir / "topk_lift_curve.png")


def plot_top5_bars(main: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("top05_precision_mean", "Precision@5%"),
        ("top05_recall_mean", "Recall@5%"),
        ("top05_f1_mean", "F1@5%"),
        ("top05_ndcg_mean", "NDCG@5%"),
        ("top05_lift_mean", "Lift@5%"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16.2, 4.8), constrained_layout=True)
    x = np.arange(len(MAIN_MODELS))
    labels = [MODEL_LABELS[m] for m in MAIN_MODELS]
    colors = [MODEL_COLORS[m] for m in MAIN_MODELS]
    for ax, (col, title) in zip(axes, metrics):
        values = [float(main.loc[main["model"] == m, col].iloc[0]) for m in MAIN_MODELS]
        bars = ax.bar(x, values, color=colors, width=0.68)
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        if col != "top05_lift_mean":
            ax.set_ylim(0, 1.02)
        style_axis(ax)
        for bar, value in zip(bars, values):
            suffix = "x" if col == "top05_lift_mean" else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}{suffix}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#111827",
            )
    fig.suptitle("Top-5% 高优先级找矿候选区指标对比（GroupKFold）", fontsize=17, fontweight="bold", y=1.08)
    save_figure(fig, out_dir / "top5_recommendation_metrics_bars.png")


def plot_main_summary(main: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("roc_auc_mean", "ROC-AUC"),
        ("average_precision_mean", "AP / AUPRC"),
        ("f1_mean", "F1"),
        ("top05_f1_mean", "F1@5%"),
        ("top05_ndcg_mean", "NDCG@5%"),
        ("top10_ndcg_mean", "NDCG@10%"),
    ]
    fig, ax = plt.subplots(figsize=(12.8, 6.4), constrained_layout=True)
    x = np.arange(len(metrics))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MAIN_MODELS))
    for model, offset in zip(MAIN_MODELS, offsets):
        values = [float(main.loc[main["model"] == model, col].iloc[0]) for col, _ in metrics]
        ax.bar(
            x + offset,
            values,
            width=width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
    ax.set_title("整体分类指标与头部排序指标联合对比（GroupKFold）", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("指标值")
    style_axis(ax)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    save_figure(fig, out_dir / "main_performance_and_topk_summary.png")


def write_report(out_dir: Path, dataset_name: str, main: pd.DataFrame) -> None:
    group = main.copy()
    cols = [
        "model",
        "model_label",
        "roc_auc_mean",
        "average_precision_mean",
        "f1_mean",
        "top05_precision_mean",
        "top05_recall_mean",
        "top05_f1_mean",
        "top05_lift_mean",
        "top05_ndcg_mean",
        "top10_recall_mean",
        "top10_ndcg_mean",
    ]
    table = group[cols].round(4)
    markdown_rows = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        markdown_rows.append("| " + " | ".join(str(row[col]) for col in table.columns) + " |")
    lines = [
        "# Top-K 搜索推荐指标图表",
        "",
        f"数据集：`{dataset_name}`",
        "",
        "验证方式：`groupkfold_state`",
        "",
        "## 输出图",
        "",
        "- `topk_precision_recall_f1_ndcg_curves.png`：Precision、Recall、F1、NDCG 的 Top-K 曲线。",
        "- `topk_lift_curve.png`：Lift@K 曲线，用于解释高分候选区相对随机选点的富集倍数。",
        "- `top5_recommendation_metrics_bars.png`：Top-5% 头部候选区指标柱状对比。",
        "- `main_performance_and_topk_summary.png`：整体分类指标与头部排序指标联合对比。",
        "",
        "## 主结果表",
        "",
        "\n".join(markdown_rows),
        "",
        "## 简要结论",
        "",
        "- `M4_Spearman_RF` 在 Top-5% 的 Precision、Recall、F1、Lift 和 NDCG 上均为最优，适合强调极低钻探预算下的头部靶区筛选。",
        "- `M4_GraphUnion_RF` 在 AP/AUPRC 和 Top-10% NDCG 上表现更强，适合强调因果图引导方法对较宽候选区排序的稳定贡献。",
    ]
    (out_dir / "topk_recommendation_figures_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Top-K recommendation metrics for mining-target ranking.")
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
        else PROJECT_ROOT / "outputs" / "paper_optimization" / "topk_recommendation_figures" / args.dataset_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    metrics = read_metrics(args.dataset_name, input_root)
    topk = summarize_topk(metrics)
    main = summarize_main(metrics)
    topk.to_csv(out_dir / "topk_curve_values.csv", index=False, encoding="utf-8-sig")
    main.round(6).to_csv(out_dir / "topk_main_metric_values.csv", index=False, encoding="utf-8-sig")

    plot_topk_curves(topk, out_dir)
    plot_lift_curve(topk, out_dir)
    plot_top5_bars(main, out_dir)
    plot_main_summary(main, out_dir)
    write_report(out_dir, args.dataset_name, main)
    print(f"Wrote Top-K recommendation figures to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
