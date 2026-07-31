from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shapefile
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon

from cdmpm_figure_style import COLORS, save_figure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("CDMPM_DATA_ROOT", PROJECT_ROOT.parent / "data_raw"))
OUT_DIR = PROJECT_ROOT / "figures"

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_datasets"
    / "by_sample_scheme"
    / "known_mining_neutral"
    / "known_mining_neutral_ratio_1_10"
    / "model_dataset_known_mining_neutral_ratio_1_10_supervised_all_features_v1.parquet"
)
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "paper_optimization"
    / "full_feature_graph_guided_m4"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
    / "full_feature_graph_guided_m4_predictions.csv"
)
METRICS_PATH = PREDICTIONS_PATH.with_name("full_feature_graph_guided_m4_metrics.csv")
STATES_SHP = DATA_ROOT / "cb_2025_us_state_500k" / "cb_2025_us_state_500k.shp"

CV = "groupkfold_state"
FOLD = 1
STATE_NAME = "Arizona"
LEFT_MODEL = "NoClimate_RF"
RIGHT_MODEL = "M4_GraphUnion_RF"
MODEL_LABELS = {
    LEFT_MODEL: "M2 No-climate RF",
    RIGHT_MODEL: "M4 GraphUnion RF",
}


def shape_to_patches(shape) -> list[MplPolygon]:
    points = shape.points
    parts = list(shape.parts) + [len(points)]
    patches: list[MplPolygon] = []
    for start, end in zip(parts[:-1], parts[1:]):
        ring = points[start:end]
        if len(ring) >= 3:
            patches.append(MplPolygon(ring, closed=True))
    return patches


def load_state_patch(name: str) -> tuple[list[MplPolygon], tuple[float, float, float, float]]:
    reader = shapefile.Reader(str(STATES_SHP))
    for shape_record in reader.iterShapeRecords():
        record = shape_record.record.as_dict()
        if record["NAME"] == name:
            shape = shape_record.shape
            return shape_to_patches(shape), tuple(shape.bbox)
    raise ValueError(f"State not found: {name}")


def load_plot_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_parquet(DATASET_PATH).reset_index().rename(columns={"index": "row_index"})
    keep_cols = ["row_index", "latitude", "longitude", "sample_type", "Y_label", "state", "sample_id"]
    dataset = dataset[keep_cols]

    pred = pd.read_csv(PREDICTIONS_PATH)
    pred = pred[
        pred["cv"].eq(CV)
        & pred["fold"].eq(FOLD)
        & pred["model"].isin([LEFT_MODEL, RIGHT_MODEL])
    ].copy()
    pred = pred.merge(dataset, on=["row_index", "sample_id", "Y_label", "state"], how="left")
    pred = pred.dropna(subset=["latitude", "longitude"])

    metrics = pd.read_csv(METRICS_PATH)
    metrics = metrics[
        metrics["cv"].eq(CV)
        & metrics["fold"].eq(FOLD)
        & metrics["model"].isin([LEFT_MODEL, RIGHT_MODEL])
    ].copy()
    return pred, metrics


def mark_topk(df: pd.DataFrame, frac: float = 0.05) -> pd.DataFrame:
    out = df.copy()
    k = max(1, int(round(len(out) * frac)))
    top_idx = out.sort_values("y_prob", ascending=False).head(k).index
    out["is_top05"] = False
    out.loc[top_idx, "is_top05"] = True
    out["top05_k"] = k
    return out


def add_state_background(ax, bbox: tuple[float, float, float, float], patches: list[MplPolygon]) -> None:
    collection = PatchCollection(
        patches,
        facecolor="#F7F8FA",
        edgecolor="#8E99A8",
        linewidth=0.70,
        zorder=1,
    )
    ax.add_collection(collection)
    xmin, ymin, xmax, ymax = bbox
    pad_x = (xmax - xmin) * 0.07
    pad_y = (ymax - ymin) * 0.07
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#C8D0DA")
        spine.set_linewidth(0.70)


def draw_sample_points(ax, sub: pd.DataFrame, top_color: str) -> None:
    positive = sub[sub["Y_label"].astype(int).eq(1)]
    negative = sub[sub["Y_label"].astype(int).eq(0)]

    ax.scatter(
        negative["longitude"],
        negative["latitude"],
        s=8,
        marker="^",
        facecolor="#B8BEC8",
        edgecolor="#677083",
        linewidth=0.25,
        alpha=0.52,
        zorder=3,
    )
    ax.scatter(
        positive["longitude"],
        positive["latitude"],
        s=12,
        marker="o",
        facecolor="#D71920",
        edgecolor="white",
        linewidth=0.22,
        alpha=0.92,
        zorder=4,
    )

    top = sub[sub["is_top05"]]
    ax.scatter(
        top["longitude"],
        top["latitude"],
        s=34,
        marker="o",
        facecolor="none",
        edgecolor=top_color,
        linewidth=1.05,
        alpha=0.98,
        zorder=6,
    )


def metric_box_text(sub: pd.DataFrame, metrics: pd.DataFrame, model: str) -> str:
    m = metrics[metrics["model"].eq(model)].iloc[0]
    top = sub[sub["is_top05"]]
    positives_top = int(top["Y_label"].astype(int).sum())
    negatives_top = int((top["Y_label"].astype(int) == 0).sum())
    return "\n".join(
        [
            f"Validation state: {STATE_NAME}",
            f"Samples: n={int(m['n_test'])}",
            f"Top-5% samples: n={int(sub['top05_k'].iloc[0])}",
            f"Positives in Top-5%: {positives_top}",
            f"Hard negatives in Top-5%: {negatives_top}",
            f"Precision@5%: {float(m['top05_precision']):.2f}",
            f"Recall@5%: {float(m['top05_recall']):.2f}",
            f"AP: {float(m['average_precision']):.2f}",
        ]
    )


def add_metric_box(ax, text: str) -> None:
    ax.text(
        0.035,
        0.035,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.3,
        color=COLORS["text"],
        linespacing=1.15,
        bbox=dict(
            boxstyle="round,pad=0.28,rounding_size=0.04",
            facecolor="white",
            edgecolor="#AEB7C2",
            linewidth=0.55,
            alpha=0.94,
        ),
        zorder=8,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patches, bbox = load_state_patch(STATE_NAME)
    pred, metrics = load_plot_data()

    left = mark_topk(pred[pred["model"].eq(LEFT_MODEL)].copy())
    right = mark_topk(pred[pred["model"].eq(RIGHT_MODEL)].copy())

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.10), dpi=180)
    fig.patch.set_facecolor("white")

    for ax, sub, model, color, title in [
        (axes[0], left, LEFT_MODEL, COLORS["orange"], "(a) M2: no-climate RF"),
        (axes[1], right, RIGHT_MODEL, COLORS["blue"], "(b) M4: climate-adjusted RF"),
    ]:
        add_state_background(ax, bbox, patches)
        draw_sample_points(ax, sub, color)
        add_metric_box(ax, metric_box_text(sub, metrics, model))
        ax.set_title(title, fontsize=7.6, fontweight="bold", color=COLORS["text"], pad=4)

    fig.suptitle("Spatial comparison of Top-5% ranked evaluation samples", fontsize=9.0, fontweight="bold", color=COLORS["text"], y=0.985)
    fig.text(0.50, 0.925, "Example state-grouped validation fold: Arizona", ha="center", va="center", fontsize=6.4, color=COLORS["muted"])

    axes[0].annotate(
        "",
        xy=(1.050, 0.52),
        xytext=(1.000, 0.52),
        xycoords=axes[0].transAxes,
        textcoords=axes[0].transAxes,
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color=COLORS["text"], mutation_scale=10),
        annotation_clip=False,
    )
    axes[0].text(
        1.026,
        0.58,
        "climate\nadjustment",
        transform=axes[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=5.2,
        color=COLORS["text"],
        linespacing=0.95,
        clip_on=False,
    )

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D71920", markeredgecolor="white", markersize=5.5, label="Positive sample"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#B8BEC8", markeredgecolor="#677083", markersize=5.8, label="Hard negative sample"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=COLORS["orange"], markeredgewidth=1.2, markersize=7.0, label="Top-5% ranked (M2)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=COLORS["blue"], markeredgewidth=1.2, markersize=7.0, label="Top-5% ranked (M4)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.045),
        ncol=4,
        frameon=True,
        fontsize=5.9,
        edgecolor="#AEB7C2",
        framealpha=0.95,
    )
    fig.text(
        0.50,
        0.014,
        "Top-5% denotes the highest-ranked 5% of evaluation samples, not a continuous target-area boundary.",
        ha="center",
        va="bottom",
        fontsize=5.7,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.885, bottom=0.145, wspace=0.12)

    png_path, pdf_path = save_figure(fig, "fig08_spatial_topk_comparison", OUT_DIR)
    plt.close(fig)

    out_csv = OUT_DIR / "fig08_spatial_topk_comparison_data.csv"
    pd.concat([left, right], ignore_index=True).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
