from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shapefile
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path.home() / "Desktop" / "\u63a2\u77ff\u6c14\u8c61\u6570\u636e\u96c6"

SAMPLES_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_datasets"
    / "by_sample_scheme"
    / "known_mining_neutral"
    / "known_mining_neutral_ratio_1_10"
    / "model_dataset_known_mining_neutral_ratio_1_10_with_neutral_all_features_v1.parquet"
)
STATES_SHP = (
    DATA_ROOT
    / "cb_2025_us_state_500k"
    / "cb_2025_us_state_500k.shp"
)
OUT_DIR = PROJECT_ROOT / "figures"

WESTERN_STATES = {
    "Arizona",
    "Washington",
    "Nevada",
    "New Mexico",
    "Oregon",
    "Utah",
    "Wyoming",
    "Colorado",
    "Idaho",
    "Montana",
    "California",
}

STATE_LABELS = {
    "Arizona": "AZ",
    "Washington": "WA",
    "Nevada": "NV",
    "New Mexico": "NM",
    "Oregon": "OR",
    "Utah": "UT",
    "Wyoming": "WY",
    "Colorado": "CO",
    "Idaho": "ID",
    "Montana": "MT",
    "California": "CA",
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


def load_state_shapes() -> tuple[list[MplPolygon], dict[str, tuple[float, float]]]:
    reader = shapefile.Reader(str(STATES_SHP))
    patches: list[MplPolygon] = []
    centroids: dict[str, tuple[float, float]] = {}

    for shape_record in reader.iterShapeRecords():
        record = shape_record.record.as_dict()
        name = record["NAME"]
        if name not in WESTERN_STATES:
            continue
        shape = shape_record.shape
        patches.extend(shape_to_patches(shape))

        xmin, ymin, xmax, ymax = shape.bbox
        centroids[name] = ((xmin + xmax) / 2, (ymin + ymax) / 2)

    return patches, centroids


def load_samples() -> pd.DataFrame:
    df = pd.read_parquet(SAMPLES_PATH)
    df = df[df["state"].isin(WESTERN_STATES)].copy()
    df = df.dropna(subset=["longitude", "latitude", "sample_type"])
    return df


def plot_points(ax, df: pd.DataFrame) -> None:
    styles = {
        "neutral": {
            "label": "Neutral background",
            "color": "#B8BFC7",
            "edgecolor": "none",
            "s": 8,
            "alpha": 0.28,
            "zorder": 3,
        },
        "negative": {
            "label": "Hard negative",
            "color": "#2B6CB0",
            "edgecolor": "none",
            "s": 10,
            "alpha": 0.34,
            "zorder": 4,
        },
        "positive": {
            "label": "Porphyry Cu positive",
            "color": "#D9482B",
            "edgecolor": "white",
            "s": 28,
            "alpha": 0.95,
            "linewidth": 0.35,
            "zorder": 5,
        },
    }

    for sample_type in ["neutral", "negative", "positive"]:
        sub = df[df["sample_type"].eq(sample_type)]
        if sub.empty:
            continue
        kwargs = styles[sample_type].copy()
        label = kwargs.pop("label")
        ax.scatter(
            sub["longitude"],
            sub["latitude"],
            label=f"{label} (n={len(sub):,})",
            **kwargs,
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_samples()
    state_patches, centroids = load_state_shapes()

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    state_collection = PatchCollection(
        state_patches,
        facecolor="#F4F6F8",
        edgecolor="#A7B0BA",
        linewidth=0.55,
        zorder=1,
    )
    ax.add_collection(state_collection)

    plot_points(ax, samples)

    for state, label in STATE_LABELS.items():
        if state not in centroids:
            continue
        x, y = centroids[state]
        if state == "California":
            x -= 1.2
        elif state == "Washington":
            y -= 0.15
        elif state == "Idaho":
            x += 0.45
        elif state == "New Mexico":
            y += 0.2
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7.8,
            color="#5C6670",
            weight="bold",
            zorder=6,
        )

    ax.set_xlim(-125.5, -101.0)
    ax.set_ylim(30.5, 49.6)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Longitude", fontsize=8.5)
    ax.set_ylabel("Latitude", fontsize=8.5)

    ax.grid(
        True,
        color="#E2E8F0",
        linewidth=0.45,
        linestyle="-",
        zorder=0,
    )
    ax.tick_params(axis="both", labelsize=7.8, colors="#4A5568", length=3)
    for spine in ax.spines.values():
        spine.set_color("#CBD5E0")
        spine.set_linewidth(0.7)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=f"Porphyry Cu positive (n={(samples['sample_type'] == 'positive').sum():,})",
            markerfacecolor="#D9482B",
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=6.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=f"Hard negative (n={(samples['sample_type'] == 'negative').sum():,})",
            markerfacecolor="#2B6CB0",
            markeredgecolor="none",
            alpha=0.62,
            markersize=5.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=f"Neutral background (n={(samples['sample_type'] == 'neutral').sum():,})",
            markerfacecolor="#B8BFC7",
            markeredgecolor="none",
            alpha=0.72,
            markersize=5.5,
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#CBD5E0",
        fontsize=7.4,
        borderpad=0.8,
        labelspacing=0.45,
        handletextpad=0.55,
    )
    legend.set_zorder(10)

    ax.text(
        0.995,
        0.01,
        "State boundaries: U.S. Census Cartographic Boundary 2025",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#718096",
    )

    fig.tight_layout(pad=0.5)

    png_path = OUT_DIR / "fig01_study_area_samples.png"
    pdf_path = OUT_DIR / "fig01_study_area_samples.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    counts = samples["sample_type"].value_counts().to_dict()
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Sample counts: {counts}")


if __name__ == "__main__":
    main()
