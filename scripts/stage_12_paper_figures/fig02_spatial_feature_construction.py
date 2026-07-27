from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import patches

from cdmpm_figure_style import COLORS, arrow, rounded_box, save_figure


def layer_icon(ax, x: float, y: float, kind: str, color: str) -> None:
    if kind == "raster":
        ax.add_patch(patches.Rectangle((x, y), 0.070, 0.050, facecolor="white", edgecolor=color, lw=0.8, zorder=4))
        for i in range(1, 4):
            ax.plot([x + i * 0.0175, x + i * 0.0175], [y, y + 0.050], color=color, lw=0.45, alpha=0.55, zorder=5)
            ax.plot([x, x + 0.070], [y + i * 0.0125, y + i * 0.0125], color=color, lw=0.45, alpha=0.55, zorder=5)
    elif kind == "points":
        xs = [x + 0.005, x + 0.020, x + 0.035, x + 0.052, x + 0.064]
        ys = [y + 0.015, y + 0.043, y + 0.024, y + 0.052, y + 0.012]
        ax.scatter(xs, ys, s=9, color=color, zorder=5)
        ax.add_patch(patches.Circle((x + 0.035, y + 0.030), 0.036, fill=False, edgecolor=color, lw=0.7, linestyle="--", zorder=4))
    elif kind == "lines":
        ax.plot([x + 0.005, x + 0.028, x + 0.045, x + 0.068], [y + 0.010, y + 0.052, y + 0.024, y + 0.058], color=color, lw=1.2, zorder=5)
        ax.plot([x + 0.012, x + 0.065], [y + 0.057, y + 0.006], color=color, lw=0.8, zorder=5)
    else:
        ax.add_patch(
            patches.Polygon(
                [(x + 0.004, y + 0.014), (x + 0.028, y + 0.056), (x + 0.061, y + 0.046), (x + 0.069, y + 0.016), (x + 0.035, y)],
                closed=True,
                facecolor=COLORS["green_light"],
                edgecolor=color,
                lw=0.8,
                zorder=5,
            )
        )


def feature_matrix(ax, x: float, y: float, w: float, h: float) -> None:
    rounded_box(ax, (x, y), w, h, "Structured feature vector", None, COLORS["surface"], COLORS["line"], title_size=7.2)
    rows = [
        ("nearest raster", COLORS["blue_light"]),
        ("neighborhood statistics", "#EAF3FF"),
        ("distance / density", COLORS["orange_light"]),
        ("polygon attributes", COLORS["green_light"]),
        ("climate normals", COLORS["purple_light"]),
    ]
    row_h = (h - 0.085) / len(rows)
    for i, (label, face) in enumerate(rows):
        yy = y + h - 0.075 - (i + 1) * row_h
        ax.add_patch(patches.Rectangle((x + 0.030, yy + 0.010), w - 0.060, row_h - 0.014, facecolor=face, edgecolor="white", lw=0.5, zorder=4))
        ax.text(x + 0.050, yy + row_h / 2, label, ha="left", va="center", fontsize=5.9, color=COLORS["text"], zorder=5)
        ax.text(x + w - 0.045, yy + row_h / 2, "...", ha="right", va="center", fontsize=6.3, color=COLORS["muted"], zorder=5)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.55), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "Point-based spatial feature construction", ha="center", va="center", fontsize=9.2, fontweight="bold", color=COLORS["text"])

    layers = [
        ("Raster grids", "gravity, DEM,\nclimate cells", "raster", COLORS["blue"], COLORS["blue_light"], 0.70),
        ("Point surveys", "geochemical\nsamples", "points", "#2B6CB0", "#EAF3FF", 0.50),
        ("Vector lines", "faults and\nstructures", "lines", COLORS["orange"], COLORS["orange_light"], 0.30),
        ("Geology polygons", "lithology and\nmap units", "poly", COLORS["green"], COLORS["green_light"], 0.10),
    ]
    for title, subtitle, kind, edge, face, y in layers:
        rounded_box(ax, (0.040, y), 0.255, 0.150, title, subtitle, face, edge, title_size=7.0, subtitle_size=5.7)
        layer_icon(ax, 0.065, y + 0.048, kind, edge)

    # Sample point and extraction rules.
    ax.add_patch(patches.Circle((0.470, 0.500), 0.165, facecolor="#F8FAFC", edgecolor="#D6DEE8", lw=0.9, zorder=1))
    ax.add_patch(patches.Circle((0.470, 0.500), 0.078, facecolor="white", edgecolor=COLORS["blue"], lw=1.0, zorder=2))
    ax.scatter([0.470], [0.500], marker="*", s=160, color=COLORS["orange"], edgecolor="white", linewidth=0.6, zorder=6)
    ax.text(0.470, 0.387, "sample point", ha="center", va="center", fontsize=6.8, fontweight="bold", color=COLORS["text"])
    ax.text(0.470, 0.332, "nearest value\nand radius statistics", ha="center", va="center", fontsize=5.8, color=COLORS["muted"], linespacing=1.12)

    for y in [0.775, 0.575, 0.375, 0.175]:
        arrow(ax, (0.295, y), (0.385, 0.520), COLORS["muted"], rad=0.08)

    feature_matrix(ax, 0.675, 0.190, 0.270, 0.610)
    arrow(ax, (0.548, 0.500), (0.675, 0.500), COLORS["muted"])

    ax.text(
        0.500,
        0.095,
        "All sources are aligned to the same sample coordinates before modeling; aggregation depends on source geometry.",
        ha="center",
        va="center",
        fontsize=5.9,
        color=COLORS["muted"],
    )

    fig.tight_layout(pad=0.16)
    png_path, pdf_path = save_figure(fig, "fig02_spatial_feature_construction")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
