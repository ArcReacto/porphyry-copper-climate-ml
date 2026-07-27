from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import patches

from cdmpm_figure_style import COLORS, arrow, rounded_box, save_figure


def state_tiles(ax, x: float, y: float) -> None:
    states = ["AZ", "NV", "UT", "CO", "WA", "OR", "ID", "NM"]
    cols = [COLORS["blue"], COLORS["blue"], COLORS["green"], COLORS["green"], COLORS["orange"], COLORS["orange"], COLORS["purple"], COLORS["purple"]]
    for i, (state, color) in enumerate(zip(states, cols)):
        xx = x + (i % 4) * 0.036
        yy = y + (1 - i // 4) * 0.037
        ax.add_patch(
            patches.FancyBboxPatch(
                (xx, yy),
                0.029,
                0.029,
                boxstyle="round,pad=0.004,rounding_size=0.006",
                facecolor="white",
                edgecolor=color,
                linewidth=0.70,
                zorder=5,
            )
        )
        ax.text(xx + 0.0145, yy + 0.0145, state, ha="center", va="center", fontsize=4.4, color=COLORS["muted"], zorder=6)


def metric_rows(ax, x: float, y: float, w: float, h: float) -> None:
    rounded_box(ax, (x, y), w, h, "Aggregate metrics", None, COLORS["surface"], COLORS["line"], title_size=6.8)
    rows = [("global", "ROC-AUC, AP"), ("Top-K", "Precision, F1, NDCG"), ("fold view", "paired differences")]
    row_h = (h - 0.075) / len(rows)
    for i, (left, right) in enumerate(rows):
        yy = y + h - 0.070 - (i + 1) * row_h
        ax.add_patch(patches.Rectangle((x + 0.020, yy + 0.008), w - 0.040, row_h - 0.012, facecolor="white", edgecolor="#E6EAF0", lw=0.5, zorder=4))
        ax.text(x + 0.038, yy + row_h / 2, left, ha="left", va="center", fontsize=5.2, fontweight="bold", color=COLORS["text"], zorder=5)
        ax.text(x + w - 0.030, yy + row_h / 2, right, ha="right", va="center", fontsize=5.0, color=COLORS["muted"], zorder=5)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.15), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.935, "Leakage-controlled evaluation protocol", ha="center", va="center", fontsize=9.0, fontweight="bold", color=COLORS["text"])

    rounded_box(ax, (0.035, 0.575), 0.165, 0.220, "Supervised\nsamples", "positive and hard\nnegative points", COLORS["blue_light"], COLORS["blue"], title_size=6.8)
    state_tiles(ax, 0.052, 0.598)

    rounded_box(ax, (0.245, 0.575), 0.165, 0.220, "State-grouped\nsplit", "hold out states\nas validation", COLORS["green_light"], COLORS["green"], title_size=6.8)
    for i, c in enumerate([COLORS["blue"], COLORS["green"], COLORS["orange"]]):
        ax.add_patch(patches.Arc((0.327, 0.653), 0.100 + i * 0.022, 0.070 + i * 0.018, theta1=195, theta2=520, color=c, lw=0.85, zorder=5))

    arrow(ax, (0.200, 0.685), (0.245, 0.685), COLORS["muted"])

    ax.add_patch(
        patches.FancyBboxPatch(
            (0.455, 0.345),
            0.505,
            0.475,
            boxstyle="round,pad=0.014,rounding_size=0.030",
            facecolor="white",
            edgecolor="#D6DEE8",
            linewidth=0.8,
            zorder=1,
        )
    )
    ax.text(0.708, 0.785, "Operations inside each training fold", ha="center", va="center", fontsize=6.4, color=COLORS["muted"], zorder=3)

    rounded_box(ax, (0.480, 0.565), 0.125, 0.150, "Select", "climate-sensitive\nfeatures", COLORS["purple_light"], COLORS["purple"], title_size=6.6, subtitle_size=5.2)
    rounded_box(ax, (0.635, 0.565), 0.125, 0.150, "Residualize", "fit correction on\ntraining only", COLORS["orange_light"], COLORS["orange"], title_size=6.6, subtitle_size=5.2)
    rounded_box(ax, (0.790, 0.565), 0.120, 0.150, "Train", "classifier and\nranking scores", COLORS["blue_light"], COLORS["blue"], title_size=6.6, subtitle_size=5.2)
    arrow(ax, (0.410, 0.685), (0.480, 0.640), COLORS["muted"])
    arrow(ax, (0.605, 0.640), (0.635, 0.640), COLORS["muted"])
    arrow(ax, (0.760, 0.640), (0.790, 0.640), COLORS["muted"])

    rounded_box(ax, (0.480, 0.200), 0.125, 0.115, "Validate", "held-out\nstates", COLORS["green_light"], COLORS["green"], title_size=6.6, subtitle_size=5.1)
    rounded_box(ax, (0.635, 0.200), 0.125, 0.115, "Apply", "fold rules\nonly", COLORS["orange_light"], COLORS["orange"], title_size=6.6, subtitle_size=5.1)
    metric_rows(ax, 0.795, 0.160, 0.160, 0.195)
    arrow(ax, (0.410, 0.640), (0.480, 0.258), COLORS["line"], rad=-0.12)
    arrow(ax, (0.605, 0.258), (0.635, 0.258), COLORS["muted"])
    arrow(ax, (0.760, 0.258), (0.795, 0.258), COLORS["muted"])
    arrow(ax, (0.850, 0.565), (0.875, 0.355), COLORS["muted"])

    ax.text(0.500, 0.075, "Climate-sensitive selection and residualization are never fit on validation states.", ha="center", va="center", fontsize=5.9, color=COLORS["muted"])

    fig.tight_layout(pad=0.16)
    png_path, pdf_path = save_figure(fig, "fig04_experimental_protocol")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
