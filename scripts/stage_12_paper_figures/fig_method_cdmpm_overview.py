from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import patches

from cdmpm_figure_style import COLORS, arrow, rounded_box, save_figure


def small_chip(ax, x: float, y: float, text: str, color: str) -> None:
    chip = patches.FancyBboxPatch(
        (x, y),
        0.070,
        0.034,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        facecolor="white",
        edgecolor=color,
        linewidth=0.65,
        zorder=5,
    )
    ax.add_patch(chip)
    ax.text(x + 0.035, y + 0.017, text, ha="center", va="center", fontsize=4.8, color=COLORS["muted"], zorder=6)


def graph_icon(ax, x: float, y: float, w: float, h: float) -> None:
    left = [(x + 0.03, y + h * 0.70), (x + 0.03, y + h * 0.38)]
    right = [(x + w - 0.03, y + h * 0.76), (x + w - 0.03, y + h * 0.55), (x + w - 0.03, y + h * 0.34)]
    for sx, sy in left:
        for tx, ty in right:
            ax.plot([sx, tx], [sy, ty], color=COLORS["purple"], alpha=0.25, lw=0.8, zorder=3)
    for sx, sy in left:
        ax.scatter([sx], [sy], s=18, color=COLORS["purple"], edgecolor="white", linewidth=0.5, zorder=6)
    for tx, ty in right:
        ax.scatter([tx], [ty], s=18, color=COLORS["orange"], edgecolor="white", linewidth=0.5, zorder=6)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.45), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "CD-MPM: climate-decoupled target ranking", ha="center", va="center", fontsize=9.2, fontweight="bold", color=COLORS["text"])

    ax.add_patch(
        patches.FancyBboxPatch(
            (0.035, 0.57),
            0.175,
            0.235,
            boxstyle="round,pad=0.014,rounding_size=0.026",
            linewidth=0.9,
            edgecolor=COLORS["blue"],
            facecolor=COLORS["blue_light"],
            zorder=2,
        )
    )
    ax.text(0.1225, 0.715, "Aligned evidence", ha="center", va="center", fontsize=7.4, fontweight="bold", color=COLORS["text"], zorder=5)
    for i, (txt, col) in enumerate(
        [
            ("chem", COLORS["orange"]),
            ("phys", COLORS["blue"]),
            ("geo", COLORS["green"]),
            ("clim", COLORS["purple"]),
        ]
    ):
        small_chip(ax, 0.055 + (i % 2) * 0.083, 0.603 + (1 - i // 2) * 0.041, txt, col)

    rounded_box(
        ax,
        (0.255, 0.57),
        0.175,
        0.235,
        "Feature roles",
        "separate surface\nobservations from\nstable background",
        COLORS["surface"],
        COLORS["line"],
    )

    rounded_box(
        ax,
        (0.485, 0.57),
        0.170,
        0.235,
        "Sensitivity graph",
        None,
        COLORS["purple_light"],
        COLORS["purple"],
    )
    graph_icon(ax, 0.505, 0.590, 0.130, 0.085)
    ax.text(
        0.570,
        0.592,
        "stable associations\nacross fold evidence",
        ha="center",
        va="center",
        fontsize=5.4,
        color=COLORS["muted"],
        linespacing=1.10,
        zorder=6,
    )

    rounded_box(
        ax,
        (0.720, 0.57),
        0.115,
        0.235,
        "Selective\nadjustment",
        r"$x_j^{res}=x_j-\hat g_j(C)$",
        COLORS["orange_light"],
        COLORS["orange"],
        title_size=6.8,
        subtitle_size=5.7,
    )

    rounded_box(
        ax,
        (0.875, 0.57),
        0.095,
        0.235,
        "Target\nranking",
        "Top-K\nprospectivity",
        COLORS["green_light"],
        COLORS["green"],
        title_size=6.8,
    )

    arrow(ax, (0.210, 0.688), (0.255, 0.688), COLORS["muted"])
    arrow(ax, (0.430, 0.688), (0.485, 0.688), COLORS["muted"])
    arrow(ax, (0.655, 0.688), (0.720, 0.688), COLORS["muted"])
    arrow(ax, (0.835, 0.688), (0.875, 0.688), COLORS["muted"])

    # Climate branch and fold boundary.
    rounded_box(
        ax,
        (0.255, 0.205),
        0.175,
        0.145,
        "Climate variables",
        "used as adjustment\nsignals",
        COLORS["purple_light"],
        COLORS["purple"],
        title_size=6.8,
    )
    rounded_box(
        ax,
        (0.485, 0.205),
        0.170,
        0.145,
        "Fold-local fitting",
        "selection and residual\nmodels fit in training",
        COLORS["surface"],
        COLORS["line"],
        title_size=6.8,
    )
    arrow(ax, (0.430, 0.278), (0.485, 0.278), COLORS["purple"])
    arrow(ax, (0.570, 0.350), (0.570, 0.570), COLORS["purple"])
    arrow(ax, (0.655, 0.278), (0.745, 0.570), COLORS["orange"], rad=0.10)

    ax.add_patch(
        patches.FancyBboxPatch(
            (0.225, 0.080),
            0.555,
            0.070,
            boxstyle="round,pad=0.012,rounding_size=0.020",
            facecolor="white",
            edgecolor="#D6DEE8",
            linewidth=0.8,
        )
    )
    ax.text(
        0.502,
        0.115,
        "Climate is not treated as a deposit label; it guides selective correction of climate-associated observations.",
        ha="center",
        va="center",
        fontsize=5.9,
        color=COLORS["muted"],
    )

    fig.tight_layout(pad=0.16)
    png_path, pdf_path = save_figure(fig, "fig_method_cdmpm_overview")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
