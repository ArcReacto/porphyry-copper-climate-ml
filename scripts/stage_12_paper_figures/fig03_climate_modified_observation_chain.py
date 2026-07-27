from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import patches

from cdmpm_figure_style import COLORS, arrow, rounded_box, save_figure


def tag(ax, x: float, y: float, text: str) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            0.098,
            0.040,
            boxstyle="round,pad=0.010,rounding_size=0.016",
            facecolor="white",
            edgecolor=COLORS["purple"],
            linewidth=0.65,
            zorder=4,
        )
    )
    ax.text(x + 0.049, y + 0.020, text, ha="center", va="center", fontsize=5.3, color=COLORS["muted"], zorder=5)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.85), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.925, "Climate-modified surface observation chain", ha="center", va="center", fontsize=9.0, fontweight="bold", color=COLORS["text"])

    y = 0.405
    w = 0.160
    h = 0.220
    stages = [
        (0.045, "Mineralization", "subsurface ore\nsystem", COLORS["blue_light"], COLORS["blue"]),
        (0.285, "Surface\nindicators", "alteration, elements,\nstructure", COLORS["green_light"], COLORS["green"]),
        (0.525, "Observed\nfeatures", "measured strength\nand visibility", COLORS["orange_light"], COLORS["orange"]),
        (0.765, "Prospectivity\nscore", "budget-aware\ntarget ranking", COLORS["surface"], COLORS["line"]),
    ]
    for x, title, subtitle, face, edge in stages:
        rounded_box(ax, (x, y), w, h, title, subtitle, face, edge, title_size=7.0, subtitle_size=5.7)

    arrow(ax, (0.045 + w, y + h / 2), (0.285, y + h / 2), COLORS["blue"])
    arrow(ax, (0.285 + w, y + h / 2), (0.525, y + h / 2), COLORS["orange"])
    arrow(ax, (0.525 + w, y + h / 2), (0.765, y + h / 2), COLORS["muted"])

    rounded_box(
        ax,
        (0.345, 0.710),
        0.285,
        0.135,
        "Long-term climate",
        "temperature, precipitation, radiation,\nsoil moisture, snow",
        COLORS["purple_light"],
        COLORS["purple"],
        title_size=6.9,
        subtitle_size=5.3,
    )
    arrow(ax, (0.488, 0.710), (0.488, 0.625), COLORS["purple"])
    ax.text(0.555, 0.650, "modifies expression", ha="left", va="center", fontsize=5.8, color=COLORS["muted"])

    ax.add_patch(
        patches.FancyBboxPatch(
            (0.310, 0.175),
            0.360,
            0.095,
            boxstyle="round,pad=0.014,rounding_size=0.024",
            facecolor="white",
            edgecolor="#D6DEE8",
            linewidth=0.8,
        )
    )
    ax.text(
        0.490,
        0.222,
        "Climate changes how mineralization-related signals are observed; it is not treated as a direct label cause.",
        ha="center",
        va="center",
        fontsize=5.8,
        color=COLORS["muted"],
    )

    for x, y0, text in [(0.325, 0.306, "weathering"), (0.435, 0.306, "leaching"), (0.545, 0.306, "erosion"), (0.435, 0.125, "exposure")]:
        tag(ax, x, y0, text)

    fig.tight_layout(pad=0.16)
    png_path, pdf_path = save_figure(fig, "fig03_climate_modified_observation_chain")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
