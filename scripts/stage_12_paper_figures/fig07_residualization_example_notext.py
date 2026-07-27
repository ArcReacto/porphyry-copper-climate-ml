from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from cdmpm_figure_style import COLORS, save_figure
from fig07_residualization_example import (
    FEATURE,
    OUT_DIR,
    climate_columns,
    load_data,
    load_state_patches,
    out_of_fold_climate_component,
)
from matplotlib.collections import PatchCollection


RANDOM_STATE = 20260622


def add_state_background(ax, patches):
    collection = PatchCollection(
        patches,
        facecolor="#F7F8FA",
        edgecolor="#AEB7C2",
        linewidth=0.38,
        zorder=1,
    )
    ax.add_collection(collection)
    ax.set_xlim(-125.5, -101.0)
    ax.set_ylim(30.5, 49.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#D0D7E2")
        spine.set_linewidth(0.55)


def plot_field_notext(ax, df, value_col: str, cmap: str, diverging: bool = False):
    patches = load_state_patches()
    add_state_background(ax, patches)

    sub = df.dropna(subset=["longitude", "latitude", value_col]).copy()
    x = sub["longitude"].to_numpy()
    y = sub["latitude"].to_numpy()
    z = sub[value_col].to_numpy()
    tri = mtri.Triangulation(x, y)

    if diverging:
        vmax = max(1.0, float(np.nanpercentile(np.abs(z), 94)))
        levels = np.linspace(-vmax, vmax, 13)
        contour = ax.tricontourf(tri, z, levels=levels, cmap=cmap, alpha=0.80, zorder=2, extend="both")
    else:
        lo, hi = np.nanpercentile(z, [4, 96])
        levels = np.linspace(lo, hi, 12)
        contour = ax.tricontourf(tri, z, levels=levels, cmap=cmap, alpha=0.82, zorder=2, extend="both")

    positives = df[df["Y_label"].astype(int).eq(1)]
    ax.scatter(
        positives["longitude"],
        positives["latitude"],
        s=7.5,
        c="#C4161C",
        edgecolors="white",
        linewidths=0.18,
        zorder=5,
    )
    return contour


def add_empty_decomposition_panel(ax):
    ax.axis("off")
    colors = [COLORS["orange"], COLORS["blue"], COLORS["purple"]]
    for i, color in enumerate(colors):
        y = 0.72 - i * 0.24
        ax.add_patch(
            plt.Rectangle(
                (0.02, y - 0.060),
                0.13,
                0.12,
                facecolor=color,
                alpha=0.18,
                edgecolor=color,
                lw=0.75,
            )
        )
        ax.plot([0.20, 0.90], [y, y], color="#D8DEE8", lw=3.4, solid_capstyle="round")
        ax.plot([0.20, 0.70], [y - 0.055, y - 0.055], color="#EEF2F7", lw=2.3, solid_capstyle="round")


def add_empty_table_panel(ax):
    ax.axis("off")
    x0, y0, w, h = 0.04, 0.12, 0.92, 0.76
    n_rows, n_cols = 5, 4
    for r in range(n_rows):
        for c in range(n_cols):
            x = x0 + c * w / n_cols
            y = y0 + (n_rows - 1 - r) * h / n_rows
            face = "#EEF6F7" if r == 0 else "white"
            ax.add_patch(
                plt.Rectangle(
                    (x, y),
                    w / n_cols,
                    h / n_rows,
                    facecolor=face,
                    edgecolor="#D8DEE8",
                    lw=0.45,
                )
            )
            line_color = COLORS["blue"] if c == 3 and r > 0 else "#CBD5E1"
            line_width = 3.0 if c == 3 and r > 0 else 2.0
            ax.plot(
                [x + 0.05 * w / n_cols, x + 0.86 * w / n_cols],
                [y + 0.50 * h / n_rows, y + 0.50 * h / n_rows],
                color=line_color,
                lw=line_width,
                alpha=0.75,
                solid_capstyle="round",
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def add_distribution_panel_notext(ax, df):
    data = [
        (df["x_obs"] - df["x_obs"].median()) / (df["x_obs"].std() or 1.0),
        (df["x_hat_climate"] - df["x_hat_climate"].median()) / (df["x_hat_climate"].std() or 1.0),
        df["residual_robust_z"],
    ]
    parts = ax.violinplot(data, positions=[1, 2, 3], widths=0.70, showmeans=False, showmedians=True, showextrema=False)
    colors = [COLORS["orange"], COLORS["blue"], COLORS["purple"]]
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
    parts["cmedians"].set_color(COLORS["text"])
    parts["cmedians"].set_linewidth(0.8)
    for i, values in enumerate(data, start=1):
        rng = np.random.default_rng(RANDOM_STATE + i)
        sample = np.asarray(values.dropna())
        if len(sample) > 450:
            sample = rng.choice(sample, size=450, replace=False)
        ax.scatter(
            rng.normal(i, 0.035, size=len(sample)),
            np.clip(sample, -2.8, 2.8),
            s=1.2,
            color=colors[i - 1],
            alpha=0.16,
            linewidths=0,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-2.9, 2.9)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.4)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#D0D7E2")
    ax.spines["bottom"].set_color("#D0D7E2")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    df = out_of_fold_climate_component(df, climate_columns())

    fig = plt.figure(figsize=(7.15, 4.72), dpi=180)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 0.82], hspace=0.34, wspace=0.12)

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    contours = [
        plot_field_notext(axes[0], df, "x_obs", "YlOrBr", diverging=False),
        plot_field_notext(axes[1], df, "x_hat_climate", "YlGnBu", diverging=False),
        plot_field_notext(axes[2], df, "residual_robust_z", "RdBu_r", diverging=True),
    ]
    for ax, contour in zip(axes, contours):
        cb = fig.colorbar(contour, ax=ax, fraction=0.045, pad=0.010)
        cb.ax.set_yticklabels([])
        cb.ax.tick_params(length=0)
        cb.outline.set_visible(False)

    add_empty_decomposition_panel(fig.add_subplot(gs[1, 0]))
    add_empty_table_panel(fig.add_subplot(gs[1, 1]))
    add_distribution_panel_notext(fig.add_subplot(gs[1, 2]), df)

    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.055, top=0.965)
    png_path, pdf_path = save_figure(fig, "fig07_residualization_example_notext", OUT_DIR)
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
