from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import PatchCollection

from cdmpm_figure_style import COLORS, save_figure
from fig08_spatial_topk_comparison import (
    CV,
    FOLD,
    LEFT_MODEL,
    OUT_DIR,
    RIGHT_MODEL,
    STATE_NAME,
    draw_sample_points,
    load_plot_data,
    load_state_patch,
    mark_topk,
)


def add_state_background_notext(ax, bbox, patches) -> None:
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


def add_empty_metric_box(ax) -> None:
    x0, y0, w, h = 0.035, 0.035, 0.36, 0.22
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            w,
            h,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="#AEB7C2",
            linewidth=0.55,
            alpha=0.94,
            zorder=8,
        )
    )
    for i in range(6):
        y = y0 + h - 0.035 - i * 0.030
        ax.plot(
            [x0 + 0.025, x0 + w * (0.88 if i < 3 else 0.68)],
            [y, y],
            transform=ax.transAxes,
            color="#D9E0EA",
            lw=2.0,
            solid_capstyle="round",
            zorder=9,
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patches, bbox = load_state_patch(STATE_NAME)
    pred, _metrics = load_plot_data()

    left = mark_topk(pred[pred["model"].eq(LEFT_MODEL)].copy())
    right = mark_topk(pred[pred["model"].eq(RIGHT_MODEL)].copy())

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.82), dpi=180)
    fig.patch.set_facecolor("white")

    for ax, sub, color in [
        (axes[0], left, COLORS["orange"]),
        (axes[1], right, COLORS["blue"]),
    ]:
        add_state_background_notext(ax, bbox, patches)
        draw_sample_points(ax, sub, color)
        add_empty_metric_box(ax)

    axes[0].annotate(
        "",
        xy=(1.050, 0.52),
        xytext=(1.000, 0.52),
        xycoords=axes[0].transAxes,
        textcoords=axes[0].transAxes,
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color=COLORS["text"], mutation_scale=10),
        annotation_clip=False,
    )

    fig.subplots_adjust(left=0.035, right=0.985, top=0.965, bottom=0.045, wspace=0.12)
    png_path, pdf_path = save_figure(fig, "fig08_spatial_topk_comparison_notext", OUT_DIR)
    plt.close(fig)

    pd.concat([left, right], ignore_index=True).to_csv(
        OUT_DIR / "fig08_spatial_topk_comparison_notext_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
