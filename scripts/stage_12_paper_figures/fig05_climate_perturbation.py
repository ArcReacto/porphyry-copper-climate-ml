from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cdmpm_figure_style import COLORS, SERIES, clean_axes, save_figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "figures"
INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "paper_optimization"
    / "final_main_experiment"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
    / "table_climate_perturbation_overview.csv"
)
DETAIL_INPUT = INPUT.with_name("table_climate_perturbation_detail.csv")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overview = pd.read_csv(INPUT)
    detail = pd.read_csv(DETAIL_INPUT)
    overview = overview[overview["cv"].eq("groupkfold_state")].copy()
    detail = detail[detail["cv"].eq("groupkfold_state")].copy()

    keep = ["Full_RF", "NoClimate_RF", "M4_Spearman_RF", "M4_GraphUnion_RF"]
    labels = {
        "Full_RF": "M1\nFull",
        "NoClimate_RF": "M2\nNo climate",
        "M4_Spearman_RF": "M4\nSpearman",
        "M4_GraphUnion_RF": "M4\nGraphUnion",
    }
    scenario_labels = {
        "climate_minus": "Climate -",
        "climate_plus": "Climate +",
        "climate_train_median": "Train median",
    }

    overview = overview[overview["model"].isin(keep)].set_index("model").loc[keep].reset_index()
    detail = detail[detail["model"].isin(keep)].copy()

    heat = (
        detail.pivot(index="model", columns="scenario", values="mean_abs_delta_mean")
        .loc[keep, list(scenario_labels)]
        .to_numpy()
    )
    worst_topk = (
        detail.groupby("model")[["top05_f1_mean", "top10_f1_mean"]]
        .min()
        .loc[keep]
        .reset_index()
    )
    original_topk = (
        detail.groupby("model")[["original_top05_f1", "original_top10_f1"]]
        .first()
        .loc[keep]
        .reset_index()
    )

    fig = plt.figure(figsize=(3.45, 4.35), dpi=180)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.02, 1.28], hspace=0.50)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    vmax = max(0.032, float(np.nanmax(heat)))
    im = ax_h.imshow(heat, cmap="YlGnBu", vmin=0, vmax=vmax, aspect="auto")
    ax_h.set_title("(a) Score response to climate perturbations", loc="left", fontsize=7.6, color=COLORS["text"], pad=5)
    ax_h.set_xticks(np.arange(len(scenario_labels)))
    ax_h.set_xticklabels(list(scenario_labels.values()), fontsize=5.9)
    ax_h.set_yticks(np.arange(len(keep)))
    ax_h.set_yticklabels([labels[m].replace("\n", " ") for m in keep], fontsize=5.9)
    ax_h.tick_params(length=0, colors=COLORS["muted"])
    for spine in ax_h.spines.values():
        spine.set_visible(False)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat[i, j]
            color = "white" if val > vmax * 0.55 else COLORS["text"]
            ax_h.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=5.6, color=color)
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.035, pad=0.025)
    cbar.ax.tick_params(labelsize=5.4, length=2, colors=COLORS["muted"])
    cbar.outline.set_visible(False)
    cbar.set_label("Mean |score change|", fontsize=5.7, color=COLORS["muted"], labelpad=3)

    x = np.arange(len(keep))
    width = 0.23
    top5 = worst_topk["top05_f1_mean"].to_numpy()
    top10 = worst_topk["top10_f1_mean"].to_numpy()
    orig5 = original_topk["original_top05_f1"].to_numpy()
    orig10 = original_topk["original_top10_f1"].to_numpy()

    ax_b.bar(x - width / 2, top5, width, color=SERIES[0], alpha=0.92, label="Worst Top-5% F1")
    ax_b.bar(x + width / 2, top10, width, color=SERIES[1], alpha=0.88, label="Worst Top-10% F1")
    ax_b.scatter(x - width / 2, orig5, s=12, facecolor="white", edgecolor=SERIES[0], linewidth=0.8, zorder=4, label="Original")
    ax_b.scatter(x + width / 2, orig10, s=12, facecolor="white", edgecolor=SERIES[1], linewidth=0.8, zorder=4)

    baseline_top5 = float(worst_topk.loc[worst_topk["model"].eq("NoClimate_RF"), "top05_f1_mean"].iloc[0])
    baseline_top10 = float(worst_topk.loc[worst_topk["model"].eq("NoClimate_RF"), "top10_f1_mean"].iloc[0])
    ax_b.axhline(baseline_top5, color=SERIES[0], lw=0.75, linestyle=":", alpha=0.75)
    ax_b.axhline(baseline_top10, color=SERIES[1], lw=0.75, linestyle=":", alpha=0.75)

    for i, (v5, v10) in enumerate(zip(top5, top10)):
        ax_b.text(i - width / 2, v5 + 0.010, f"{v5:.3f}", ha="center", va="bottom", fontsize=5.3, color=COLORS["text"])
        ax_b.text(i + width / 2, v10 + 0.010, f"{v10:.3f}", ha="center", va="bottom", fontsize=5.3, color=COLORS["text"])

    ax_b.set_title("(b) Worst-case Top-K ranking after perturbation", loc="left", fontsize=7.6, color=COLORS["text"], pad=5)
    ax_b.set_ylabel("F1 under perturbation", fontsize=6.2, color=COLORS["muted"])
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([labels[m] for m in keep], fontsize=5.8)
    ax_b.set_ylim(0.50, 0.82)
    ax_b.grid(axis="y")
    ax_b.set_axisbelow(True)
    clean_axes(ax_b)
    ax_b.legend(loc="upper left", frameon=False, fontsize=5.5, ncol=1, bbox_to_anchor=(0.01, 0.99))
    ax_b.text(
        0.99,
        -0.30,
        "Dots show original scores; bars show the minimum over climate-minus, climate-plus, and train-median scenarios.",
        transform=ax_b.transAxes,
        ha="right",
        va="center",
        fontsize=5.1,
        color=COLORS["muted"],
    )

    fig.subplots_adjust(left=0.24, right=0.92, top=0.95, bottom=0.13)
    png_path, pdf_path = save_figure(fig, "fig05_climate_perturbation", OUT_DIR)
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
