from __future__ import annotations

from pathlib import Path

import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cdmpm_figure_style import COLORS, clean_axes, save_figure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "figures"
PANEL_DIR = OUT_DIR / "fig07_split_panels"


def load_fig07_module():
    path = Path(__file__).with_name("fig07_residualization_example.py")
    spec = importlib.util.spec_from_file_location("fig07_residualization_example", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_compact_colorbar(fig: plt.Figure, ax: plt.Axes, contour, label: str) -> None:
    cb = fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.010)
    cb.set_label(label, fontsize=6.2, color=COLORS["muted"], labelpad=3)
    cb.ax.tick_params(labelsize=5.6, length=2, colors=COLORS["muted"])
    cb.outline.set_visible(False)


def save_map_panel(fig07, df: pd.DataFrame, value_col: str, title: str, cmap: str, stem: str, diverging: bool) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.75), dpi=220)
    contour = fig07.plot_field(ax, df, value_col, title, cmap, diverging=diverging)
    if value_col == "residual_robust_z":
        cbar_label = "residual z-score"
    else:
        cbar_label = "log-scaled value"
    add_compact_colorbar(fig, ax, contour, cbar_label)
    fig.text(
        0.02,
        0.02,
        "Red points: known porphyry copper positives",
        ha="left",
        va="bottom",
        fontsize=5.7,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.02, right=0.93, bottom=0.08, top=0.90)
    save_figure(fig, stem, PANEL_DIR)
    plt.close(fig)


def save_decomposition_panel(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(2.65, 2.65), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.50,
        0.93,
        "Decomposition",
        ha="center",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["text"],
    )
    rows = [
        (
            r"$X_j$",
            COLORS["orange"],
            "Observed feature",
        ),
        (
            r"$\hat{g}_j(C)$",
            COLORS["blue"],
            "Climate-predicted component",
        ),
        (
            r"$r_j$",
            COLORS["purple"],
            "Residual feature",
        ),
    ]
    y_positions = [0.70, 0.48, 0.26]
    for (symbol, color, name), y in zip(rows, y_positions):
        ax.add_patch(
            plt.Rectangle(
                (0.09, y - 0.055),
                0.23,
                0.11,
                facecolor=color,
                alpha=0.16,
                edgecolor=color,
                lw=0.8,
            )
        )
        ax.text(0.205, y, symbol, ha="center", va="center", fontsize=9.0, color=color, fontweight="bold")
        ax.text(0.40, y, name, ha="left", va="center", fontsize=6.9, fontweight="bold", color=COLORS["text"])
    save_figure(fig, "fig07_panel_d_decomposition", PANEL_DIR)
    plt.close(fig)


def save_correlation_panel(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(2.65, 2.65), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.50,
        0.94,
        "Spearman Association",
        ha="center",
        va="top",
        fontsize=7.6,
        fontweight="bold",
        color=COLORS["text"],
    )
    table = summary.copy().sort_values("drop", ascending=False).head(4)
    headers = ["Climate", r"$|\rho_X|$", r"$|\rho_r|$", "Drop"]
    col_x = [0.045, 0.44, 0.625, 0.81]
    col_w = [0.36, 0.15, 0.15, 0.14]
    row_h = 0.115
    header_y = 0.75

    for x, w, header in zip(col_x, col_w, headers):
        ax.add_patch(plt.Rectangle((x, header_y), w, row_h, facecolor=COLORS["blue_light"], edgecolor="#D8DEE8", lw=0.55))
        ax.text(x + w / 2, header_y + row_h / 2, header, ha="center", va="center", fontsize=4.8, fontweight="bold", color=COLORS["text"])

    variable_map = {
        "Vapor pressure": "Vapor pressure",
        "Solar radiation": "Solar rad.",
        "Temp. range": "Temp. range",
        "Wind speed": "Wind speed",
    }
    for r, (_, row) in enumerate(table.iterrows()):
        y = header_y - (r + 1) * row_h
        fill = "white" if r % 2 == 0 else COLORS["gray_light"]
        vals = [
            variable_map.get(row["climate"], str(row["climate"])),
            f"{abs(row['rho_x']):.2f}",
            f"{abs(row['rho_r']):.2f}",
            f"\u2193 {row['drop']:.2f}",
        ]
        colors = [COLORS["text"], COLORS["orange"], COLORS["purple"], COLORS["blue"]]
        weights = ["normal", "bold", "bold", "bold"]
        for x, w, val, color, weight in zip(col_x, col_w, vals, colors, weights):
            ax.add_patch(plt.Rectangle((x, y), w, row_h, facecolor=fill, edgecolor="#D8DEE8", lw=0.45))
            ha = "left" if x == col_x[0] else "center"
            tx = x + 0.014 if x == col_x[0] else x + w / 2
            ax.text(tx, y + row_h / 2, val, ha=ha, va="center", fontsize=4.75, color=color, fontweight=weight)

    save_figure(fig, "fig07_panel_e_spearman_reduction", PANEL_DIR)
    plt.close(fig)


def save_distribution_panel(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.2), dpi=220)
    data = [
        (df["x_obs"] - df["x_obs"].median()) / (df["x_obs"].std() or 1.0),
        (df["x_hat_climate"] - df["x_hat_climate"].median()) / (df["x_hat_climate"].std() or 1.0),
        df["residual_robust_z"],
    ]
    labels = [r"$X_j$", r"$\hat{g}_j(C)$", r"$r_j$"]
    colors = [COLORS["orange"], COLORS["blue"], COLORS["purple"]]

    parts = ax.violinplot(data, positions=[1, 2, 3], widths=0.65, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.26)
    parts["cmedians"].set_color(COLORS["text"])
    parts["cmedians"].set_linewidth(0.8)
    for i, values in enumerate(data, start=1):
        arr = np.asarray(pd.Series(values).dropna())
        lo, q1, med, q3, hi = np.percentile(arr, [5, 25, 50, 75, 95])
        ax.plot([i, i], [lo, hi], color=colors[i - 1], lw=1.0)
        ax.add_patch(plt.Rectangle((i - 0.10, q1), 0.20, q3 - q1, facecolor="white", edgecolor=colors[i - 1], lw=0.8, zorder=4))
        ax.plot([i - 0.10, i + 0.10], [med, med], color=COLORS["text"], lw=0.8, zorder=5)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=7.0)
    ax.set_ylabel("standardized value", fontsize=6.4, color=COLORS["muted"])
    ax.set_ylim(-2.9, 2.9)
    ax.set_title("Distribution before and after decomposition", loc="left", fontsize=8.0, fontweight="bold", color=COLORS["text"])
    ax.grid(axis="y", color=COLORS["grid"], lw=0.45)
    clean_axes(ax)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.84)
    save_figure(fig, "fig07_panel_f_distribution", PANEL_DIR)
    plt.close(fig)


def main() -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    fig07 = load_fig07_module()
    df = fig07.load_data()
    climate_cols = fig07.climate_columns()
    df = fig07.out_of_fold_climate_component(df, climate_cols)
    corr = fig07.spearman_summary(df, climate_cols)

    save_map_panel(fig07, df, "x_obs", r"(a) Observed feature $X_j$", "YlOrBr", "fig07_panel_a_observed_map", False)
    save_map_panel(
        fig07,
        df,
        "x_hat_climate",
        r"(b) Climate-predicted component $\hat{g}_j(C)$",
        "YlGnBu",
        "fig07_panel_b_climate_component_map",
        False,
    )
    save_map_panel(fig07, df, "residual_robust_z", r"(c) Residual feature $r_j$", "RdBu_r", "fig07_panel_c_residual_map", True)
    save_decomposition_panel(df)
    save_correlation_panel(corr)
    save_distribution_panel(df)

    df[
        [
            "sample_id",
            "Y_label",
            "sample_type",
            "state",
            "latitude",
            "longitude",
            fig07.FEATURE,
            "x_obs",
            "x_hat_climate",
            "residual",
            "residual_robust_z",
        ]
    ].to_csv(PANEL_DIR / "fig07_split_panel_data.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(PANEL_DIR / "fig07_split_panel_spearman_summary.csv", index=False, encoding="utf-8-sig")
    print(PANEL_DIR)


if __name__ == "__main__":
    main()
