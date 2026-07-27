from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cdmpm_figure_style import COLORS, clean_axes, save_figure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "paper_optimization"
    / "shap_explanations"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)


def shorten_feature(name: str) -> str:
    replacements = [
        ("_climate_resid", " resid."),
        ("geochem1_usgs_", "USGS "),
        ("geochem2_nure_", "NURE "),
        ("gravity_cmmi_", "CMMI "),
        ("gravity_na_", "NA gravity "),
        ("_value_", " "),
        ("_ppm_", " ppm "),
        ("_median_", " med. "),
        ("_max_", " max "),
        ("_", " "),
    ]
    for old, new in replacements:
        name = name.replace(old, new)
    return name[:38] + ("..." if len(name) > 38 else "")


def main() -> None:
    group = pd.read_csv(INPUT_DIR / "shap_feature_group_importance.csv")
    detail = pd.read_csv(INPUT_DIR / "shap_feature_importance_detail.csv")

    model_order = ["M2_NoClimate_XGB", "M4_GraphUnion_XGB"]
    model_labels = ["M2 no climate", "M4 graph-guided"]
    role_labels = {
        "geochemistry": "Geochemistry",
        "geo_structure": "Geology + terrain",
        "geophysics": "Geophysics",
        "climate": "Climate",
        "other": "Other",
    }
    role_colors = {
        "geochemistry": COLORS["orange"],
        "geo_structure": COLORS["green"],
        "geophysics": COLORS["blue"],
        "climate": COLORS["purple"],
        "other": COLORS["line"],
    }
    roles = [r for r in ["geochemistry", "geo_structure", "geophysics", "climate", "other"] if r in set(group["role"])]

    shares = np.zeros((len(model_order), len(roles)))
    for i, model in enumerate(model_order):
        sub = group[group["model"].eq(model)].set_index("role")
        for j, role in enumerate(roles):
            shares[i, j] = float(sub.loc[role, "share"]) if role in sub.index else 0.0

    m4_top = detail[detail["model"].eq("M4_GraphUnion_XGB")].head(9).iloc[::-1].copy()
    bar_colors = np.where(m4_top["is_climate_residual"].to_numpy(), COLORS["orange"], COLORS["blue"])

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.25),
        dpi=180,
        gridspec_kw={"width_ratios": [0.82, 1.18], "wspace": 0.36},
    )
    fig.patch.set_facecolor("white")

    bottom = np.zeros(len(model_order))
    x = np.arange(len(model_order))
    for role in roles:
        vals = shares[:, roles.index(role)]
        ax1.bar(x, vals, bottom=bottom, color=role_colors.get(role, COLORS["line"]), alpha=0.86, label=role_labels.get(role, role))
        bottom += vals
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels)
    ax1.set_ylim(0, 1.04)
    ax1.set_ylabel("Share of mean |SHAP|", color=COLORS["muted"])
    ax1.set_title("Feature-family contribution", loc="left", fontsize=8.5, fontweight="bold", color=COLORS["text"])
    ax1.grid(axis="y")
    clean_axes(ax1)
    ax1.legend(frameon=False, fontsize=5.3, loc="upper center", bbox_to_anchor=(0.50, -0.14), ncol=1)

    y = np.arange(len(m4_top))
    ax2.barh(y, m4_top["mean_abs_shap"], color=bar_colors, alpha=0.86)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    for yi, feature in zip(y, m4_top["feature"]):
        ax2.text(
            0.008,
            yi,
            shorten_feature(feature),
            ha="left",
            va="center",
            fontsize=5.0,
            color=COLORS["text"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.6),
            zorder=6,
        )
    ax2.set_xlabel("Mean |SHAP|", color=COLORS["muted"])
    ax2.set_title("Top graph-guided M4 features", loc="left", fontsize=8.5, fontweight="bold", color=COLORS["text"])
    ax2.grid(axis="x")
    clean_axes(ax2)

    ax2.text(
        0.98,
        -0.20,
        "Orange marks indicate climate-residualized features.",
        transform=ax2.transAxes,
        ha="right",
        va="center",
        fontsize=5.2,
        color=COLORS["muted"],
    )

    fig.suptitle("Model explanation after graph-guided adjustment", fontsize=9.0, fontweight="bold", color=COLORS["text"], y=0.990)
    fig.subplots_adjust(left=0.090, right=0.985, bottom=0.190, top=0.850, wspace=0.52)
    png_path, pdf_path = save_figure(fig, "fig07_shap_explanation")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
