from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd

from cdmpm_figure_style import COLORS, clean_axes, save_figure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "paper_optimization"
    / "climate_sensitivity_graph"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
    / "climate_sensitivity_edges.csv"
)


def pretty(name: str) -> str:
    mapping = {
        "vapor_pressure": "vapor pressure",
        "precipitation": "precipitation",
        "soil_moisture": "soil moisture",
        "mean_temperature": "mean temperature",
        "evapotranspiration_ratio": "ET ratio",
        "vapor_pressure_deficit": "VPD",
        "actual_evapotranspiration": "actual ET",
        "terrain_elevation": "terrain elevation",
        "regional_gravity_anomaly": "regional gravity",
        "deep_regional_gravity_anomaly": "deep gravity",
        "terrain_relief": "terrain relief",
        "deep_gravity_gradient": "deep gradient",
        "shallow_gravity_source_density": "shallow source density",
        "deep_gravity_source_strength": "deep source strength",
        "terrain_roughness": "terrain roughness",
        "terrain_slope": "terrain slope",
    }
    return mapping.get(name, name.replace("_", " "))


def short_edge_label(climate: str, target: str) -> str:
    return f"{pretty(climate).replace('vapor pressure', 'vapor').replace('precipitation', 'precip.')} -> {pretty(target).replace('regional gravity', 'reg. gravity').replace('terrain elevation', 'elev.').replace('shallow source density', 'shallow density')}"


def main() -> None:
    edges = pd.read_csv(INPUT)
    edges = edges[edges["selected_for_graph_guided_m4"].astype(bool)].copy()
    edges = edges.sort_values("combined_climate_sensitivity_score", ascending=False).head(12)

    color_by_group = {
        "terrain": COLORS["green"],
        "geochemistry": COLORS["orange"],
        "geophysics": COLORS["blue"],
        "geology": COLORS["muted"],
    }
    face_by_group = {
        "terrain": COLORS["green_light"],
        "geochemistry": COLORS["orange_light"],
        "geophysics": COLORS["blue_light"],
        "geology": COLORS["surface"],
    }

    climates = edges["climate_concept"].drop_duplicates().tolist()
    targets = edges["target_concept"].drop_duplicates().tolist()
    cy = {c: 0.82 - i * (0.62 / max(1, len(climates) - 1)) for i, c in enumerate(climates)}
    ty = {t: 0.82 - i * (0.62 / max(1, len(targets) - 1)) for i, t in enumerate(targets)}
    target_group = edges.drop_duplicates("target_concept").set_index("target_concept")["target_group"].to_dict()

    fig, (ax_graph, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.75),
        dpi=180,
        gridspec_kw={"width_ratios": [1.23, 0.77], "wspace": 0.26},
    )
    fig.patch.set_facecolor("white")

    ax_graph.set_xlim(0, 1)
    ax_graph.set_ylim(0, 1)
    ax_graph.axis("off")
    ax_graph.set_title("Selected climate-sensitivity graph", loc="left", fontsize=8.6, fontweight="bold", color=COLORS["text"], pad=5)
    ax_graph.text(0.16, 0.90, "Climate concepts", ha="center", va="center", fontsize=6.2, color=COLORS["muted"])
    ax_graph.text(0.73, 0.90, "Prospecting concepts", ha="center", va="center", fontsize=6.2, color=COLORS["muted"])

    score_min = edges["combined_climate_sensitivity_score"].min()
    score_max = edges["combined_climate_sensitivity_score"].max()
    for _, row in edges.iterrows():
        c = row["climate_concept"]
        t = row["target_concept"]
        group = row["target_group"]
        score = row["combined_climate_sensitivity_score"]
        lw = 0.75 + 2.8 * ((score - score_min) / max(1e-9, score_max - score_min))
        alpha = 0.18 + 0.38 * float(row["m4_sensitive_rate"])
        ax_graph.plot([0.29, 0.61], [cy[c], ty[t]], color=color_by_group.get(group, COLORS["line"]), lw=lw, alpha=alpha, zorder=1)

    for c, y in cy.items():
        ax_graph.add_patch(
            patches.FancyBboxPatch((0.035, y - 0.026), 0.250, 0.052, boxstyle="round,pad=0.008,rounding_size=0.016", facecolor=COLORS["purple_light"], edgecolor=COLORS["purple"], lw=0.75, zorder=3)
        )
        ax_graph.text(0.160, y, pretty(c), ha="center", va="center", fontsize=5.8, color=COLORS["text"], zorder=4)

    for t, y in ty.items():
        group = target_group.get(t, "")
        ax_graph.add_patch(
            patches.FancyBboxPatch((0.620, y - 0.026), 0.310, 0.052, boxstyle="round,pad=0.008,rounding_size=0.016", facecolor=face_by_group.get(group, COLORS["surface"]), edgecolor=color_by_group.get(group, COLORS["line"]), lw=0.75, zorder=3)
        )
        ax_graph.text(0.775, y, pretty(t), ha="center", va="center", fontsize=5.45, color=COLORS["text"], zorder=4)

    for i, (group, color) in enumerate([("terrain", COLORS["green"]), ("geochemistry", COLORS["orange"]), ("geophysics", COLORS["blue"])]):
        x = 0.035 + i * 0.150
        ax_graph.plot([x, x + 0.032], [0.055, 0.055], color=color, lw=2.0, alpha=0.55)
        ax_graph.text(x + 0.040, 0.055, group, ha="left", va="center", fontsize=5.1, color=COLORS["muted"])
    ax_graph.text(0.930, 0.055, "width = graph score", ha="right", va="center", fontsize=5.1, color=COLORS["muted"])

    top = edges.head(8).iloc[::-1]
    labels = [short_edge_label(r.climate_concept, r.target_concept) for r in top.itertuples()]
    colors = [color_by_group.get(g, COLORS["line"]) for g in top["target_group"]]
    y = range(len(top))
    ax_bar.barh(list(y), top["global_abs_spearman"], color=colors, alpha=0.82, label="|Spearman|")
    ax_bar.scatter(top["m4_sensitive_rate"], list(y), color=COLORS["purple"], s=18, zorder=5, label="fold rate")
    ax_bar.set_yticks(list(y))
    ax_bar.set_yticklabels([])
    for yi, label in zip(y, labels):
        ax_bar.text(
            0.018,
            yi,
            label,
            ha="left",
            va="center",
            fontsize=4.9,
            color=COLORS["text"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.74, pad=0.6),
            zorder=6,
        )
    ax_bar.set_xlim(0, 1.02)
    ax_bar.set_xlabel("association / fold frequency", color=COLORS["muted"])
    ax_bar.set_title("Top edge evidence", loc="left", fontsize=8.6, fontweight="bold", color=COLORS["text"], pad=5)
    ax_bar.grid(axis="x")
    clean_axes(ax_bar)
    ax_bar.legend(frameon=False, fontsize=5.3, loc="lower right")

    fig.suptitle("Climate sensitivity evidence, not fully identified causal effects", x=0.50, y=0.990, fontsize=9.0, fontweight="bold", color=COLORS["text"])
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.105, top=0.895, wspace=0.30)
    png_path, pdf_path = save_figure(fig, "fig06_climate_sensitivity_graph")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
