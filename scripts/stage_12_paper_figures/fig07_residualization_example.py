from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import shapefile
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cdmpm_figure_style import COLORS, save_figure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("CDMPM_DATA_ROOT", PROJECT_ROOT.parent / "data_raw"))
OUT_DIR = PROJECT_ROOT / "figures"

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_datasets"
    / "by_sample_scheme"
    / "known_mining_neutral"
    / "known_mining_neutral_ratio_1_10"
    / "model_dataset_known_mining_neutral_ratio_1_10_supervised_all_features_v1.parquet"
)
ROLE_TABLE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "standardized_runs"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
    / "00_dataset_profile"
    / "feature_roles.csv"
)
STATES_SHP = DATA_ROOT / "cb_2025_us_state_500k" / "cb_2025_us_state_500k.shp"

FEATURE = "geochem1_usgs_Bi_value_mean_50km"
FEATURE_LABEL = "USGS Bi mean within 50 km"
RIDGE_ALPHA = 10.0
RANDOM_STATE = 20260622

WESTERN_STATES = {
    "Arizona",
    "Washington",
    "Nevada",
    "New Mexico",
    "Oregon",
    "Utah",
    "Wyoming",
    "Colorado",
    "Idaho",
    "Montana",
    "California",
}

CLIMATE_LABELS = {
    "climate_vap_annual_mean": "Vapor pressure",
    "climate_vpd_annual_mean": "VPD",
    "climate_tmin_annual_mean": "Min temperature",
    "env_climate_ppt_annual_mm": "Precipitation",
    "env_aridity_index_ppt_pet": "Aridity index",
    "climate_water_balance_annual_mm": "Water balance",
    "climate_srad_annual_mean": "Solar radiation",
    "climate_dtr_annual_mean": "Temp. range",
    "climate_ws_annual_mean": "Wind speed",
}


def shape_to_patches(shape) -> list[MplPolygon]:
    points = shape.points
    parts = list(shape.parts) + [len(points)]
    patches: list[MplPolygon] = []
    for start, end in zip(parts[:-1], parts[1:]):
        ring = points[start:end]
        if len(ring) >= 3:
            patches.append(MplPolygon(ring, closed=True))
    return patches


def load_state_patches() -> list[MplPolygon]:
    reader = shapefile.Reader(str(STATES_SHP))
    patches: list[MplPolygon] = []
    for shape_record in reader.iterShapeRecords():
        record = shape_record.record.as_dict()
        if record["NAME"] in WESTERN_STATES:
            patches.extend(shape_to_patches(shape_record.shape))
    return patches


def climate_columns() -> list[str]:
    role = pd.read_csv(ROLE_TABLE_PATH)
    cols = role.loc[
        role["is_numeric"].astype(bool) & role["used_as_climate_adjuster"].astype(bool),
        "column",
    ].tolist()
    preferred = [c for c in CLIMATE_LABELS if c in cols]
    rest = [c for c in cols if c not in preferred]
    return preferred + rest


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PATH)
    df = df[df["state"].isin(WESTERN_STATES)].copy()
    required = ["longitude", "latitude", "state", "Y_label", FEATURE]
    df = df.dropna(subset=required)
    # Log transform makes the geochemical anomaly map interpretable under heavy tails.
    floor = max(float(df[FEATURE].quantile(0.01)) * 0.25, 1e-6)
    df["x_obs"] = np.log10(df[FEATURE].astype(float).clip(lower=floor))
    return df


def out_of_fold_climate_component(df: pd.DataFrame, climate_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["x_hat_climate"] = np.nan
    groups = out["state"].fillna("unknown").astype(str)
    splitter = GroupKFold(n_splits=min(5, groups.nunique()))

    for train_idx, test_idx in splitter.split(out, out["Y_label"].astype(int), groups):
        train = out.iloc[train_idx]
        test = out.iloc[test_idx]
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=RIDGE_ALPHA)),
            ]
        )
        model.fit(train[climate_cols], train["x_obs"])
        out.iloc[test_idx, out.columns.get_loc("x_hat_climate")] = model.predict(test[climate_cols])

    out["residual"] = out["x_obs"] - out["x_hat_climate"]
    med = float(out["residual"].median())
    mad = float(np.median(np.abs(out["residual"] - med))) or float(out["residual"].std()) or 1.0
    out["residual_robust_z"] = np.clip((out["residual"] - med) / (1.4826 * mad), -2.5, 2.5)
    return out


def add_state_background(ax, patches: list[MplPolygon]) -> None:
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


def plot_field(ax, df: pd.DataFrame, value_col: str, title: str, cmap: str, diverging: bool = False) -> None:
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
    ax.set_title(title, loc="left", fontsize=7.2, fontweight="bold", color=COLORS["text"], pad=3)
    return contour


def spearman_summary(df: pd.DataFrame, climate_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in climate_cols:
        if col not in df.columns:
            continue
        valid = df[[col, "x_obs", "residual_robust_z"]].dropna()
        if len(valid) < 30:
            continue
        rho_x = spearmanr(valid[col], valid["x_obs"]).statistic
        rho_r = spearmanr(valid[col], valid["residual_robust_z"]).statistic
        rows.append(
            {
                "climate": CLIMATE_LABELS.get(col, col.replace("climate_", "").replace("_annual_mean", "").replace("_", " ")),
                "rho_x": float(rho_x),
                "rho_r": float(rho_r),
                "drop": abs(float(rho_x)) - abs(float(rho_r)),
            }
        )
    return pd.DataFrame(rows).sort_values("drop", ascending=False).head(4)


def add_explanation_box(ax) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        "Decomposition used in the example",
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["text"],
    )
    lines = [
        r"$X_j$: observed geochemical anomaly",
        r"$\hat{g}_j(C)$: climate-predicted component",
        r"$r_j=X_j-\hat{g}_j(C)$: residual anomaly",
    ]
    colors = [COLORS["orange"], COLORS["blue"], COLORS["purple"]]
    for i, (line, color) in enumerate(zip(lines, colors)):
        y = 0.74 - i * 0.23
        ax.add_patch(plt.Rectangle((0.00, y - 0.055), 0.10, 0.105, facecolor=color, alpha=0.18, edgecolor=color, lw=0.65))
        ax.text(0.125, y, line, ha="left", va="center", fontsize=6.2, color=COLORS["text"])
    ax.text(
        0.0,
        0.05,
        "The residual keeps local positive anomalies\nafter removing climate-associated\nbackground variation.",
        ha="left",
        va="bottom",
        fontsize=5.4,
        color=COLORS["muted"],
        wrap=True,
    )


def add_correlation_table(ax, summary: pd.DataFrame) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        "Spearman association with climate",
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["text"],
    )
    cols = ["Climate", r"$|\rho_X|$", r"$|\rho_r|$", "drop"]
    cell_text = []
    for _, row in summary.iterrows():
        cell_text.append(
            [
                str(row["climate"]),
                f"{abs(row['rho_x']):.2f}",
                f"{abs(row['rho_r']):.2f}",
                f"{row['drop']:.2f}",
            ]
        )
    table = ax.table(
        cellText=cell_text,
        colLabels=cols,
        loc="upper left",
        cellLoc="center",
        colLoc="center",
        bbox=[0.00, 0.02, 1.00, 0.78],
        colWidths=[0.43, 0.19, 0.19, 0.19],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.4)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D8DEE8")
        cell.set_linewidth(0.35)
        if r == 0:
            cell.set_facecolor("#EEF6F7")
            cell.set_text_props(weight="bold", color=COLORS["text"])
        elif c == 3:
            cell.set_text_props(color=COLORS["blue"], weight="bold")


def add_distribution_panel(ax, df: pd.DataFrame) -> None:
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
    ax.set_title("Distribution after decomposition", loc="left", fontsize=7.2, fontweight="bold", color=COLORS["text"], pad=2)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([r"$X_j$", r"$\hat{g}_j(C)$", r"$r_j$"], fontsize=6.0)
    ax.set_ylabel("standardized value", fontsize=5.6, color=COLORS["muted"])
    ax.set_ylim(-2.9, 2.9)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.4)
    ax.tick_params(axis="y", labelsize=5.4, colors=COLORS["muted"], length=2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#D0D7E2")
    ax.spines["bottom"].set_color("#D0D7E2")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    climate_cols = climate_columns()
    df = out_of_fold_climate_component(df, climate_cols)
    corr = spearman_summary(df, climate_cols)

    fig = plt.figure(figsize=(7.15, 4.95), dpi=180)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 0.82], hspace=0.30, wspace=0.12)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    c1 = plot_field(ax1, df, "x_obs", r"(a) Observed $X_j$", "YlOrBr", diverging=False)
    c2 = plot_field(ax2, df, "x_hat_climate", r"(b) Climate component $\hat{g}_j(C)$", "YlGnBu", diverging=False)
    c3 = plot_field(ax3, df, "residual_robust_z", r"(c) Residual $r_j=X_j-\hat{g}_j(C)$", "RdBu_r", diverging=True)

    for ax, contour in [
        (ax1, c1),
        (ax2, c2),
        (ax3, c3),
    ]:
        cb = fig.colorbar(contour, ax=ax, fraction=0.045, pad=0.010)
        cb.ax.tick_params(labelsize=4.8, length=1.8, colors=COLORS["muted"])
        cb.outline.set_visible(False)

    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    add_explanation_box(ax4)
    add_correlation_table(ax5, corr)
    add_distribution_panel(ax6, df)

    fig.suptitle(
        f"Climate residualization example: {FEATURE_LABEL}",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["text"],
        y=0.985,
    )
    fig.text(
        0.012,
        0.012,
        "Out-of-fold climate component is estimated with Ridge regression under state-grouped folds; red points mark known porphyry copper positives.",
        ha="left",
        va="bottom",
        fontsize=5.4,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.085, top=0.905)

    png_path, pdf_path = save_figure(fig, "fig07_residualization_example", OUT_DIR)
    plt.close(fig)

    out_csv = OUT_DIR / "fig07_residualization_example_data.csv"
    df[
        [
            "sample_id",
            "Y_label",
            "sample_type",
            "state",
            "latitude",
            "longitude",
            FEATURE,
            "x_obs",
            "x_hat_climate",
            "residual",
            "residual_robust_z",
        ]
    ].to_csv(out_csv, index=False, encoding="utf-8-sig")
    corr.to_csv(OUT_DIR / "fig07_residualization_spearman_summary.csv", index=False, encoding="utf-8-sig")

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
