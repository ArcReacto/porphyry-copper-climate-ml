from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "figures"
PRED_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "paper_optimization"
    / "full_feature_graph_guided_m4"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
    / "full_feature_graph_guided_m4_predictions.csv"
)

TEXT = "#1F2937"
MUTED = "#64748B"
BLUE = "#0F6B7A"
BLUE_LIGHT = "#E8F6F8"
ORANGE = "#D95F02"
ORANGE_LIGHT = "#FFF1E6"
GREEN = "#3A7D44"
GREEN_LIGHT = "#ECF7EE"
RED = "#B42318"
RED_LIGHT = "#FEF3F2"
GRAY_LIGHT = "#F6F8FB"


def clean_regime(value: str) -> str:
    if not isinstance(value, str) or not value:
        return "unknown regime"
    return value.replace("__", "\n").replace("_", " ")


def draw_card(ax, x, y, w, h, title, subtitle, lines, face, edge):
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.016,rounding_size=0.032",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + 0.035, y + h - 0.055, title, ha="left", va="center", fontsize=7.3, fontweight="bold", color=TEXT, zorder=3)
    ax.text(x + 0.035, y + h - 0.095, subtitle, ha="left", va="center", fontsize=5.8, color=MUTED, zorder=3)
    yy = y + h - 0.155
    for label, value in lines:
        ax.text(x + 0.035, yy, label, ha="left", va="center", fontsize=5.8, color=MUTED, zorder=3)
        if label == "evidence":
            ax.text(x + 0.035, yy - 0.042, value, ha="left", va="center", fontsize=5.55, color=TEXT, zorder=3)
            yy -= 0.085
        else:
            ax.text(x + w - 0.035, yy, value, ha="right", va="center", fontsize=5.9, color=TEXT, zorder=3)
            yy -= 0.055


def rank_percentile(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rank"] = out.groupby("model")["y_prob"].rank(ascending=False, method="first")
    out["rank_pct"] = out["rank"] / out.groupby("model")["y_prob"].transform("count")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PRED_PATH)
    pred = pred[pred["cv"].eq("groupkfold_state")].copy()
    pred = pred[pred["model"].isin(["NoClimate_RF", "M4_GraphUnion_RF"])].copy()
    pred = rank_percentile(pred)

    m2 = pred[pred["model"].eq("NoClimate_RF")].set_index("sample_id")
    m4 = pred[pred["model"].eq("M4_GraphUnion_RF")].set_index("sample_id")
    joined = m4.join(m2[["y_prob", "rank_pct"]], rsuffix="_m2")
    joined = joined.rename(columns={"y_prob": "m4_prob", "rank_pct": "m4_rank_pct", "y_prob_m2": "m2_prob", "rank_pct_m2": "m2_rank_pct"})
    joined["prob_gain"] = joined["m4_prob"] - joined["m2_prob"]
    joined["rank_gain"] = joined["m2_rank_pct"] - joined["m4_rank_pct"]

    top_tp = joined[joined["Y_label"].eq(1)].sort_values("m4_prob", ascending=False).iloc[0]
    gain_tp = joined[joined["Y_label"].eq(1)].sort_values(["rank_gain", "prob_gain"], ascending=False).iloc[0]
    false_pos = joined[joined["Y_label"].eq(0)].sort_values("m4_prob", ascending=False).iloc[0]

    cases = [
        (
            "Top-ranked positive",
            "strong mineralization evidence",
            top_tp,
            GREEN_LIGHT,
            GREEN,
            "geochemistry + structure",
        ),
        (
            "Decoupling gain",
            "rank improves after M4",
            gain_tp,
            ORANGE_LIGHT,
            ORANGE,
            "climate-adjusted features",
        ),
        (
            "High-score error",
            "limitation / hard negative",
            false_pos,
            RED_LIGHT,
            RED,
            "ambiguous geoscience signal",
        ),
    ]

    fig, ax = plt.subplots(figsize=(7.1, 3.15), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.935, "Interpretable ranking cases", ha="center", va="center", fontsize=10.7, fontweight="bold", color=TEXT)

    card_w = 0.285
    xs = [0.045, 0.3575, 0.67]
    for x, (title, subtitle, row, face, edge, evidence) in zip(xs, cases):
        rank_text = f"top {row['m4_rank_pct'] * 100:.1f}%"
        delta_text = f"{row['prob_gain']:+.3f}"
        lines = [
            ("state", str(row["state"])),
            ("label", "positive" if int(row["Y_label"]) == 1 else "hard negative"),
            ("M4 score", f"{row['m4_prob']:.3f}"),
            ("M4 rank", rank_text),
            ("vs. M2 score", delta_text),
            ("evidence", evidence),
        ]
        draw_card(ax, x, 0.31, card_w, 0.48, title, subtitle, lines, face, edge)
        ax.text(x + card_w / 2, 0.215, clean_regime(row.get("env_weathering_regime", "")), ha="center", va="center", fontsize=5.7, color=MUTED, linespacing=1.12)

    ax.add_patch(
        patches.FancyBboxPatch(
            (0.16, 0.055),
            0.68,
            0.085,
            boxstyle="round,pad=0.012,rounding_size=0.026",
            facecolor=GRAY_LIGHT,
            edgecolor="#CBD5E1",
            linewidth=0.8,
            zorder=1,
        )
    )
    ax.text(
        0.50,
        0.098,
        "Cases are selected from GroupKFold predictions: best true positive, largest M4 rank gain, and highest-scored false positive.",
        ha="center",
        va="center",
        fontsize=5.9,
        color=MUTED,
        zorder=2,
    )

    fig.tight_layout(pad=0.18)
    png_path = OUT_DIR / "fig08_case_studies.png"
    pdf_path = OUT_DIR / "fig08_case_studies.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
