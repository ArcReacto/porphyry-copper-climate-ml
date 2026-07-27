from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "figures"

COLORS = {
    "text": "#172033",
    "muted": "#667085",
    "line": "#A7B1C2",
    "grid": "#E6EAF0",
    "surface": "#F7F9FC",
    "blue": "#0B6F82",
    "blue_light": "#E7F5F7",
    "orange": "#D7651B",
    "orange_light": "#FFF0E6",
    "green": "#2F7D52",
    "green_light": "#EAF6EF",
    "purple": "#6653A6",
    "purple_light": "#F0EEFA",
    "red": "#B42318",
    "red_light": "#FDECEC",
    "gray_light": "#F4F6FA",
}

SERIES = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["purple"],
    COLORS["red"],
]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.2,
            "axes.edgecolor": "#C9D2DF",
            "axes.linewidth": 0.7,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str, out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    return png_path, pdf_path


def clean_axes(ax: plt.Axes, keep_left: bool = True, keep_bottom: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(keep_left)
    ax.spines["bottom"].set_visible(keep_bottom)
    if keep_left:
        ax.spines["left"].set_color("#C9D2DF")
    if keep_bottom:
        ax.spines["bottom"].set_color("#C9D2DF")
    ax.tick_params(colors=COLORS["muted"], length=2.5, width=0.6)


def panel_label(ax: plt.Axes, label: str, x: float = -0.02, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["text"],
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str | None = None,
    facecolor: str | None = None,
    edgecolor: str | None = None,
    title_size: float = 7.4,
    subtitle_size: float = 5.9,
) -> patches.FancyBboxPatch:
    facecolor = facecolor or COLORS["surface"]
    edgecolor = edgecolor or COLORS["line"]
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.014,rounding_size=0.026",
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(box)
    x, y = xy
    title_y = y + height * (0.60 if subtitle else 0.50)
    ax.text(
        x + width * 0.50,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["text"],
        zorder=3,
    )
    if subtitle:
        ax.text(
            x + width * 0.50,
            y + height * 0.32,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=COLORS["muted"],
            linespacing=1.12,
            zorder=3,
        )
    return box


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str | None = None,
    lw: float = 0.9,
    rad: float = 0.0,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color or COLORS["line"],
            lw=lw,
            shrinkA=3,
            shrinkB=4,
            mutation_scale=9,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=5,
    )


def add_direct_labels(
    ax: plt.Axes,
    xs: Iterable[float],
    ys: Iterable[float],
    labels: Iterable[str],
    dy: float = 0.01,
    color: str | None = None,
    fmt_size: float = 5.8,
) -> None:
    for x, y, label in zip(xs, ys, labels):
        ax.text(
            x,
            y + dy,
            label,
            ha="center",
            va="bottom",
            fontsize=fmt_size,
            color=color or COLORS["text"],
        )


apply_style()
