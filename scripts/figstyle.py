"""
Shared figure style for the multicentre complexity paper.

Journal conventions applied here:
  - no figure titles and no suptitles; all explanation lives in the figure legend
    (see FIGURE_LEGENDS.md)
  - panel titles, where present, are short and descriptive, never interpretive
  - panel letters are plain lowercase-weight capitals placed outside the axes
  - muted categorical palette, hairline spines, top/right spines removed
  - every figure is written as both PNG (300 dpi) and SVG (editable vector)

Usage:
    import figstyle as fs
    fs.apply()
    ...
    fs.save(fig, OUT, "Figure3")
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── palette ───────────────────────────────────────────────────────────────────
# Muted, colour-blind-safe. Group colours are deliberately low-saturation so that
# overlaid confidence bands remain legible in print.
GROUP = {"Control": "#4C72B0", "PD": "#C44E52", "Other": "#8C8C8C"}
COHORT = {"CETRAM": "#4C72B0", "Cruces": "#55A868", "Nagoya": "#DD8452"}
SEQ = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
       "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
ACCENT = "#8172B3"
MUTED = "#8C8C8C"

_RC = {
    "figure.facecolor": "white",
    "figure.dpi": 110,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.titlepad": 6,
    "axes.labelsize": 9,
    "axes.labelcolor": "0.15",
    "axes.edgecolor": "0.55",
    "axes.linewidth": 0.8,
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": plt.cycler(color=SEQ),
    "grid.color": "0.90",
    "grid.linewidth": 0.7,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.color": "0.25",
    "ytick.color": "0.25",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.handlelength": 1.6,
    "legend.borderaxespad": 0.4,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "patch.linewidth": 0.7,
    "boxplot.flierprops.markersize": 3,
    "svg.fonttype": "none",        # keep text editable in the SVG
    "pdf.fonttype": 42,
}


def apply() -> None:
    """Install the publication rcParams."""
    plt.rcParams.update(_RC)


def panel(ax, letter: str, dx: float = -0.16, dy: float = 1.06) -> None:
    """Panel letter, outside the axes, upper left."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def despine(ax, left: bool = False, bottom: bool = False) -> None:
    for side, off in (("top", True), ("right", True), ("left", left), ("bottom", bottom)):
        ax.spines[side].set_visible(not off)


def save(fig, outdir: str, stem: str, dpi: int = 300) -> str:
    """Write PNG + SVG. No title is added; titles belong in the legend."""
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"{stem}.png")
    fig.savefig(png, dpi=dpi)
    fig.savefig(os.path.join(outdir, f"{stem}.svg"))
    plt.close(fig)
    return png
