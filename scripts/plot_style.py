"""Shared matplotlib styling for ViHoRec paper figures (ACL column-width)."""

from __future__ import annotations

import matplotlib.pyplot as plt

# Palette aligned with acl_latex.tex (vihblue / vihaccent / vihgray).
VIH_BLUE = "#2C6FBB"
VIH_ACCENT = "#E08A1E"
VIH_GRAY = "#6B6B6B"
VIH_LIGHT = "#EAF1FB"

# Single-column ~3.33 in; two-panel ~7.0 in at 11 pt body.
COL_WIDTH = 3.35
PAGE_WIDTH = 7.0

ACL_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#CCCCCC",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_acl_style() -> None:
    plt.rcParams.update(ACL_RC)


def save_fig(fig, path, *, transparent: bool = False) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02,
                transparent=transparent)
