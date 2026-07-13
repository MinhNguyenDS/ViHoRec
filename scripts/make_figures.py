"""Generate ACL-style statistics figures for the ViHoRec paper.

Run:  python make_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

import config as C
import plot_style as ps

IMG_DIR = C.PAPER_IMG_DIR


def _load():
    inter = pd.read_csv(C.OUT_RELEASE / "interactions.csv")
    hotels = pd.read_csv(C.OUT_RELEASE / "hotels.csv")
    return inter, hotels


def fig_hotels_per_location(hotels: pd.DataFrame) -> None:
    counts = hotels["location"].value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(ps.COL_WIDTH, 2.6))
    ax.barh(counts.index, counts.values, color=ps.VIH_BLUE, height=0.72, edgecolor="none")
    ax.set_xlabel("Hotels")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.35)
    for i, v in enumerate(counts.values):
        ax.text(v + 0.4, i, str(int(v)), va="center", fontsize=7)
    ps.save_fig(fig, IMG_DIR / "NumberHotel.png")
    plt.close(fig)


def fig_rating_distribution(inter: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(ps.COL_WIDTH, 2.4))
    ax.hist(inter["rating"], bins=[i * 0.5 for i in range(2, 21)],
            color=ps.VIH_BLUE, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Reviews")
    ax.grid(axis="y", alpha=0.35)
    ps.save_fig(fig, IMG_DIR / "phanphoiHotel.png")
    plt.close(fig)


def fig_ratings_per_hotel(inter: pd.DataFrame) -> None:
    per_hotel = inter.groupby("hotel_id").size()
    fig, ax = plt.subplots(figsize=(ps.COL_WIDTH, 2.4))
    ax.hist(per_hotel, bins=35, color=ps.VIH_BLUE, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Reviews per hotel")
    ax.set_ylabel("Hotels")
    ax.grid(axis="y", alpha=0.35)
    ps.save_fig(fig, IMG_DIR / "HotelRating.png")
    plt.close(fig)


def fig_ratings_per_user(inter: pd.DataFrame) -> None:
    per_user = inter.groupby("user_id").size()
    cap = 30
    clipped = per_user.clip(upper=cap)
    fig, ax = plt.subplots(figsize=(ps.COL_WIDTH, 2.4))
    ax.hist(clipped, bins=range(1, cap + 2), color=ps.VIH_BLUE, edgecolor="white",
            linewidth=0.4, align="left", rwidth=0.92)
    ax.set_xlabel(f"Reviews per user (cap {cap})")
    ax.set_ylabel("Users")
    ax.set_yscale("log")
    ax.set_xlim(0, cap + 1)
    ax.grid(axis="y", alpha=0.35)
    ps.save_fig(fig, IMG_DIR / "CusRating.png")
    plt.close(fig)


def fig_pipeline() -> None:
    steps = [
        "Crawl\n(Booking, Traveloka, Ivivu)",
        "Entity\nresolution",
        "Quality\ncontrol",
        "Anonymize\n(HMAC)",
        "Release +\nbenchmark",
    ]
    fig, ax = plt.subplots(figsize=(ps.PAGE_WIDTH, 1.35))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, text in enumerate(steps):
        box = FancyBboxPatch((i + 0.05, 0.22), 0.86, 0.56,
                             boxstyle="round,pad=0.02,rounding_size=0.04",
                             linewidth=0.9, edgecolor=ps.VIH_BLUE, facecolor=ps.VIH_LIGHT)
        ax.add_patch(box)
        ax.text(i + 0.48, 0.50, text, ha="center", va="center", fontsize=7)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((i + 0.92, 0.50), (i + 1.03, 0.50),
                         arrowstyle="-|>", mutation_scale=10, color=ps.VIH_GRAY,
                         linewidth=1.0))
    ps.save_fig(fig, IMG_DIR / "collect_preprocess.png")
    plt.close(fig)


def run() -> None:
    ps.apply_acl_style()
    inter, hotels = _load()
    fig_hotels_per_location(hotels)
    fig_rating_distribution(inter)
    fig_ratings_per_hotel(inter)
    fig_ratings_per_user(inter)
    fig_pipeline()
    print("Figures written to", IMG_DIR)


if __name__ == "__main__":
    run()
