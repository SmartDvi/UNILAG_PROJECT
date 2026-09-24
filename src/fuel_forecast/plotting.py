"""Consistent figure styling and saving."""

from pathlib import Path

import matplotlib.pyplot as plt

from .config import FIGURE_DIR


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F8F6",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "white",
        "grid.linewidth": 1.2,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "font.family": "DejaVu Sans",
        "figure.dpi": 110,
    })


def save(fig, name: str) -> Path:
    """Save a figure as PNG under ``outputs/figures`` and return its path."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
