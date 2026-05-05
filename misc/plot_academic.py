"""
misc/plot_academic.py

Shared plot style, palette, and academic-quality plot functions for GTA-PS.
Standardized for ML paper quality.

Usage
-----
from plot_academic import apply_style, get_palette, LABELS, get_utility_color

apply_style()
colors = get_palette("line")  # Strict Viridis for time-series
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical style
# ---------------------------------------------------------------------------

RCPARAMS: dict = {
    "font.size":        12,
    "axes.titlesize":   14,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "lines.linewidth":   2,
    "lines.markersize":  8,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.frameon":    False,
    "savefig.dpi":      300,
}

def apply_style() -> None:
    """Apply canonical rcParams."""
    plt.rcParams.update(RCPARAMS)

# ---------------------------------------------------------------------------
# Color Palette Toolkit
# ---------------------------------------------------------------------------

def _get_viridis_hex(val: float) -> str:
    """Sample strictly from the Viridis colormap."""
    return mcolors.to_hex(plt.get_cmap("viridis")(val))


def _get_cmap_hex(name: str, val: float) -> str:
    """Sample a Matplotlib colormap at a fixed location."""
    return mcolors.to_hex(plt.get_cmap(name)(val))


def _extend_palette(base: dict[str, str], extras: list[str], values: list[float], sampler) -> dict[str, str]:
    palette = dict(base)
    missing = [key for key in extras if key not in palette]
    if not missing:
        return palette

    if len(missing) > len(values):
        extra_values = np.linspace(0.08, 0.92, num=len(missing)).tolist()
    else:
        extra_values = values[:len(missing)]

    for key, val in zip(missing, extra_values):
        palette[key] = sampler(val)
    return palette

# PALETTE_LINE  — Viridis (sampled for time-series).
# PALETTE_POINT — High-contrast (Red/Black/Blue); for discrete condition scatter.
PALETTE_LINE: dict[str, str] = {
    "standard_ps": _get_viridis_hex(0.0),  # Darkest Viridis (Black/Purple)
    "gta_base":    _get_viridis_hex(0.9),  # Yellow/Green
    "gta_beta":    _get_viridis_hex(0.45),  # Teal
    "forward_gta": _get_viridis_hex(0.9),
    "full_gta":    _get_viridis_hex(0.45),
}

LINE_EXTENSION_VALUES = [0.82, 0.42, 0.94, 0.72, 0.32, 0.52]

PALETTE_POINT: dict[str, str] = {
    "standard_ps": _get_cmap_hex("RdBu_r", 0.88),
    "gta_base":    mcolors.to_hex("black"),
    "gta_beta":    _get_cmap_hex("RdBu_r", 0.12),
    "forward_gta": mcolors.to_hex("black"),
    "full_gta":    _get_cmap_hex("RdBu_r", 0.12),
}

POINT_EXTENSION_VALUES = [0.28, 0.72, 0.4, 0.6]

UTILITY_COLORS: dict[str, str] = {
    "theory":     mcolors.to_hex("tab:orange"),
    "raw_signal": mcolors.to_hex("tab:brown"),
    "grid":       mcolors.to_hex("silver"),
    "bg_gray":    mcolors.to_hex("whitesmoke"),
    "text_main":  mcolors.to_hex("darkslategray"),
}

def get_palette(style: str = "line", extra_keys: list[str] | tuple[str, ...] | None = None) -> dict[str, str]:
    extras = list(extra_keys or [])
    if style == "line":
        return _extend_palette(PALETTE_LINE, extras, LINE_EXTENSION_VALUES, _get_viridis_hex)
    return _extend_palette(PALETTE_POINT, extras, POINT_EXTENSION_VALUES, lambda val: _get_cmap_hex("RdBu_r", val))

def get_utility_color(key: str) -> str:
    return UTILITY_COLORS.get(key, mcolors.to_hex("black"))


# Back-compat alias
PALETTE = PALETTE_LINE

LABELS: dict[str, str] = {
    "standard_ps": "Standard-PS",
    "gta_base":    "GTA-base",
    "gta_beta":    "GTA-beta",
    "forward_gta": "Forward-GTA",
    "full_gta":    "Full-GTA",
}

# ---------------------------------------------------------------------------
# Pre-defined plot renderers (RQ2, RQ3, swift test)
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR   = ROOT / "plots"


def plot_swift_test_template() -> Path:
    """Render a fast palette/template figure with dummy data for visual checks."""
    apply_style()

    line_keys = ["standard_ps", "forward_gta", "full_gta"]
    viridis = plt.get_cmap("viridis")
    viridis_positions = np.linspace(0.02, 0.98, num=len(line_keys))
    line_colors = {
        key: viridis(pos)
        for key, pos in zip(line_keys, viridis_positions)
    }
    point_colors = get_palette("point")
    grid_color = get_utility_color("grid")
    bg_color = get_utility_color("bg_gray")
    theory_color = get_utility_color("theory")
    raw_color = get_utility_color("raw_signal")

    backup_steps = np.array([0, 20, 40, 60, 80, 100], dtype=float)
    line_series: dict[str, np.ndarray] = {
        "standard_ps": np.array([0.74, 0.69, 0.63, 0.59, 0.56, 0.53]),
        "forward_gta": np.array([0.72, 0.64, 0.56, 0.49, 0.44, 0.40]),
        "full_gta": np.array([0.70, 0.60, 0.50, 0.41, 0.34, 0.28]),
    }
    line_cis: dict[str, np.ndarray] = {
        "standard_ps": np.array([0.020, 0.018, 0.017, 0.015, 0.014, 0.013]),
        "forward_gta": np.array([0.018, 0.017, 0.015, 0.014, 0.013, 0.012]),
        "full_gta": np.array([0.016, 0.015, 0.013, 0.012, 0.011, 0.010]),
    }

    budget_labels = ["K=10", "K=50", "K=100"]
    budget_positions = np.arange(len(budget_labels), dtype=float)
    point_means: dict[str, np.ndarray] = {
        "standard_ps": np.array([0.62, 0.57, 0.54]),
        "forward_gta": np.array([0.59, 0.50, 0.45]),
        "full_gta": np.array([0.55, 0.43, 0.36]),
    }
    point_errors: dict[str, np.ndarray] = {
        "standard_ps": np.array([0.025, 0.020, 0.018]),
        "forward_gta": np.array([0.020, 0.018, 0.016]),
        "full_gta": np.array([0.018, 0.016, 0.014]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle("Swift Test Template", y=1.02)

    line_ax, point_ax = axes
    for ax in axes:
        ax.set_facecolor(bg_color)
        ax.grid(True, color=grid_color)

    for key in line_keys:
        line_ax.fill_between(
            backup_steps,
            line_series[key] - line_cis[key],
            line_series[key] + line_cis[key],
            color=line_colors[key],
            alpha=0.10,
        )
        line_ax.plot(
            backup_steps,
            line_series[key],
            color=line_colors[key],
            label=LABELS[key],
            linewidth=1.5,
        )

    line_ax.axhline(0.50, color=theory_color, linestyle="--", linewidth=1.5, label="Theory Ref.")
    line_ax.set_xlabel("Planning Backups")
    line_ax.set_ylabel("Dummy Error")
    line_ax.set_title("Template Timeline")
    line_ax.legend(loc="upper right")

    point_offsets = {
        "standard_ps": -0.18,
        "forward_gta": 0.00,
        "full_gta": 0.18,
    }
    for key in ["standard_ps", "forward_gta", "full_gta"]:
        point_ax.errorbar(
            budget_positions + point_offsets[key],
            point_means[key],
            yerr=point_errors[key],
            fmt="o",
            capsize=4,
            color=point_colors[key],
            label=LABELS[key],
        )

    point_ax.plot(
        budget_positions,
        np.array([0.66, 0.61, 0.58]),
        color=raw_color,
        linewidth=1.5,
        linestyle=":",
        marker="x",
        label="Raw Signal",
    )
    point_ax.set_xticks(budget_positions)
    point_ax.set_xticklabels(budget_labels)
    point_ax.set_xlabel("Budget Template")
    point_ax.set_ylabel("Dummy Score")
    point_ax.set_title("Template Points")
    point_ax.legend(loc="upper right")

    plt.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    output_path = PLOTS_DIR / "swift_test_palette_template.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def main() -> None:
    output_path = plot_swift_test_template()
    print(f"Saved template plot to {output_path}")

if __name__ == "__main__":
    main()
