"""Modern Medical AI design system - single source of truth for the project.

All notebooks and scripts import this module for a consistent dark theme.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

BG_PRIMARY = "#0F2854"        # page background
BG_PANEL = "#142F63"          # card / panel background
BG_ELEVATED = "#1A3A77"       # hover / elevated panel
ACCENT_PRIMARY = "#1C4D8D"    # primary accent (buttons, headers)
ACCENT_SECONDARY = "#4988C4"  # secondary accent (links, borders)
HIGHLIGHT = "#BDE8F5"         # highlight (titles, key values)
TEXT_PRIMARY = "#F8FAFC"      # main text
TEXT_SECONDARY = "#CBD5E1"    # muted text
GRID_COLOR = "#24406B"        # gridlines / separators
WARN_COLOR = "#E8C56A"        # warnings
POSITIVE_COLOR = "#4CD98C"    # success / positive

CHART_COLORS = [
    "#4988C4", "#BDE8F5", "#7AB0E0", "#1C4D8D", "#5FA8D3", "#8FD3F0",
]


def apply_style() -> None:
    """Apply the theme to matplotlib's global rcParams."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": BG_PRIMARY,
        "axes.facecolor": BG_PRIMARY,
        "savefig.facecolor": BG_PRIMARY,
        "text.color": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.edgecolor": GRID_COLOR,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def palette() -> dict[str, str]:
    """Return the full palette as a dict (for scripts that need it)."""
    return {
        "bg": BG_PRIMARY,
        "panel": BG_PANEL,
        "elevated": BG_ELEVATED,
        "accent": ACCENT_PRIMARY,
        "secondary": ACCENT_SECONDARY,
        "highlight": HIGHLIGHT,
        "text": TEXT_PRIMARY,
        "muted": TEXT_SECONDARY,
        "grid": GRID_COLOR,
        "warn": WARN_COLOR,
        "positive": POSITIVE_COLOR,
    }
