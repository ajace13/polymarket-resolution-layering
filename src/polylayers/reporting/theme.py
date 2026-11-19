"""Light and dark chart themes (validated categorical palette, recessive chrome).

Categorical slots are always assigned in the same order, and charts that place dots or
whiskers next to each other use at most three of them (the first three slots stay
distinguishable for color-vision deficiencies in every pairing).
"""

from dataclasses import dataclass

import matplotlib as mpl


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink_primary: str
    ink_secondary: str
    ink_muted: str
    gridline: str
    baseline: str
    series: tuple[str, ...]


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink_primary="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    gridline="#e1e0d9",
    baseline="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
)
DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink_primary="#f0efec",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    gridline="#2c2c2a",
    baseline="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"),
)
THEMES = (LIGHT, DARK)


def apply_theme(theme: Theme) -> None:
    """Set matplotlib defaults: system sans, hairline solid grid, quiet axes."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "axes.edgecolor": theme.baseline,
            "axes.linewidth": 0.8,
            "axes.labelcolor": theme.ink_secondary,
            "axes.titlecolor": theme.ink_primary,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": theme.gridline,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": theme.ink_muted,
            "ytick.color": theme.ink_muted,
            "xtick.labelcolor": theme.ink_secondary,
            "ytick.labelcolor": theme.ink_secondary,
            "legend.frameon": False,
            "legend.labelcolor": theme.ink_secondary,
            "text.color": theme.ink_primary,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )
