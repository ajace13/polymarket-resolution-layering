"""README and report figures, rendered once per theme (light and dark)."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polylayers.evaluation.experiment import GNN_STAGES, REFERENCE_MODEL
from polylayers.features.panel import FUNDAMENTAL_STAGES, LAYER_STAGES
from polylayers.reporting.theme import THEMES, Theme, apply_theme

logger = logging.getLogger(__name__)

HORIZON_ORDER = [30, 14, 7, 1]
MODEL_LABELS = {
    "uniform": "Uniform over buckets",
    "market_raw": "Market price",
    "market_norm": "Market price (normalized)",
    "logistic": "Logistic regression",
    "clogit": "Conditional logit",
    "gcn": "GCN",
    "graphconv": "GraphConv",
}
STAGE_LABELS = {
    "market": "Market only",
    "market+macro": "+ macro",
    "market+macro+news": "+ news",
    "market+macro+news+text": "+ Fed text",
    "macro": "Macro",
    "macro+news": "Macro + news",
    "macro+news+text": "Macro + news + text",
}


def horizon_title(horizon: int) -> str:
    return f"{horizon} day before" if horizon == 1 else f"{horizon} days before"


def _marker_style(theme: Theme, color: str) -> dict:
    """Filled marker with a surface-colored ring so overlaps stay legible."""
    return {
        "marker": "o",
        "markersize": 7,
        "markerfacecolor": color,
        "markeredgecolor": theme.surface,
        "markeredgewidth": 1.5,
    }


def _horizon_axis(axis) -> None:
    axis.set_xticks(range(len(HORIZON_ORDER)))
    axis.set_xticklabels([f"{h}d" for h in HORIZON_ORDER])
    axis.set_xlabel("Days before the decision")


def _line_by_horizon(axis, frame: pd.DataFrame, metric: str, color: str, label: str, theme):
    values = frame.set_index("horizon_days").reindex(HORIZON_ORDER)[metric].to_numpy()
    axis.plot(
        range(len(HORIZON_ORDER)), values, color=color, label=label, **_marker_style(theme, color)
    )


def plot_market_by_horizon(scores: pd.DataFrame, theme: Theme) -> plt.Figure:
    """How much the market already knows: top-1 accuracy and Brier score by horizon."""
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    market = scores[scores["model"] == REFERENCE_MODEL]
    uniform = scores[scores["model"] == "uniform"]
    panels = (
        ("top1_accuracy", "Market favorite wins", "Share of test meetings"),
        ("brier", "Brier score (lower is better)", "Brier score"),
    )
    for axis, (metric, title, ylabel) in zip(axes, panels, strict=True):
        _line_by_horizon(
            axis, market, metric, theme.series[0], MODEL_LABELS[REFERENCE_MODEL], theme
        )
        _line_by_horizon(axis, uniform, metric, theme.ink_muted, MODEL_LABELS["uniform"], theme)
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        _horizon_axis(axis)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="center right")
    return figure


def _forest(axis, rows: pd.DataFrame, stages: list[str], models: list[str], theme: Theme):
    """Dot-and-whisker plot of differences vs. the market, one row per stage."""
    offsets = np.linspace(-0.18, 0.18, len(models)) if len(models) > 1 else [0.0]
    for slot, (model, offset) in enumerate(zip(models, offsets, strict=True)):
        color = theme.series[slot]
        subset = rows[rows["model"] == model].set_index("stage").reindex(stages)
        positions = np.arange(len(stages)) + offset
        axis.hlines(positions, subset["ci_low"], subset["ci_high"], color=color, linewidth=2)
        axis.plot(
            subset["difference"],
            positions,
            linestyle="none",
            label=MODEL_LABELS[model],
            **_marker_style(theme, color),
        )
    axis.axvline(0, color=theme.baseline, linewidth=1)
    axis.set_yticks(range(len(stages)))
    axis.set_yticklabels([STAGE_LABELS[stage] for stage in stages])
    axis.grid(axis="x", visible=True)
    axis.grid(axis="y", visible=False)


def plot_differences(
    differences: pd.DataFrame, stages: list[str], models: list[str], title: str, theme: Theme
) -> plt.Figure:
    """Brier difference vs. the market (95% interval) per stage, one panel per horizon."""
    figure, axes = plt.subplots(
        1,
        len(HORIZON_ORDER),
        figsize=(11, 0.55 * len(stages) + 2.0),
        sharey=True,
        layout="constrained",
    )
    brier = differences[(differences["metric"] == "brier") & differences["model"].isin(models)]
    for axis, horizon in zip(axes, HORIZON_ORDER, strict=True):
        _forest(axis, brier[brier["horizon_days"] == horizon], stages, models, theme)
        axis.set_title(horizon_title(horizon), loc="left")
        axis.set_xlabel("Brier difference vs. market")
    # The y-axis is shared, so invert it once: stages read top to bottom in the order added.
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower left", ncol=len(models))
    figure.suptitle(
        title, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=theme.ink_primary
    )
    return figure


def plot_fundamentals(scores: pd.DataFrame, theme: Theme) -> plt.Figure:
    """Can fundamentals alone pick the outcome? Brier by horizon without market prices."""
    figure, axis = plt.subplots(figsize=(7.5, 3.8), layout="constrained")
    best_stage = list(FUNDAMENTAL_STAGES)[-1]
    stage_text = STAGE_LABELS[best_stage].lower()
    lines = [
        (
            scores[scores["model"] == REFERENCE_MODEL],
            theme.series[0],
            MODEL_LABELS[REFERENCE_MODEL],
        ),
        (
            scores[(scores["model"] == "clogit") & (scores["stage"] == best_stage)],
            theme.series[1],
            f"Conditional logit, {stage_text}",
        ),
        (
            scores[(scores["model"] == "logistic") & (scores["stage"] == best_stage)],
            theme.series[2],
            f"Logistic regression, {stage_text}",
        ),
        (scores[scores["model"] == "uniform"], theme.ink_muted, MODEL_LABELS["uniform"]),
    ]
    for frame, color, label in lines:
        _line_by_horizon(axis, frame, "brier", color, label, theme)
    axis.set_title("Fundamentals alone vs. the market (Brier, lower is better)", loc="left")
    axis.set_ylabel("Brier score")
    _horizon_axis(axis)
    axis.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return figure


def save_all_figures(
    scores: pd.DataFrame, differences: pd.DataFrame, output_dir: Path
) -> list[Path]:
    """Render every figure in both themes; returns the written paths."""
    plt.switch_backend("Agg")
    output_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        "market_by_horizon": lambda theme: plot_market_by_horizon(scores, theme),
        "layers_vs_market": lambda theme: plot_differences(
            differences,
            list(LAYER_STAGES),
            ["logistic", "clogit"],
            "Adding information layers on top of the market's price",
            theme,
        ),
        "graph_models_vs_market": lambda theme: plot_differences(
            differences,
            list(GNN_STAGES),
            ["logistic", "gcn", "graphconv"],
            "Graph networks vs. a flat model, relative to the market",
            theme,
        ),
        "fundamentals_only": lambda theme: plot_fundamentals(scores, theme),
    }
    written = []
    for theme in THEMES:
        apply_theme(theme)
        for name, build in builders.items():
            figure = build(theme)
            path = output_dir / f"{name}_{theme.name}.png"
            figure.savefig(path, dpi=200)
            plt.close(figure)
            written.append(path)
    logger.info("Wrote %d figures to %s", len(written), output_dir)
    return written
