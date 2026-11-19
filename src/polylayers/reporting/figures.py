"""README and report figures, rendered once per theme (light and dark)."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polylayers.evaluation.experiment import GNN_STAGES, MARKET_MODEL, RECALIBRATED_MODEL
from polylayers.features.panel import LAYER_STAGES
from polylayers.reporting.theme import THEMES, Theme, apply_theme

logger = logging.getLogger(__name__)

HORIZON_ORDER = [30, 14, 7, 1]
MODEL_LABELS = {
    "uniform": "Uniform over buckets",
    "bucket_base_rate": "Bucket base rate",
    "market_raw": "Market price",
    "market_norm": "Market price (normalized)",
    "market_recalibrated": "Market price, recalibrated",
    "logistic": "Logistic regression",
    "clogit": "Conditional logit",
    "gcn": "GCN",
    "graphconv": "GraphConv",
    "gcn_no_edges": "GCN, no edges",
    "gcn_cross_meeting": "GCN, cross-meeting edges only",
    "gcn_rivals_traders": "GCN, same-meeting + trader edges only",
}
STAGE_LABELS = {
    "market": "Market price",
    "market+trading": "+ trading",
    "market+trading+macro": "+ macro",
    "market+trading+macro+news": "+ news",
    "market+trading+macro+news+text": "+ Fed text",
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


def _rows(scores: pd.DataFrame, model: str, stage: str = "-") -> pd.DataFrame:
    return scores[(scores["model"] == model) & (scores["stage"] == stage)]


def plot_market_by_horizon(scores: pd.DataFrame, theme: Theme) -> plt.Figure:
    """The market, its recalibrated version, and the naive bucket base rate by horizon."""
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), layout="constrained")
    colors = {
        MARKET_MODEL: theme.series[0],
        RECALIBRATED_MODEL: theme.series[1],
        "bucket_base_rate": theme.series[2],
    }
    # Temperature scaling never changes a meeting's favorite, so the recalibrated price would
    # sit exactly on top of the market in the top-1 panel; it is drawn only for Brier.
    panels = (
        (
            "top1_accuracy",
            "Favorite bucket wins",
            "Share of test meetings",
            [MARKET_MODEL, "bucket_base_rate"],
        ),
        ("brier", "Brier score (lower is better)", "Brier score", list(colors)),
    )
    for axis, (metric, title, ylabel, models) in zip(axes, panels, strict=True):
        for model in models:
            _line_by_horizon(
                axis, _rows(scores, model), metric, colors[model], MODEL_LABELS[model], theme
            )
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        _horizon_axis(axis)
    axes[0].set_ylim(0, 1.05)
    axes[1].legend(loc="center right")
    return figure


def _forest(axis, rows: pd.DataFrame, stages: list[str], models: list[str], theme: Theme):
    """Dot-and-whisker plot of differences vs. a reference, one row per stage."""
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
    differences: pd.DataFrame,
    reference: str,
    stages: list[str],
    models: list[str],
    title: str,
    theme: Theme,
) -> plt.Figure:
    """Brier difference vs. `reference` (95% interval) per stage, one panel per horizon."""
    figure, axes = plt.subplots(
        1,
        len(HORIZON_ORDER),
        figsize=(11, 0.55 * len(stages) + 2.0),
        sharey=True,
        layout="constrained",
    )
    brier = differences[
        (differences["metric"] == "brier")
        & (differences["reference"] == reference)
        & differences["model"].isin(models)
    ]
    for axis, horizon in zip(axes, HORIZON_ORDER, strict=True):
        _forest(axis, brier[brier["horizon_days"] == horizon], stages, models, theme)
        axis.set_title(horizon_title(horizon), loc="left")
        axis.set_xlabel("Brier difference")
    # The y-axis is shared, so invert it once: stages read top to bottom in the order added.
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower left", ncol=len(models))
    figure.suptitle(
        title, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=theme.ink_primary
    )
    return figure


def plot_fundamentals(scores: pd.DataFrame, theme: Theme) -> plt.Figure:
    """Without the market's price: macro models vs. the bucket base rate (and the market)."""
    figure, axis = plt.subplots(figsize=(7.5, 3.8), layout="constrained")
    lines = [
        (_rows(scores, MARKET_MODEL), theme.series[0], MODEL_LABELS[MARKET_MODEL]),
        (_rows(scores, "logistic", "macro"), theme.series[1], "Logistic regression, macro"),
        (_rows(scores, "clogit", "macro"), theme.series[2], "Conditional logit, macro"),
        (_rows(scores, "bucket_base_rate"), theme.ink_muted, MODEL_LABELS["bucket_base_rate"]),
    ]
    for frame, color, label in lines:
        _line_by_horizon(axis, frame, "brier", color, label, theme)
    axis.set_title("Without the market's price (Brier, lower is better)", loc="left")
    axis.set_ylabel("Brier score")
    _horizon_axis(axis)
    axis.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return figure


def plot_temperatures(temperatures: pd.DataFrame, theme: Theme) -> plt.Figure:
    """Fitted temperature of the market price per test meeting (beta > 1: underconfident)."""
    figure, axis = plt.subplots(figsize=(8, 3.6), layout="constrained")
    horizons = [30, 7, 1]
    for slot, horizon in enumerate(horizons):
        rows = temperatures[temperatures["horizon_days"] == horizon].copy()
        rows["meeting"] = pd.to_datetime(rows["test_meeting"])
        color = theme.series[slot]
        axis.plot(
            rows["meeting"],
            rows["beta"],
            color=color,
            label=horizon_title(horizon),
            **_marker_style(theme, color),
        )
    axis.axhline(1.0, color=theme.baseline, linewidth=1)
    axis.set_yscale("log")
    ticks = [0.5, 1, 2, 4, 8]
    axis.set_yticks(ticks)
    axis.set_yticklabels([f"{tick:g}" for tick in ticks])
    axis.minorticks_off()
    axis.set_title(
        "Temperature that best recalibrates the market, fit on past meetings", loc="left"
    )
    axis.set_ylabel("beta (log scale; 1 = calibrated)")
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower left", ncol=len(horizons))
    return figure


def save_all_figures(
    scores: pd.DataFrame,
    differences: pd.DataFrame,
    temperatures: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Render every figure in both themes; returns the written paths."""
    plt.switch_backend("Agg")
    output_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        "market_by_horizon": lambda theme: plot_market_by_horizon(scores, theme),
        "layers_vs_recalibrated_market": lambda theme: plot_differences(
            differences,
            RECALIBRATED_MODEL,
            list(LAYER_STAGES),
            ["logistic", "clogit"],
            "What each layer adds on top of the recalibrated market price",
            theme,
        ),
        "graph_models": lambda theme: plot_differences(
            differences,
            RECALIBRATED_MODEL,
            list(GNN_STAGES),
            ["logistic", "gcn", "graphconv"],
            "Graph networks vs. a flat model, relative to the recalibrated market price",
            theme,
        ),
        "fundamentals_only": lambda theme: plot_fundamentals(scores, theme),
        "market_temperature": lambda theme: plot_temperatures(temperatures, theme),
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
