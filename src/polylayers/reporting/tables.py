"""Markdown summary tables generated from the evaluation results."""

import pandas as pd

from polylayers.evaluation.bootstrap import paired_meeting_bootstrap
from polylayers.evaluation.experiment import (
    FULL_STAGE,
    GCN_ABLATIONS,
    MARKET_MODEL,
    RECALIBRATED_MODEL,
)
from polylayers.features.panel import LAYER_STAGES
from polylayers.reporting.figures import HORIZON_ORDER, MODEL_LABELS, STAGE_LABELS

BOOTSTRAP_RESAMPLES = 2_000


def _label(model: str, stage: str) -> str:
    if stage == "-":
        return MODEL_LABELS[model]
    return f"{MODEL_LABELS[model]} · {STAGE_LABELS[stage]}"


def _interval(result: dict | pd.Series) -> str:
    return f"{result['difference']:+.4f} [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]"


def scores_markdown(scores: pd.DataFrame, models: list[tuple[str, str]]) -> str:
    """Brier score and top-1 accuracy per horizon for the listed (model, stage) pairs."""
    header = "| Forecaster | " + " | ".join(f"{h}d Brier | {h}d top-1" for h in HORIZON_ORDER)
    lines = [header + " |", "|---|" + "---:|---:|" * len(HORIZON_ORDER)]
    for model, stage in models:
        rows = scores[(scores["model"] == model) & (scores["stage"] == stage)]
        rows = rows.set_index("horizon_days").reindex(HORIZON_ORDER)
        cells = [f"{row.brier:.3f} | {row.top1_accuracy:.0%}" for row in rows.itertuples()]
        lines.append(f"| {_label(model, stage)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def differences_markdown(
    differences: pd.DataFrame, reference: str, models: list[tuple[str, str]]
) -> str:
    """Brier difference vs. `reference` with 95% interval, per horizon (negative = better)."""
    brier = differences[
        (differences["metric"] == "brier") & (differences["reference"] == reference)
    ]
    header = "| Forecaster | " + " | ".join(f"{h}d" for h in HORIZON_ORDER) + " |"
    lines = [header, "|---|" + "---:|" * len(HORIZON_ORDER)]
    for model, stage in models:
        rows = brier[(brier["model"] == model) & (brier["stage"] == stage)]
        rows = rows.set_index("horizon_days").reindex(HORIZON_ORDER)
        cells = [_interval(row) for _, row in rows.iterrows()]
        lines.append(f"| {_label(model, stage)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def temperatures_markdown(temperatures: pd.DataFrame, differences: pd.DataFrame) -> str:
    """Median fitted temperature per horizon and the recalibration's gain over the raw price."""
    medians = temperatures.groupby("horizon_days")["beta"].median().reindex(HORIZON_ORDER)
    gains = (
        differences[
            (differences["metric"] == "brier")
            & (differences["reference"] == MARKET_MODEL)
            & (differences["model"] == RECALIBRATED_MODEL)
        ]
        .set_index("horizon_days")
        .reindex(HORIZON_ORDER)
    )
    header = "| | " + " | ".join(f"{h}d" for h in HORIZON_ORDER) + " |"
    lines = [header, "|---|" + "---:|" * len(HORIZON_ORDER)]
    lines.append(
        "| Median fitted temperature (beta) | " + " | ".join(f"{b:.2f}" for b in medians) + " |"
    )
    lines.append(
        "| Recalibrated minus raw market, Brier | "
        + " | ".join(_interval(row) for _, row in gains.iterrows())
        + " |"
    )
    return "\n".join(lines)


def fundamentals_markdown(predictions: pd.DataFrame, models: list[tuple[str, str]]) -> str:
    """Fundamentals-only models vs. the bucket base rate: Brier difference with 95% interval."""
    header = (
        "| Forecaster vs. bucket base rate | " + " | ".join(f"{h}d" for h in HORIZON_ORDER) + " |"
    )
    lines = [header, "|---|" + "---:|" * len(HORIZON_ORDER)]
    for model, stage in models:
        cells = []
        for horizon in HORIZON_ORDER:
            at_horizon = predictions[predictions["horizon_days"] == horizon]
            candidate = at_horizon[(at_horizon["model"] == model) & (at_horizon["stage"] == stage)]
            base = at_horizon[at_horizon["model"] == "bucket_base_rate"]
            cells.append(
                _interval(paired_meeting_bootstrap(candidate, base, "brier", BOOTSTRAP_RESAMPLES))
            )
        lines.append(f"| {_label(model, stage)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def favorite_hold_share(predictions: pd.DataFrame) -> float:
    """Share of meetings where the forecaster's favorite bucket is 'hold'."""
    favorites = predictions.loc[predictions.groupby("meeting_id")["probability"].idxmax(), "bucket"]
    return float((favorites == "hold").mean())


def gcn_mechanism_markdown(predictions: pd.DataFrame, scores: pd.DataFrame) -> str:
    """GCN on subsets of the graph: Brier, top-1, and how often its favorite is 'hold'."""
    models = ["gcn", *GCN_ABLATIONS, "graphconv"]
    header = "| Model (market price features) | " + " | ".join(
        f"{h}d Brier · top-1 · favorite=hold" for h in HORIZON_ORDER
    )
    lines = [header + " |", "|---|" + "---:|" * len(HORIZON_ORDER)]
    for model in models:
        cells = []
        for horizon in HORIZON_ORDER:
            score = scores[
                (scores["model"] == model)
                & (scores["stage"] == "market")
                & (scores["horizon_days"] == horizon)
            ].iloc[0]
            rows = predictions[
                (predictions["model"] == model)
                & (predictions["stage"] == "market")
                & (predictions["horizon_days"] == horizon)
            ]
            hold = favorite_hold_share(rows)
            cells.append(f"{score.brier:.3f} · {score.top1_accuracy:.0%} · {hold:.0%}")
        lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summary_markdown(
    scores: pd.DataFrame,
    differences: pd.DataFrame,
    predictions: pd.DataFrame,
    temperatures: pd.DataFrame,
) -> str:
    """Every results table for the README and report, as one markdown block."""
    references = [
        ("uniform", "-"),
        ("bucket_base_rate", "-"),
        ("market_raw", "-"),
        (MARKET_MODEL, "-"),
        (RECALIBRATED_MODEL, "-"),
    ]
    layered = [(model, stage) for model in ("logistic", "clogit") for stage in LAYER_STAGES]
    graphs = [(model, stage) for model in ("gcn", "graphconv") for stage in ("market", FULL_STAGE)]
    fundamentals = [
        (model, stage) for model in ("logistic", "clogit") for stage in ("macro", "macro+news+text")
    ]
    return "\n\n".join(
        [
            "**How good is the market?** Brier score (lower is better) and the share of test "
            "meetings where the forecaster's favorite bucket won.",
            scores_markdown(scores, references),
            "**Is the market calibrated?** The temperature (beta) that best recalibrates the "
            "normalized price, fit walk-forward on past meetings (beta > 1: the market was "
            "underconfident), and what the recalibration gains over the raw price.",
            temperatures_markdown(temperatures, differences),
            "**Does any layer add information?** Brier difference vs. the recalibrated market "
            "price, with a 95% interval from a meeting-level paired bootstrap (negative = "
            "better than the recalibrated market).",
            differences_markdown(differences, RECALIBRATED_MODEL, layered + graphs),
            "**Without the market's price.** Fundamentals-only models vs. the bucket base rate "
            "(how often each kind of move has won before): Brier difference with 95% interval.",
            fundamentals_markdown(predictions, fundamentals),
            "**What breaks GCN?** GCN on subsets of the graph, with market price features.",
            gcn_mechanism_markdown(predictions, scores),
        ]
    )
