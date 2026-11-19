"""Markdown summary tables generated from the evaluation results."""

import pandas as pd

from polylayers.evaluation.experiment import REFERENCE_MODEL
from polylayers.reporting.figures import HORIZON_ORDER, MODEL_LABELS, STAGE_LABELS


def _label(model: str, stage: str) -> str:
    if stage == "-":
        return MODEL_LABELS[model]
    return f"{MODEL_LABELS[model]}, {STAGE_LABELS[stage].lower()}"


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


def differences_markdown(differences: pd.DataFrame, models: list[tuple[str, str]]) -> str:
    """Brier difference vs. the market with 95% interval, per horizon (negative = better)."""
    brier = differences[differences["metric"] == "brier"]
    header = "| Forecaster | " + " | ".join(f"{h}d" for h in HORIZON_ORDER) + " |"
    lines = [header, "|---|" + "---:|" * len(HORIZON_ORDER)]
    for model, stage in models:
        rows = brier[(brier["model"] == model) & (brier["stage"] == stage)]
        rows = rows.set_index("horizon_days").reindex(HORIZON_ORDER)
        cells = [
            f"{row.difference:+.4f} [{row.ci_low:+.4f}, {row.ci_high:+.4f}]"
            for row in rows.itertuples()
        ]
        lines.append(f"| {_label(model, stage)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summary_markdown(scores: pd.DataFrame, differences: pd.DataFrame) -> str:
    """The README results block: market strength, then what each layer adds."""
    reference = [("uniform", "-"), ("market_raw", "-"), (REFERENCE_MODEL, "-")]
    layered = [
        (model, stage)
        for model in ("logistic", "clogit")
        for stage in ("market", "market+macro", "market+macro+news", "market+macro+news+text")
    ]
    fundamentals = [(model, "macro+news+text") for model in ("logistic", "clogit")]
    graphs = [
        (model, stage)
        for model in ("gcn", "graphconv")
        for stage in ("market", "market+macro+news+text")
    ]
    return "\n\n".join(
        [
            "**How good is the market?** Brier score (lower is better) and the share of meetings "
            "where the forecaster's favorite bucket won.",
            scores_markdown(scores, reference + fundamentals),
            "**Does anything improve on it?** Brier difference vs. the normalized market price, "
            "with a 95% interval from a meeting-level paired bootstrap (negative = better than "
            "the market).",
            differences_markdown(differences, layered + graphs),
        ]
    )
