"""Paired bootstrap over meetings for the difference in a loss between two forecasters.

Markets within a meeting are not independent (exactly one resolves YES), so the meeting is
the resampling unit. Both forecasters are scored on the same resampled meetings.
"""

import numpy as np
import pandas as pd

from polylayers.evaluation.metrics import LOG_FLOOR


def market_losses(labels: np.ndarray, probabilities: np.ndarray, metric: str) -> np.ndarray:
    """Per-market loss for 'brier' or 'log_loss'."""
    if metric == "brier":
        return (probabilities - labels) ** 2
    if metric == "log_loss":
        clipped = np.clip(probabilities, LOG_FLOOR, 1 - LOG_FLOOR)
        return -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    raise ValueError(f"unknown metric {metric!r}")


def paired_meeting_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    metric: str,
    n_resamples: int = 2_000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean per-market loss difference (candidate - reference) with a 95% interval.

    Negative differences favor the candidate. `p_candidate_better` is the share of
    resamples in which the candidate's loss is lower.
    """
    keys = ["meeting_id", "market_id"]
    joined = candidate[[*keys, "label", "probability"]].merge(
        reference[[*keys, "probability"]], on=keys, suffixes=("", "_reference"), validate="1:1"
    )
    labels = joined["label"].to_numpy(dtype=float)
    joined["difference"] = market_losses(
        labels, joined["probability"].to_numpy(), metric
    ) - market_losses(labels, joined["probability_reference"].to_numpy(), metric)
    per_meeting = joined.groupby("meeting_id")["difference"].agg(["sum", "count"])
    sums, counts = per_meeting["sum"].to_numpy(), per_meeting["count"].to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(per_meeting), size=(n_resamples, len(per_meeting)))
    resampled = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {
        "difference": float(sums.sum() / counts.sum()),
        "ci_low": float(np.quantile(resampled, 0.025)),
        "ci_high": float(np.quantile(resampled, 0.975)),
        "p_candidate_better": float((resampled < 0).mean()),
        "meetings": int(len(per_meeting)),
    }
