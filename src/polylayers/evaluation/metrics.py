"""Proper scoring rules and meeting-level accuracy for probability forecasts."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LOG_FLOOR = 1e-6


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error of the probabilities (lower is better)."""
    return float(np.mean((probabilities - labels) ** 2))


def log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean negative log likelihood (lower is better)."""
    clipped = np.clip(probabilities, LOG_FLOOR, 1 - LOG_FLOOR)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """ROC AUC, or NaN if only one class is present."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def top1_accuracy(predictions: pd.DataFrame) -> float:
    """Share of meetings whose highest-probability bucket is the one that resolved YES.

    Meetings whose winning bucket was not yet listed at the forecast time are skipped.
    """
    hits = []
    for _, meeting in predictions.groupby("meeting_id"):
        if meeting["label"].max() != 1:
            continue
        hits.append(meeting.loc[meeting["probability"].idxmax(), "label"] == 1)
    return float(np.mean(hits)) if hits else float("nan")


def score_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Metrics for every (horizon, model, stage) group of a predictions table."""
    rows = []
    keys = ["horizon_days", "model", "stage"]
    for (horizon, model, stage), group in predictions.groupby(keys, sort=False):
        labels = group["label"].to_numpy(dtype=float)
        probabilities = group["probability"].to_numpy(dtype=float)
        rows.append(
            {
                "horizon_days": horizon,
                "model": model,
                "stage": stage,
                "markets": len(group),
                "meetings": group["meeting_id"].nunique(),
                "brier": brier_score(labels, probabilities),
                "log_loss": log_loss(labels, probabilities),
                "auc": auc(labels, probabilities),
                "top1_accuracy": top1_accuracy(group),
            }
        )
    return pd.DataFrame(rows)
