"""Run every model over every walk-forward fold and summarize against the market.

Models:
- uniform:      1 / (buckets open), the no-information reference
- market_raw:   the YES price at the forecast time
- market_norm:  the YES price normalized within the meeting (the baseline to beat)
- logistic:     per-market logistic regression on a layer stage
- clogit:       conditional logit (softmax over each meeting's buckets) on a layer stage
- gcn / graphconv: graph networks on the market graph built at the forecast time
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from polylayers.evaluation.bootstrap import paired_meeting_bootstrap
from polylayers.evaluation.metrics import score_table
from polylayers.evaluation.walk_forward import walk_forward_folds
from polylayers.features.panel import FUNDAMENTAL_STAGES, LAYER_STAGES
from polylayers.models import baselines
from polylayers.models.conditional_logit import fit_predict_conditional_logit
from polylayers.models.gnn import fit_predict_gnn
from polylayers.models.logistic import fit_predict_logistic

logger = logging.getLogger(__name__)

REFERENCE_MODEL = "market_norm"
NO_STAGE = "-"
STAGES = {**LAYER_STAGES, **FUNDAMENTAL_STAGES}
GNN_STAGES = ("market", "market+macro+news+text")

Predictor = Callable[[pd.DataFrame, pd.DataFrame, tuple[str, ...], pd.DataFrame], np.ndarray]

PREDICTORS: dict[str, Predictor] = {
    "uniform": lambda train, test, groups, wallets: baselines.uniform_within_meeting(test),
    "market_raw": lambda train, test, groups, wallets: baselines.market_price(test),
    "market_norm": lambda train, test, groups, wallets: baselines.market_price_normalized(test),
    "logistic": lambda train, test, groups, wallets: fit_predict_logistic(train, test, groups),
    "clogit": lambda train, test, groups, wallets: fit_predict_conditional_logit(
        train, test, groups
    ),
    "gcn": lambda train, test, groups, wallets: fit_predict_gnn(
        "gcn", train, test, groups, wallets
    ),
    "graphconv": lambda train, test, groups, wallets: fit_predict_gnn(
        "graphconv", train, test, groups, wallets
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    """A model and the layer stage whose features it uses."""

    model: str
    stage: str = NO_STAGE

    @property
    def groups(self) -> tuple[str, ...]:
        return STAGES.get(self.stage, ())


def default_model_specs() -> list[ModelSpec]:
    specs = [ModelSpec("uniform"), ModelSpec("market_raw"), ModelSpec(REFERENCE_MODEL)]
    for stage in [*FUNDAMENTAL_STAGES, *LAYER_STAGES]:
        specs += [ModelSpec("logistic", stage), ModelSpec("clogit", stage)]
    for stage in GNN_STAGES:
        specs += [ModelSpec("gcn", stage), ModelSpec("graphconv", stage)]
    return specs


def run_walk_forward(
    panel: pd.DataFrame,
    wallets: pd.DataFrame,
    specs: list[ModelSpec],
    min_train_meetings: int,
) -> pd.DataFrame:
    """Out-of-sample predictions for every (fold, model, stage)."""
    folds = walk_forward_folds(panel, min_train_meetings)
    logger.info(
        "Walk-forward: %d folds (%d horizons), %d model specs",
        len(folds),
        panel["horizon_days"].nunique(),
        len(specs),
    )
    records = []
    started = time.time()
    for number, fold in enumerate(folds, start=1):
        train_rows = panel.loc[fold.train_index]
        test_rows = panel.loc[fold.test_index]
        for spec in specs:
            probabilities = PREDICTORS[spec.model](train_rows, test_rows, spec.groups, wallets)
            records.append(
                test_rows[["horizon_days", "meeting_id", "market_id", "bucket", "label"]].assign(
                    model=spec.model, stage=spec.stage, probability=probabilities
                )
            )
        if number % 10 == 0 or number == len(folds):
            logger.info(
                "Fold %d/%d done (%.0fs elapsed)", number, len(folds), time.time() - started
            )
    return pd.concat(records, ignore_index=True)


def compare_to_reference(predictions: pd.DataFrame, n_resamples: int) -> pd.DataFrame:
    """Brier and log-loss differences vs. the normalized market price, with bootstrap CIs."""
    rows = []
    for horizon, by_horizon in predictions.groupby("horizon_days"):
        reference = by_horizon[by_horizon["model"] == REFERENCE_MODEL]
        for (model, stage), candidate in by_horizon.groupby(["model", "stage"], sort=False):
            if model == REFERENCE_MODEL:
                continue
            for metric in ("brier", "log_loss"):
                result = paired_meeting_bootstrap(candidate, reference, metric, n_resamples)
                rows.append(
                    {"horizon_days": horizon, "model": model, "stage": stage, "metric": metric}
                    | result
                )
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame, n_resamples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(scores per horizon/model/stage, differences vs. the market with intervals)."""
    return score_table(predictions), compare_to_reference(predictions, n_resamples)
