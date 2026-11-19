"""Run every model over every walk-forward fold and summarize against the market.

Models:
- uniform:             1 / (buckets open), the no-information reference
- bucket_base_rate:    how often each kind of move has won before (the naive baseline)
- market_raw:          the YES price at the forecast time
- market_norm:         the YES price normalized within the meeting
- market_recalibrated: the normalized price, temperature-scaled on past meetings; layers
                       must beat this to claim information rather than calibration
- logistic / clogit:   logistic regression and conditional logit on a layer stage
- gcn / graphconv:     graph networks on the market graph built at the forecast time
- gcn_no_edges, gcn_cross_meeting, gcn_rivals_traders: GCN on subsets of the graph, to
                       locate what makes the full-graph GCN collapse
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import torch

from polylayers.evaluation.bootstrap import paired_meeting_bootstrap
from polylayers.evaluation.metrics import score_table
from polylayers.evaluation.walk_forward import walk_forward_folds
from polylayers.features.panel import FUNDAMENTAL_STAGES, LAYER_STAGES
from polylayers.models import baselines
from polylayers.models.conditional_logit import fit_predict_conditional_logit
from polylayers.models.gnn import fit_predict_gnn
from polylayers.models.logistic import fit_predict_logistic
from polylayers.models.recalibration import fit_predict_recalibrated_market

logger = logging.getLogger(__name__)

MARKET_MODEL = "market_norm"
RECALIBRATED_MODEL = "market_recalibrated"
REFERENCE_MODELS = (MARKET_MODEL, RECALIBRATED_MODEL)
NO_STAGE = "-"
STAGES = {**LAYER_STAGES, **FUNDAMENTAL_STAGES}
FULL_STAGE = list(LAYER_STAGES)[-1]
GNN_STAGES = ("market", FULL_STAGE)
GCN_ABLATIONS = {
    "gcn_no_edges": (),
    "gcn_cross_meeting": ("temporal", "same_bucket"),
    "gcn_rivals_traders": ("same_meeting", "trader_overlap"),
}

Predictor = Callable[[pd.DataFrame, pd.DataFrame, tuple[str, ...], pd.DataFrame], np.ndarray]

PREDICTORS: dict[str, Predictor] = {
    "uniform": lambda train, test, groups, wallets: baselines.uniform_within_meeting(test),
    "bucket_base_rate": lambda train, test, groups, wallets: baselines.bucket_base_rate(
        train, test
    ),
    "market_raw": lambda train, test, groups, wallets: baselines.market_price(test),
    "market_norm": lambda train, test, groups, wallets: baselines.market_price_normalized(test),
    "market_recalibrated": lambda train, test, groups, wallets: fit_predict_recalibrated_market(
        train, test
    ),
    "logistic": lambda train, test, groups, wallets: fit_predict_logistic(train, test, groups),
    "clogit": lambda train, test, groups, wallets: fit_predict_conditional_logit(
        train, test, groups
    ),
    "gcn": partial(fit_predict_gnn, "gcn"),
    "graphconv": partial(fit_predict_gnn, "graphconv"),
    **{
        name: partial(fit_predict_gnn, "gcn", edge_types=edge_types)
        for name, edge_types in GCN_ABLATIONS.items()
    },
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
    specs = [
        ModelSpec("uniform"),
        ModelSpec("bucket_base_rate"),
        ModelSpec("market_raw"),
        ModelSpec(MARKET_MODEL),
        ModelSpec(RECALIBRATED_MODEL),
    ]
    for stage in [*FUNDAMENTAL_STAGES, *LAYER_STAGES]:
        specs += [ModelSpec("logistic", stage), ModelSpec("clogit", stage)]
    for stage in GNN_STAGES:
        specs += [ModelSpec("gcn", stage), ModelSpec("graphconv", stage)]
    specs += [ModelSpec(name, "market") for name in GCN_ABLATIONS]
    return specs


def run_walk_forward(
    panel: pd.DataFrame,
    wallets: pd.DataFrame,
    specs: list[ModelSpec],
    min_train_meetings: int,
) -> pd.DataFrame:
    """Out-of-sample predictions for every (fold, model, stage)."""
    # The graphs have ~100 nodes; thread start-up costs more than parallelism saves.
    torch.set_num_threads(1)
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
            elapsed = time.time() - started
            logger.info("Fold %d/%d done (%.0fs elapsed)", number, len(folds), elapsed)
    return pd.concat(records, ignore_index=True)


def compare_to_reference(
    predictions: pd.DataFrame, reference_model: str, n_resamples: int
) -> pd.DataFrame:
    """Brier and log-loss differences vs. one reference forecaster, with bootstrap CIs."""
    rows = []
    for horizon, by_horizon in predictions.groupby("horizon_days"):
        reference = by_horizon[by_horizon["model"] == reference_model]
        for (model, stage), candidate in by_horizon.groupby(["model", "stage"], sort=False):
            if model == reference_model:
                continue
            for metric in ("brier", "log_loss"):
                result = paired_meeting_bootstrap(candidate, reference, metric, n_resamples)
                rows.append(
                    {
                        "reference": reference_model,
                        "horizon_days": horizon,
                        "model": model,
                        "stage": stage,
                        "metric": metric,
                    }
                    | result
                )
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame, n_resamples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(scores per horizon/model/stage, differences vs. both market references)."""
    differences = pd.concat(
        [compare_to_reference(predictions, model, n_resamples) for model in REFERENCE_MODELS],
        ignore_index=True,
    )
    return score_table(predictions), differences
