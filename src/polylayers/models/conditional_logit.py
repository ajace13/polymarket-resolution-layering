"""Conditional logit: one softmax over each meeting's buckets (McFadden, 1974).

Exactly one bucket per meeting resolves YES, so the likelihood of a meeting is the softmax
probability of its winning bucket among the buckets open at the forecast time. Mutual
exclusivity is built into the model instead of being something it has to learn, and the
predicted probabilities of a meeting's buckets always sum to one.
"""

import logging

import numpy as np
import pandas as pd
import torch

from polylayers.models.baselines import clip_probability
from polylayers.models.design import StageDesign
from polylayers.models.logistic import inner_split

logger = logging.getLogger(__name__)

L2_GRID = (0.001, 0.01, 0.1)
INNER_VALIDATION_MEETINGS = 4


def group_log_softmax(scores: torch.Tensor, groups: torch.Tensor, n_groups: int) -> torch.Tensor:
    """log softmax of `scores` within each group (numerically stable)."""
    group_max = torch.full((n_groups,), -torch.inf, dtype=scores.dtype)
    group_max = group_max.scatter_reduce(0, groups, scores, reduce="amax", include_self=True)
    shifted = scores - group_max[groups].detach()
    totals = torch.zeros(n_groups, dtype=scores.dtype).scatter_add(0, groups, shifted.exp())
    return shifted - totals.log()[groups]


class ConditionalLogit:
    """Linear scores softmax-normalized within meetings, with an L2 penalty."""

    def __init__(self, l2: float, max_iterations: int = 200) -> None:
        self.l2 = l2
        self.max_iterations = max_iterations

    def fit(self, features: np.ndarray, meetings: np.ndarray, labels: np.ndarray):
        """Fit on meetings whose winning bucket is among the rows (others carry no signal)."""
        codes, uniques = pd.factorize(meetings)
        has_winner = pd.Series(labels).groupby(codes).transform("max").to_numpy() == 1
        if not has_winner.all():
            logger.debug(
                "Conditional logit: skipping %d rows from meetings without a winner",
                int((~has_winner).sum()),
            )
        codes, _ = pd.factorize(meetings[has_winner])
        x = torch.tensor(features[has_winner], dtype=torch.float64)
        y = torch.tensor(labels[has_winner] == 1)
        groups = torch.tensor(codes, dtype=torch.long)
        n_groups = int(codes.max()) + 1
        self.weights_ = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [self.weights_], max_iter=self.max_iterations, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            log_probabilities = group_log_softmax(x @ self.weights_, groups, n_groups)
            loss = -log_probabilities[y].sum() / n_groups + self.l2 * self.weights_.pow(2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def predict(self, features: np.ndarray, meetings: np.ndarray) -> np.ndarray:
        codes, uniques = pd.factorize(meetings)
        with torch.no_grad():
            scores = torch.tensor(features, dtype=torch.float64) @ self.weights_
            log_probabilities = group_log_softmax(
                scores, torch.tensor(codes, dtype=torch.long), len(uniques)
            )
        return log_probabilities.exp().numpy()


def _fit(rows: pd.DataFrame, groups: tuple[str, ...], l2: float):
    design = StageDesign(groups).fit(rows)
    model = ConditionalLogit(l2).fit(
        design.transform(rows), rows["meeting_id"].to_numpy(), rows["label"].to_numpy()
    )
    return design, model


def meeting_log_loss(probabilities: np.ndarray, rows: pd.DataFrame) -> float:
    """Mean negative log probability of each meeting's winning bucket."""
    winners = rows["label"].to_numpy() == 1
    return float(-np.log(clip_probability(probabilities[winners])).mean())


def choose_l2(train_rows: pd.DataFrame, groups: tuple[str, ...]) -> float:
    fit_index, validation_index = inner_split(train_rows, INNER_VALIDATION_MEETINGS)
    fit_rows, validation_rows = train_rows.loc[fit_index], train_rows.loc[validation_index]
    best_l2, best_loss = L2_GRID[-1], np.inf
    for l2 in sorted(L2_GRID, reverse=True):
        design, model = _fit(fit_rows, groups, l2)
        probabilities = model.predict(
            design.transform(validation_rows), validation_rows["meeting_id"].to_numpy()
        )
        loss = meeting_log_loss(probabilities, validation_rows)
        if loss < best_loss - 1e-9:
            best_l2, best_loss = l2, loss
    return best_l2


def fit_predict_conditional_logit(
    train_rows: pd.DataFrame, test_rows: pd.DataFrame, groups: tuple[str, ...]
) -> np.ndarray:
    """P(YES) for test rows; the probabilities of each test meeting's buckets sum to one."""
    l2 = choose_l2(train_rows, groups)
    design, model = _fit(train_rows, groups, l2)
    probabilities = model.predict(design.transform(test_rows), test_rows["meeting_id"].to_numpy())
    return clip_probability(probabilities)
