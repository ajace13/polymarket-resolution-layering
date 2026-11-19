"""Logistic regression per market, with the L2 strength chosen on the latest training meetings.

No class weighting: the evaluation scores probabilities (Brier, log loss), and reweighting
classes would deliberately miscalibrate them.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from polylayers.models.baselines import clip_probability
from polylayers.models.design import StageDesign

C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
INNER_VALIDATION_MEETINGS = 4


def inner_split(train_rows: pd.DataFrame, n_validation: int) -> tuple[pd.Index, pd.Index]:
    """Earlier training meetings for fitting, the latest `n_validation` for validation."""
    order = train_rows.groupby("meeting_id")["decision_date"].first().sort_values()
    validation_meetings = set(order.index[-n_validation:])
    is_validation = train_rows["meeting_id"].isin(validation_meetings)
    return train_rows.index[~is_validation], train_rows.index[is_validation]


def _fit(rows: pd.DataFrame, groups: tuple[str, ...], c_value: float):
    design = StageDesign(groups).fit(rows)
    model = LogisticRegression(C=c_value, max_iter=5_000)
    model.fit(design.transform(rows), rows["label"].to_numpy())
    return design, model


def choose_c(train_rows: pd.DataFrame, groups: tuple[str, ...]) -> float:
    """C with the lowest log loss on the latest training meetings (smaller C wins ties)."""
    fit_index, validation_index = inner_split(train_rows, INNER_VALIDATION_MEETINGS)
    fit_rows, validation_rows = train_rows.loc[fit_index], train_rows.loc[validation_index]
    best_c, best_loss = C_GRID[0], np.inf
    for c_value in C_GRID:
        design, model = _fit(fit_rows, groups, c_value)
        probabilities = model.predict_proba(design.transform(validation_rows))[:, 1]
        loss = log_loss(validation_rows["label"], probabilities, labels=[0, 1])
        if loss < best_loss - 1e-9:
            best_c, best_loss = c_value, loss
    return best_c


def fit_predict_logistic(
    train_rows: pd.DataFrame, test_rows: pd.DataFrame, groups: tuple[str, ...]
) -> np.ndarray:
    """P(YES) for test rows from a logistic regression refit on all training rows."""
    c_value = choose_c(train_rows, groups)
    design, model = _fit(train_rows, groups, c_value)
    return clip_probability(model.predict_proba(design.transform(test_rows))[:, 1])
