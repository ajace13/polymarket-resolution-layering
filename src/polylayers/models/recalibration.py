"""Temperature-scaled market price: the market's own forecast, recalibrated.

Within each meeting, P(bucket wins) = softmax(beta * logit(normalized price)). A single
parameter beta is learned on the training meetings: beta > 1 sharpens the market's
probabilities (it was underconfident), beta < 1 flattens them (overconfident), and beta = 1
leaves them unchanged. Any information layer has to beat this, not just the raw price,
before it can claim to add information rather than calibration.
"""

import numpy as np
import pandas as pd

from polylayers.evaluation.walk_forward import walk_forward_folds
from polylayers.models.baselines import clip_probability
from polylayers.models.conditional_logit import ConditionalLogit

FEATURE = "mkt_logit_norm"
# A single parameter cannot overfit much, so the penalty is only a token stabilizer.
L2_PENALTY = 1e-4


def fit_temperature(train_rows: pd.DataFrame) -> ConditionalLogit:
    """Conditional logit on the market's log-odds alone; its single weight is beta."""
    return ConditionalLogit(L2_PENALTY).fit(
        train_rows[[FEATURE]].to_numpy(dtype=float),
        train_rows["meeting_id"].to_numpy(),
        train_rows["label"].to_numpy(),
    )


def fit_predict_recalibrated_market(
    train_rows: pd.DataFrame, test_rows: pd.DataFrame
) -> np.ndarray:
    """P(YES) for test rows from the temperature-scaled market price."""
    model = fit_temperature(train_rows)
    probabilities = model.predict(
        test_rows[[FEATURE]].to_numpy(dtype=float), test_rows["meeting_id"].to_numpy()
    )
    return clip_probability(probabilities)


def temperatures_by_fold(panel: pd.DataFrame, min_train_meetings: int) -> pd.DataFrame:
    """The fitted beta for every walk-forward fold (how the market's calibration drifts)."""
    rows = []
    for fold in walk_forward_folds(panel, min_train_meetings):
        model = fit_temperature(panel.loc[fold.train_index])
        rows.append(
            {
                "horizon_days": fold.horizon_days,
                "test_meeting": fold.test_meeting,
                "beta": float(model.weights_.detach()[0]),
            }
        )
    return pd.DataFrame(rows)
