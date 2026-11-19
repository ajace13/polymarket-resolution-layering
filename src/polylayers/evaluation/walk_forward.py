"""Walk-forward folds by FOMC meeting.

For a test meeting at horizon h, the forecast is made on (decision date - h). Training uses
every earlier meeting whose decision was announced before that forecast date, so all
training labels were public when the forecast is made. Each meeting is tested once per
horizon, after a burn-in of `min_train_meetings`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """Row positions (into the panel) for one test meeting at one horizon."""

    horizon_days: int
    test_meeting: str
    train_index: np.ndarray
    test_index: np.ndarray


def walk_forward_folds(panel: pd.DataFrame, min_train_meetings: int) -> list[Fold]:
    """Folds for every horizon and every meeting after the burn-in, oldest first."""
    folds = []
    for horizon, rows in panel.groupby("horizon_days"):
        decisions = rows.groupby("meeting_id")["decision_date"].first().sort_values()
        for position, (meeting_id, decision_date) in enumerate(decisions.items()):
            if position < min_train_meetings:
                continue
            forecast_date = decision_date - pd.Timedelta(days=int(horizon))
            known = decisions[decisions < forecast_date].index
            train_index = rows.index[rows["meeting_id"].isin(known)].to_numpy()
            test_index = rows.index[rows["meeting_id"] == meeting_id].to_numpy()
            folds.append(Fold(int(horizon), meeting_id, train_index, test_index))
    return folds
