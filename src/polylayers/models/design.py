"""Design matrices for a layer stage, with preprocessing fit on training rows only.

Market-level columns enter as they are. Meeting-level columns (macro, news, text) are the
same for every market in a meeting, so on their own they cannot favor one outcome over
another; they enter only through interactions with the bucket: (feature x signed move in
25 bp units) and (feature x hold flag). A positive weight on `implied_3m_bps_x_move` then
means "when bills trade above the policy rate, hike buckets gain and cut buckets lose".
"""

import logging

import numpy as np
import pandas as pd

from polylayers.features.panel import FEATURE_GROUPS, MEETING_LEVEL_GROUPS

logger = logging.getLogger(__name__)

MOVE_UNIT_BPS = 25.0


def stage_columns(groups: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """(market-level columns, meeting-level columns) for the feature groups of a stage."""
    market_level = [c for g in groups if g not in MEETING_LEVEL_GROUPS for c in FEATURE_GROUPS[g]]
    meeting_level = [c for g in groups if g in MEETING_LEVEL_GROUPS for c in FEATURE_GROUPS[g]]
    return market_level, meeting_level


class StageDesign:
    """Select a stage's columns, impute, add bucket interactions, and standardize.

    Every statistic (which columns exist, medians, means, scales) comes from `fit` rows, so
    test rows never influence their own preprocessing.
    """

    def __init__(self, groups: tuple[str, ...]) -> None:
        self.market_columns, self.meeting_columns = stage_columns(groups)

    def fit(self, rows: pd.DataFrame) -> "StageDesign":
        raw = rows[self.market_columns + self.meeting_columns]
        self.kept_columns_ = [column for column in raw.columns if raw[column].notna().any()]
        dropped = sorted(set(raw.columns) - set(self.kept_columns_))
        if dropped:
            logger.debug("Dropping columns with no training data: %s", dropped)
        self.medians_ = raw[self.kept_columns_].median()
        design = self._unscaled(rows)
        self.means_ = design.mean()
        self.scales_ = design.std(ddof=0).replace(0.0, 1.0)
        self.feature_names_ = list(design.columns)
        return self

    def transform(self, rows: pd.DataFrame) -> np.ndarray:
        design = self._unscaled(rows)
        return ((design - self.means_) / self.scales_).to_numpy(dtype=np.float64)

    def fit_transform(self, rows: pd.DataFrame) -> np.ndarray:
        return self.fit(rows).transform(rows)

    def _unscaled(self, rows: pd.DataFrame) -> pd.DataFrame:
        raw = rows[self.kept_columns_].fillna(self.medians_)
        market = raw[[column for column in self.market_columns if column in self.kept_columns_]]
        move = rows["bucket_move_bps"].fillna(0.0) / MOVE_UNIT_BPS
        hold = rows["bucket_is_hold"]
        interactions = {}
        for column in self.meeting_columns:
            if column in self.kept_columns_:
                interactions[f"{column}_x_move"] = raw[column] * move
                interactions[f"{column}_x_hold"] = raw[column] * hold
        return pd.concat([market, pd.DataFrame(interactions, index=rows.index)], axis=1)
