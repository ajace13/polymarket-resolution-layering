"""Reference forecasts that need no training."""

import numpy as np
import pandas as pd

PROBABILITY_FLOOR = 0.005


def clip_probability(values: np.ndarray) -> np.ndarray:
    """Keep probabilities away from 0 and 1 so log loss stays finite."""
    return np.clip(values, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)


def uniform_within_meeting(test_rows: pd.DataFrame) -> np.ndarray:
    """1 / (number of markets open for the meeting): no information beyond the menu of buckets."""
    group_sizes = test_rows.groupby("meeting_id")["market_id"].transform("count")
    return clip_probability(1.0 / group_sizes.to_numpy(dtype=float))


def market_price(test_rows: pd.DataFrame) -> np.ndarray:
    """The market's own YES price at the forecast time."""
    return clip_probability(test_rows["mkt_price"].to_numpy(dtype=float))


def market_price_normalized(test_rows: pd.DataFrame) -> np.ndarray:
    """YES price divided by the sum of prices across the meeting's buckets (sums to 1)."""
    return clip_probability(test_rows["mkt_price_norm"].to_numpy(dtype=float))
