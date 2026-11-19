"""Reference forecasts: no training, or (bucket base rate) a frequency count."""

import numpy as np
import pandas as pd

PROBABILITY_FLOOR = 0.005
# Additive smoothing for bucket win rates: half a win per bucket group, one appearance.
BASE_RATE_PRIOR_WINS = 0.5
BASE_RATE_PRIOR_COUNT = 1.0


def clip_probability(values: np.ndarray) -> np.ndarray:
    """Keep probabilities away from 0 and 1 so log loss stays finite."""
    return np.clip(values, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)


def normalize_within_meeting(values: np.ndarray, meetings: pd.Series) -> np.ndarray:
    """Divide each value by its meeting's total, so a meeting's buckets sum to one."""
    totals = pd.Series(values, index=meetings.index).groupby(meetings).transform("sum")
    return values / totals.to_numpy()


def uniform_within_meeting(test_rows: pd.DataFrame) -> np.ndarray:
    """1 / (number of markets open for the meeting): no information beyond the menu of buckets."""
    group_sizes = test_rows.groupby("meeting_id")["market_id"].transform("count")
    return clip_probability(1.0 / group_sizes.to_numpy(dtype=float))


def bucket_base_rate(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
    """How often each kind of move (hold, small cut, ...) won in training, normalized per meeting.

    The honest naive baseline for this period: it learns that the Fed usually holds.
    """
    counts = train_rows.groupby("bucket_group")["label"].agg(["sum", "count"])
    win_rates = (counts["sum"] + BASE_RATE_PRIOR_WINS) / (counts["count"] + BASE_RATE_PRIOR_COUNT)
    prior = BASE_RATE_PRIOR_WINS / BASE_RATE_PRIOR_COUNT
    raw = test_rows["bucket_group"].map(win_rates).fillna(prior).to_numpy(dtype=float)
    return clip_probability(normalize_within_meeting(raw, test_rows["meeting_id"]))


def market_price(test_rows: pd.DataFrame) -> np.ndarray:
    """The market's own YES price at the forecast time."""
    return clip_probability(test_rows["mkt_price"].to_numpy(dtype=float))


def market_price_normalized(test_rows: pd.DataFrame) -> np.ndarray:
    """YES price divided by the sum of prices across the meeting's buckets (sums to 1)."""
    return clip_probability(test_rows["mkt_price_norm"].to_numpy(dtype=float))
