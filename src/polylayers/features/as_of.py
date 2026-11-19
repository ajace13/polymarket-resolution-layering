"""Point-in-time lookups against tables carrying an `available_date` column.

`available_date` is the first calendar date on which an observation may be used at 00:00 UTC,
so a value is usable at forecast date T exactly when available_date <= T.
"""

import numpy as np
import pandas as pd


def naive_ns(values: pd.Series) -> pd.Series:
    """Timezone-naive datetimes at nanosecond resolution (merge_asof needs matching dtypes)."""
    return pd.to_datetime(values).astype("datetime64[ns]")


def utc_ns(values: pd.Series) -> pd.Series:
    """UTC datetimes at nanosecond resolution."""
    return pd.to_datetime(values, utc=True).astype("datetime64[ns, UTC]")


def asof_lookup(observations: pd.DataFrame, forecast_dates: pd.Series) -> np.ndarray:
    """Latest `value` available at each forecast date (NaN where nothing is available yet).

    `observations` needs `available_date` and `value`. For the series used here, availability
    rises with the observation date, so the latest available row is also the most recent
    observation.
    """
    right = observations[["available_date", "value"]].copy()
    right["available_date"] = naive_ns(right["available_date"])
    right = right.sort_values("available_date")
    left = pd.DataFrame({"forecast_date": naive_ns(pd.Series(forecast_dates)).to_numpy()})
    left["order"] = np.arange(len(left))
    merged = pd.merge_asof(
        left.sort_values("forecast_date"),
        right,
        left_on="forecast_date",
        right_on="available_date",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values("order")["value"].to_numpy()


def available_before(observations: pd.DataFrame, forecast_date: pd.Timestamp) -> pd.DataFrame:
    """Rows usable at `forecast_date`, sorted by observation date."""
    usable = observations[observations["available_date"] <= forecast_date]
    return usable.sort_values("date")
