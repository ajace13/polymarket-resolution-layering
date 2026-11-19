"""News-layer features: SF Fed news sentiment and GDELT coverage, as of each forecast date.

Windows count the most recent *available* days, so publication lags shift the window back
rather than letting unpublished days leak in.
"""

import numpy as np
import pandas as pd

from polylayers.collect.gdelt import TONE_MODE, VOLUME_MODE
from polylayers.features.as_of import available_before, naive_ns

SENTIMENT_COLUMNS = ["sent_mean_7d", "sent_mean_30d", "sent_chg_7d_vs_30d", "sent_std_30d"]
GDELT_WINDOW_DAYS = 14
# GDELT's daily totals are complete once the day ends.
GDELT_PUBLICATION_LAG = pd.Timedelta(days=1)
GDELT_SHARE_QUERIES = ("fed", "powell", "inflation", "unemployment", "banking_stress")
GDELT_TONE_QUERIES = ("fed", "powell", "inflation", "econ_all")
GDELT_COLUMNS = [
    *(f"gd_{query}_share_14d" for query in GDELT_SHARE_QUERIES),
    *(f"gd_{query}_tone_14d" for query in GDELT_TONE_QUERIES),
    "gd_hawk_dove_balance_14d",
]


def sentiment_features(sentiment: pd.DataFrame, forecast_dates: pd.Series) -> pd.DataFrame:
    """Mean of the last 7 and 30 available days, their difference, and 30-day volatility."""
    sentiment = sentiment.assign(
        date=naive_ns(sentiment["date"]), available_date=naive_ns(sentiment["available_date"])
    )
    rows = []
    for forecast_date in naive_ns(pd.Series(forecast_dates)):
        usable = available_before(sentiment, forecast_date)["sentiment"]
        last_week, last_month = usable.tail(7), usable.tail(30)
        rows.append(
            {
                "forecast_date": forecast_date,
                "sent_mean_7d": last_week.mean(),
                "sent_mean_30d": last_month.mean(),
                "sent_chg_7d_vs_30d": last_week.mean() - last_month.mean(),
                "sent_std_30d": last_month.std(),
            }
        )
    return pd.DataFrame(rows)


def weighted_tone(tone: pd.DataFrame, volume: pd.DataFrame) -> float:
    """Daily tone averaged with that day's article counts as weights (plain mean if no counts)."""
    weights = tone["date"].map(volume.set_index("date")["value"]) if not volume.empty else None
    if weights is None or weights.isna().all() or weights.sum() <= 0:
        return float(tone["value"].mean())
    weights = weights.fillna(0)
    return float((tone["value"] * weights).sum() / weights.sum())


def gdelt_features(daily: pd.DataFrame, forecast_dates: pd.Series) -> pd.DataFrame:
    """Coverage share, tone, and hawkish-vs-dovish balance over the last 14 available days.

    Returns all-NaN columns when GDELT data is missing (the API may still be backfilling).
    """
    dates = naive_ns(pd.Series(forecast_dates)).reset_index(drop=True)
    features = pd.DataFrame({"forecast_date": dates})
    for column in GDELT_COLUMNS:
        features[column] = np.nan
    if daily.empty:
        return features
    daily = daily.assign(date=naive_ns(daily["date"]))
    daily["available_date"] = daily["date"] + GDELT_PUBLICATION_LAG
    volume = daily[daily["mode"] == VOLUME_MODE]
    tone = daily[daily["mode"] == TONE_MODE]

    for index, forecast_date in enumerate(dates):
        # The last GDELT_WINDOW_DAYS available days: [T - 14 days, T - 1 day].
        window_start = forecast_date - pd.Timedelta(days=GDELT_WINDOW_DAYS)
        in_window = (daily["available_date"] <= forecast_date) & (daily["date"] >= window_start)
        volume_window = volume[in_window.loc[volume.index]]
        tone_window = tone[in_window.loc[tone.index]]
        counts = volume_window.groupby("query")["value"].sum()
        totals = volume_window.groupby("query")["norm"].sum()
        for query in GDELT_SHARE_QUERIES:
            if query in counts.index and totals.get(query, 0) > 0:
                # Articles per 10,000 monitored articles.
                features.at[index, f"gd_{query}_share_14d"] = counts[query] / totals[query] * 1e4
        for query in GDELT_TONE_QUERIES:
            query_tone = tone_window[tone_window["query"] == query]
            if not query_tone.empty:
                query_volume = volume_window[volume_window["query"] == query]
                features.at[index, f"gd_{query}_tone_14d"] = weighted_tone(query_tone, query_volume)
        hawkish, dovish = counts.get("hawkish", np.nan), counts.get("dovish", np.nan)
        if hawkish + dovish > 0:
            features.at[index, "gd_hawk_dove_balance_14d"] = (hawkish - dovish) / (hawkish + dovish)
    return features
