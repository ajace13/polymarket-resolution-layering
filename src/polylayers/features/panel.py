"""Assemble the modeling panel: one row per (market, horizon), every layer point-in-time.

Feature groups:
- bucket:  what the market pays out on (signed move in bps, open-ended flag, hold flag)
- market:  the market's own price, within-meeting normalized probability, momentum
- trading: wallets, trades, USD volume, and net YES flow before the forecast
- macro:   rates-implied policy path, inflation, labor, activity, financial conditions
- news:    SF Fed news sentiment and GDELT coverage/tone
- text:    the latest FOMC statement's action, guidance, tone, and dissents

Macro, news, and text are meeting-level: identical for every market in a meeting at a given
horizon. Models interact them with the bucket encoding so they can favor one outcome over
another (see `polylayers.models.design`).
"""

import logging

import pandas as pd

from polylayers.features.macro import MACRO_COLUMNS, macro_features
from polylayers.features.market import (
    BUCKET_COLUMNS,
    PRICE_COLUMNS,
    TRADING_COLUMNS,
    add_price_features,
    add_trading_features,
    forecast_skeleton,
)
from polylayers.features.news import (
    GDELT_COLUMNS,
    SENTIMENT_COLUMNS,
    gdelt_features,
    sentiment_features,
)
from polylayers.features.text import TEXT_COLUMNS, statement_features
from polylayers.fomc_calendar import all_meetings

logger = logging.getLogger(__name__)

FEATURE_GROUPS: dict[str, list[str]] = {
    "bucket": BUCKET_COLUMNS,
    "market": PRICE_COLUMNS,
    "trading": TRADING_COLUMNS,
    "macro": MACRO_COLUMNS,
    "news": [*SENTIMENT_COLUMNS, *GDELT_COLUMNS],
    "text": TEXT_COLUMNS,
}
MEETING_LEVEL_GROUPS = ("macro", "news", "text")

# Cumulative information layers compared in the evaluation, in the order they are added.
LAYER_STAGES: dict[str, tuple[str, ...]] = {
    "market": ("bucket", "market", "trading"),
    "market+macro": ("bucket", "market", "trading", "macro"),
    "market+macro+news": ("bucket", "market", "trading", "macro", "news"),
    "market+macro+news+text": ("bucket", "market", "trading", "macro", "news", "text"),
}
# The same layers without the market's own price: can fundamentals alone pick the outcome?
FUNDAMENTAL_STAGES: dict[str, tuple[str, ...]] = {
    "macro": ("bucket", "macro"),
    "macro+news": ("bucket", "macro", "news"),
    "macro+news+text": ("bucket", "macro", "news", "text"),
}
ID_COLUMNS = [
    "market_id",
    "meeting_id",
    "decision_date",
    "horizon_days",
    "forecast_date",
    "forecast_time",
    "question",
    "bucket",
    "bucket_group",
    "label",
]


def previous_decision_dates(decision_dates: pd.Series) -> pd.Series:
    """Decision date of the scheduled meeting before each given decision date."""
    calendar = [pd.Timestamp(meeting.decision_date) for meeting in all_meetings()]
    previous = {current: earlier for earlier, current in zip(calendar, calendar[1:], strict=False)}
    return pd.to_datetime(decision_dates).map(previous)


def meeting_level_features(
    forecasts: pd.DataFrame,
    fred_observations: pd.DataFrame,
    sentiment: pd.DataFrame,
    statements: pd.DataFrame,
    gdelt_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Macro, news, and text features for each (meeting, horizon) forecast date."""
    forecasts = forecasts.reset_index(drop=True)
    dates = forecasts["forecast_date"]
    previous = previous_decision_dates(forecasts["decision_date"])
    blocks = [
        macro_features(fred_observations, dates, previous),
        sentiment_features(sentiment, dates),
        gdelt_features(gdelt_daily, dates),
        statement_features(statements, dates),
    ]
    columns = [block.drop(columns="forecast_date").reset_index(drop=True) for block in blocks]
    return pd.concat([forecasts, *columns], axis=1)


def build_panel(
    markets: pd.DataFrame,
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    coverage: pd.DataFrame,
    fred_observations: pd.DataFrame,
    sentiment: pd.DataFrame,
    statements: pd.DataFrame,
    gdelt_daily: pd.DataFrame,
    horizons_days: tuple[int, ...],
) -> pd.DataFrame:
    """The full modeling panel with ID columns followed by every feature group."""
    skeleton = forecast_skeleton(markets, horizons_days)
    logger.info("Panel skeleton: %s (markets x horizons)", skeleton.shape)
    panel = add_price_features(skeleton, prices)
    panel = add_trading_features(panel, trades, coverage)
    keys = ["meeting_id", "horizon_days", "forecast_date", "decision_date"]
    forecasts = panel[keys].drop_duplicates()
    meeting_level = meeting_level_features(
        forecasts, fred_observations, sentiment, statements, gdelt_daily
    )
    before = len(panel)
    panel = panel.merge(meeting_level, on=keys, how="left", validate="many_to_one")
    if len(panel) != before:
        raise ValueError(f"Meeting-level merge changed the row count: {before} -> {len(panel)}")
    feature_columns = [column for group in FEATURE_GROUPS.values() for column in group]
    panel = panel[ID_COLUMNS + feature_columns].sort_values(["horizon_days", "decision_date"])
    panel = panel.reset_index(drop=True)
    logger.info(
        "Panel: %s; rows per horizon %s",
        panel.shape,
        panel.groupby("horizon_days").size().to_dict(),
    )
    return panel
