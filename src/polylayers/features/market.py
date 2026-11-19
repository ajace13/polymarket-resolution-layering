"""Market-layer features: the market's own price and trading activity at the forecast time.

A forecast for horizon h is made at 00:00 UTC on (decision date - h days). A price point is
usable once its observation window has closed (`available_at`), and only trades strictly
before the forecast time count.
"""

import logging

import numpy as np
import pandas as pd

from polylayers.features.as_of import utc_ns
from polylayers.features.buckets import parse_bucket

logger = logging.getLogger(__name__)

PROBABILITY_FLOOR = 0.005
MOMENTUM_LOOKBACK = pd.Timedelta(days=7)
ACTIVITY_WINDOW = pd.Timedelta(days=7)

BUCKET_COLUMNS = ["bucket_move_bps", "bucket_open_ended", "bucket_is_hold"]
PRICE_COLUMNS = [
    "mkt_price",
    "mkt_price_norm",
    "mkt_logit_norm",
    "mkt_price_change_7d",
    "mkt_is_favorite",
    "mkt_overround",
    "mkt_age_days",
]
TRADING_COLUMNS = [
    "trd_wallets_total",
    "trd_wallets_7d",
    "trd_trades_7d",
    "trd_volume_usd_7d",
    "trd_net_yes_flow_7d",
]


def forecast_skeleton(markets: pd.DataFrame, horizons_days: tuple[int, ...]) -> pd.DataFrame:
    """One row per (market, horizon) with the forecast date/time and bucket encoding."""
    rows = []
    for market in markets.itertuples():
        bucket = parse_bucket(market.question)
        for horizon in horizons_days:
            forecast_date = pd.Timestamp(market.decision_date) - pd.Timedelta(days=horizon)
            rows.append(
                {
                    "market_id": market.market_id,
                    "meeting_id": market.meeting_id,
                    "decision_date": pd.Timestamp(market.decision_date),
                    "horizon_days": horizon,
                    "forecast_date": forecast_date,
                    "forecast_time": forecast_date.tz_localize("UTC"),
                    "question": market.question,
                    "bucket": bucket.name,
                    "bucket_group": bucket.group,
                    "bucket_move_bps": bucket.move_bps,
                    "bucket_open_ended": float(bucket.open_ended),
                    "bucket_is_hold": float(bucket.name == "hold"),
                    "label": market.label,
                }
            )
    return pd.DataFrame(rows)


def prices_at(skeleton: pd.DataFrame, prices: pd.DataFrame, offset: pd.Timedelta) -> np.ndarray:
    """Latest usable YES price at (forecast_time - offset) for every skeleton row."""
    left = skeleton[["market_id", "forecast_time"]].copy()
    left["lookup_time"] = utc_ns(left["forecast_time"]) - offset
    left["order"] = np.arange(len(left))
    right = prices[["market_id", "available_at", "price"]].copy()
    right["available_at"] = utc_ns(right["available_at"])
    right = right.sort_values("available_at")
    merged = pd.merge_asof(
        left.sort_values("lookup_time"),
        right,
        left_on="lookup_time",
        right_on="available_at",
        by="market_id",
        direction="backward",
    )
    return merged.sort_values("order")["price"].to_numpy()


def add_price_features(skeleton: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Price, within-meeting normalized probability, momentum, favorite flag, and age.

    Rows whose market had no usable price yet at the forecast time are dropped (logged per
    horizon): the market did not exist, or had not traded, when the forecast is made.
    """
    panel = skeleton.copy()
    panel["mkt_price"] = prices_at(panel, prices, pd.Timedelta(0))
    panel["price_7d_before"] = prices_at(panel, prices, MOMENTUM_LOOKBACK)
    missing = panel["mkt_price"].isna()
    if missing.any():
        counts = panel.loc[missing].groupby("horizon_days").size().to_dict()
        logger.info("Dropped market-horizon rows with no usable price yet: %s", counts)
    panel = panel.loc[~missing].copy()

    group = panel.groupby(["meeting_id", "horizon_days"])["mkt_price"]
    panel["mkt_overround"] = group.transform("sum")
    panel["mkt_price_norm"] = panel["mkt_price"] / panel["mkt_overround"].where(
        panel["mkt_overround"] > 0
    )
    clipped = panel["mkt_price_norm"].clip(PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    panel["mkt_logit_norm"] = np.log(clipped / (1 - clipped))
    panel["mkt_price_change_7d"] = panel["mkt_price"] - panel["price_7d_before"]
    panel["mkt_is_favorite"] = (panel["mkt_price"] == group.transform("max")).astype(float)

    first_seen = utc_ns(prices["available_at"]).groupby(prices["market_id"]).min()
    age = utc_ns(panel["forecast_time"]) - utc_ns(panel["market_id"].map(first_seen))
    panel["mkt_age_days"] = age.dt.total_seconds() / 86_400
    return panel.drop(columns="price_7d_before")


def yes_flow_sign(side: pd.Series, outcome_index: pd.Series) -> np.ndarray:
    """+1 for trades that add YES exposure (buy YES, sell NO), -1 for the reverse, 0 if unknown."""
    buys = side.to_numpy() == "BUY"
    sign = np.where(outcome_index.to_numpy() == 0, np.where(buys, 1.0, -1.0), 0.0)
    return np.where(outcome_index.to_numpy() == 1, np.where(buys, -1.0, 1.0), sign)


def add_trading_features(
    panel: pd.DataFrame, trades: pd.DataFrame, coverage: pd.DataFrame
) -> pd.DataFrame:
    """Wallet counts, trade counts, USD volume, and net YES flow from trades before the forecast.

    Markets without a complete trade history get NaN rather than misleadingly low counts.
    """
    panel = panel.copy()
    for column in TRADING_COLUMNS:
        panel[column] = np.nan
    complete = set(coverage.loc[coverage["history_complete"], "market_id"])
    for market_id, market_trades in trades.groupby("market_id", sort=False):
        if market_id not in complete:
            continue
        rows = panel.index[panel["market_id"] == market_id]
        if len(rows) == 0:
            continue
        market_trades = market_trades.sort_values("timestamp")
        times = utc_ns(market_trades["timestamp"]).dt.tz_convert(None).to_numpy()
        wallets = market_trades["wallet"].to_numpy()
        usd = (market_trades["size"] * market_trades["price"]).to_numpy()
        flow = usd * yes_flow_sign(market_trades["side"], market_trades["outcome_index"])
        for row in rows:
            cutoff = np.datetime64(panel.at[row, "forecast_time"].tz_convert(None), "ns")
            end = np.searchsorted(times, cutoff, side="left")
            start = np.searchsorted(times, cutoff - ACTIVITY_WINDOW.to_timedelta64(), side="left")
            panel.at[row, "trd_wallets_total"] = len(np.unique(wallets[:end]))
            panel.at[row, "trd_wallets_7d"] = len(np.unique(wallets[start:end]))
            panel.at[row, "trd_trades_7d"] = end - start
            panel.at[row, "trd_volume_usd_7d"] = usd[start:end].sum()
            panel.at[row, "trd_net_yes_flow_7d"] = flow[start:end].sum()
    no_history = panel["trd_wallets_total"].isna().sum()
    if no_history:
        logger.info("Trading features unavailable (incomplete history) for %d rows", no_history)
    return panel
