"""FRED macro series with point-in-time availability dates.

Two kinds of series, handled by separate fetchers:

- Daily market series (Treasury yields, the effective fed funds rate, VIX) are effectively
  never revised. An observation for day d is published by the next business day, so it is
  usable from 00:00 UTC on the day after that.
- Macro releases (inflation, labor, activity, financial conditions) are revised after they
  come out. We pull ALFRED's first-release values (`output_type=4`); each row's
  `realtime_start` is its publication date, so it is usable from the next day. A feature at
  time t therefore only sees numbers that had been published before t.
"""

import json
import logging
import os
from datetime import date
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BDay

from polylayers.http import PoliteClient, RetryPolicy

logger = logging.getLogger(__name__)

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_POLICY = RetryPolicy(min_interval_seconds=0.6, backoff_initial_seconds=2.0)

DAILY_SERIES = {
    "EFFR": "Effective federal funds rate (%)",
    "DGS3MO": "3-month Treasury constant-maturity yield (%)",
    "DGS6MO": "6-month Treasury constant-maturity yield (%)",
    "DGS2": "2-year Treasury constant-maturity yield (%)",
    "T10Y2Y": "10-year minus 2-year Treasury spread (percentage points)",
    "VIXCLS": "CBOE VIX volatility index",
}
RELEASE_SERIES = {
    "CPILFESL": "Core CPI index (first release)",
    "PCEPILFE": "Core PCE price index (first release)",
    "UNRATE": "Unemployment rate, % (first release)",
    "ICSA": "Initial jobless claims (first release)",
    "NFCI": "Chicago Fed National Financial Conditions Index (first release)",
    "INDPRO": "Industrial production index (first release)",
}
OBSERVATION_COLUMNS = ["series_id", "date", "value", "available_date"]


def api_key() -> str:
    """The FRED API key from the environment (loaded from .env by the CLI)."""
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise OSError("FRED_API_KEY is not set; copy .env.example to .env and add a key.")
    return key


def fetch_observations(
    client: PoliteClient, params: dict[str, str], cache_path: Path
) -> list[dict[str, str]]:
    """Raw FRED observations for one request, cached as JSON (the key is never cached)."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    response = client.get(FRED_URL, params={**params, "api_key": api_key(), "file_type": "json"})
    response.raise_for_status()
    observations = response.json()["observations"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(observations), encoding="utf-8")
    return observations


def daily_series_table(series_id: str, observations: list[dict[str, str]]) -> pd.DataFrame:
    """Daily observations with availability = next business day + 1 calendar day."""
    frame = _numeric_observations(series_id, observations)
    frame["available_date"] = frame["date"] + BDay(1) + pd.Timedelta(days=1)
    return frame[OBSERVATION_COLUMNS]


def release_series_table(series_id: str, observations: list[dict[str, str]]) -> pd.DataFrame:
    """First-release observations with availability = publication date + 1 day."""
    frame = _numeric_observations(series_id, observations)
    frame["available_date"] = frame["realtime_start"] + pd.Timedelta(days=1)
    return frame[OBSERVATION_COLUMNS]


def collect_fred(client: PoliteClient, raw_dir: Path, start: date, end: date) -> pd.DataFrame:
    """All configured series as one long table: series_id, date, value, available_date."""
    frames = []
    window = {"observation_start": start.isoformat(), "observation_end": end.isoformat()}
    for series_id in DAILY_SERIES:
        cache_path = raw_dir / f"{series_id}_daily_{start:%Y%m%d}_{end:%Y%m%d}.json"
        observations = fetch_observations(client, {"series_id": series_id, **window}, cache_path)
        frames.append(daily_series_table(series_id, observations))
    for series_id in RELEASE_SERIES:
        params = {
            "series_id": series_id,
            "output_type": "4",
            "realtime_start": start.isoformat(),
            "realtime_end": "9999-12-31",
            **window,
        }
        cache_path = raw_dir / f"{series_id}_first_release_{start:%Y%m%d}_{end:%Y%m%d}.json"
        observations = fetch_observations(client, params, cache_path)
        frames.append(release_series_table(series_id, observations))
    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values(["series_id", "date"]).reset_index(drop=True)
    logger.info("FRED: %d observations across %d series", len(table), table["series_id"].nunique())
    return table


def _numeric_observations(series_id: str, observations: list[dict[str, str]]) -> pd.DataFrame:
    """Parse dates and values, dropping FRED's '.' placeholders for missing days."""
    frame = pd.DataFrame(observations)
    total = len(frame)
    frame = frame[frame["value"] != "."].copy()
    if len(frame) < total:
        logger.info("FRED %s: dropped %d placeholder rows ('.')", series_id, total - len(frame))
    frame["series_id"] = series_id
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = frame["value"].astype(float)
    if "realtime_start" in frame.columns:
        frame["realtime_start"] = pd.to_datetime(frame["realtime_start"])
    return frame
