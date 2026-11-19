"""San Francisco Fed Daily News Sentiment Index (Shapiro, Sudhof and Wilson, 2022).

A daily, lexicon-based sentiment score of economics articles in major U.S. newspapers,
published as an Excel workbook that is updated weekly. Because of the weekly cadence, a
value for day d is treated as usable only `PUBLICATION_LAG_DAYS` later.
"""

import io
import logging
from pathlib import Path

import pandas as pd

from polylayers.http import PoliteClient, RetryPolicy

logger = logging.getLogger(__name__)

WORKBOOK_URL = "https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx"
SF_FED_POLICY = RetryPolicy(min_interval_seconds=1.0, backoff_initial_seconds=5.0)
PUBLICATION_LAG_DAYS = 8


def download_workbook(client: PoliteClient, cache_path: Path) -> bytes:
    """The workbook bytes, downloaded once and cached so reruns use the same vintage."""
    if cache_path.exists():
        return cache_path.read_bytes()
    response = client.get(WORKBOOK_URL)
    response.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(response.content)
    return response.content


def parse_workbook(content: bytes) -> pd.DataFrame:
    """Daily sentiment with availability dates: date, sentiment, available_date."""
    raw = pd.read_excel(io.BytesIO(content), sheet_name="Data", engine="openpyxl")
    frame = raw.rename(columns={raw.columns[0]: "date", raw.columns[1]: "sentiment"})
    frame = frame[["date", "sentiment"]].copy()
    total = len(frame)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["sentiment"] = pd.to_numeric(frame["sentiment"], errors="coerce")
    frame = frame.dropna().drop_duplicates(subset="date", keep="last").sort_values("date")
    dropped = total - len(frame)
    if dropped:
        logger.info("SF Fed sentiment: dropped %d unparseable or duplicate rows", dropped)
    frame["available_date"] = frame["date"] + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
    logger.info(
        "SF Fed sentiment: %d days, %s to %s",
        len(frame),
        frame["date"].min().date(),
        frame["date"].max().date(),
    )
    return frame.reset_index(drop=True)
