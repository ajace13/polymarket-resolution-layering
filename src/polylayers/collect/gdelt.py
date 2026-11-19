"""GDELT DOC 2.0 API: daily news-coverage timelines for Fed-related searches.

For each search we request a daily timeline per calendar year: article counts
(`timelinevolraw`, which also reports the total number of articles GDELT monitored that day)
and, where useful, average tone (`timelinetone`). Daily series let the feature step compute
windows ending at any forecast horizon, instead of fixing one window per meeting.

Each successful response is cached as raw JSON under data/raw/gdelt/, so an interrupted run
resumes where it stopped. GDELT throttles aggressively (HTTP 429 with a plain-text body),
so requests are spaced out and throttling triggers exponential backoff until a deadline.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from polylayers.config import GdeltSettings
from polylayers.http import DeadlineReachedError, PoliteClient, RetriesExhaustedError, RetryPolicy

logger = logging.getLogger(__name__)

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
VOLUME_MODE = "timelinevolraw"
TONE_MODE = "timelinetone"


@dataclass(frozen=True)
class GdeltQuery:
    """One GDELT search and the timeline modes collected for it."""

    name: str
    query: str
    modes: tuple[str, ...]


# Themes come from GDELT's Global Knowledge Graph taxonomy; keywords cover what themes don't.
QUERIES: tuple[GdeltQuery, ...] = (
    GdeltQuery("fed", "theme:ECON_INTEREST_RATES", (VOLUME_MODE, TONE_MODE)),
    GdeltQuery("powell", '"Jerome Powell"', (VOLUME_MODE, TONE_MODE)),
    GdeltQuery("hawkish", "hawkish", (VOLUME_MODE,)),
    GdeltQuery("dovish", "dovish", (VOLUME_MODE,)),
    GdeltQuery("inflation", "theme:ECON_INFLATION", (VOLUME_MODE, TONE_MODE)),
    GdeltQuery("unemployment", "theme:ECON_UNEMPLOYMENT", (VOLUME_MODE,)),
    GdeltQuery("banking_stress", "theme:ECON_BANKRUPTCY", (VOLUME_MODE,)),
    GdeltQuery(
        "econ_all",
        "(theme:ECON_INTEREST_RATES OR theme:ECON_INFLATION OR theme:ECON_UNEMPLOYMENT"
        " OR theme:ECON_BANKRUPTCY)",
        (TONE_MODE,),
    ),
)


@dataclass(frozen=True)
class FetchTask:
    """One API request: a query, a timeline mode, and an inclusive date window."""

    query: GdeltQuery
    mode: str
    window_start: date
    window_end: date

    @property
    def cache_name(self) -> str:
        return (
            f"{self.query.name}__{self.mode}__"
            f"{self.window_start:%Y%m%d}_{self.window_end:%Y%m%d}.json"
        )

    def request_params(self) -> dict[str, str]:
        return {
            "query": self.query.query,
            "mode": self.mode,
            "format": "json",
            "startdatetime": f"{self.window_start:%Y%m%d}000000",
            "enddatetime": f"{self.window_end:%Y%m%d}235959",
        }


@dataclass
class FetchSummary:
    """What a fetch run did, for logging and for deciding whether to rerun."""

    cached_before: int = 0
    fetched: int = 0
    rejected: list[str] = field(default_factory=list)
    remaining: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.rejected and not self.remaining


def yearly_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end] into calendar-year windows with inclusive bounds."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    windows = []
    for year in range(start.year, end.year + 1):
        window_start = max(start, date(year, 1, 1))
        window_end = min(end, date(year, 12, 31))
        windows.append((window_start, window_end))
    return windows


def build_tasks(queries: tuple[GdeltQuery, ...], start: date, end: date) -> list[FetchTask]:
    """Every (query, mode, yearly window) request needed to cover [start, end]."""
    return [
        FetchTask(query, mode, window_start, window_end)
        for query in queries
        for mode in query.modes
        for window_start, window_end in yearly_windows(start, end)
    ]


def parse_payload(text: str) -> dict[str, Any] | None:
    """Decode a GDELT response body; None when GDELT replied with a plain-text message."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "timeline" not in payload:
        return None
    return payload


def fetch_timelines(
    tasks: list[FetchTask],
    raw_dir: Path,
    client: PoliteClient,
    deadline: float,
) -> FetchSummary:
    """Fetch every task not already cached; stop early if throttling outlasts the deadline."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary = FetchSummary()
    pending = []
    for task in tasks:
        if (raw_dir / task.cache_name).exists():
            summary.cached_before += 1
        else:
            pending.append(task)
    logger.info(
        "GDELT: %d/%d responses cached, %d to fetch",
        summary.cached_before,
        len(tasks),
        len(pending),
    )

    for position, task in enumerate(pending):
        try:
            response = client.get(API_URL, params=task.request_params(), deadline=deadline)
        except (DeadlineReachedError, RetriesExhaustedError) as error:
            summary.remaining = [pending_task.cache_name for pending_task in pending[position:]]
            logger.warning(
                "GDELT: stopping with %d requests left (%s)", len(summary.remaining), error
            )
            return summary
        payload = parse_payload(response.text)
        if response.status_code != 200 or payload is None:
            message = response.text[:160].replace("\n", " ")
            summary.rejected.append(task.cache_name)
            logger.error(
                "GDELT rejected %s (HTTP %d): %s", task.cache_name, response.status_code, message
            )
            continue
        (raw_dir / task.cache_name).write_text(json.dumps(payload), encoding="utf-8")
        summary.fetched += 1
        logger.info("GDELT: cached %s (%d/%d)", task.cache_name, position + 1, len(pending))
    return summary


def parse_timeline(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten a timeline payload to one row per point: UTC `timestamp`, `value`, `norm`.

    `norm` (total articles GDELT monitored in the interval) only exists for volume timelines
    and is NaN otherwise. `daily_from_points` aggregates the points to calendar days.
    """
    timeline = payload.get("timeline") or []
    points = timeline[0].get("data", []) if timeline else []
    if not points:
        return pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                "value": pd.Series(dtype=float),
                "norm": pd.Series(dtype=float),
            }
        )
    frame = pd.DataFrame(points)
    frame["timestamp"] = pd.to_datetime(frame["date"], format="%Y%m%dT%H%M%SZ", utc=True)
    if "norm" not in frame.columns:
        frame["norm"] = float("nan")
    return frame[["timestamp", "value", "norm"]].astype({"value": float, "norm": float})


def daily_from_points(points: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Aggregate timeline points to calendar days: counts are summed, tone is averaged."""
    days = points.assign(date=points["timestamp"].dt.tz_localize(None).dt.normalize())
    if mode == VOLUME_MODE:
        return days.groupby("date", as_index=False)[["value", "norm"]].sum(min_count=1)
    return days.groupby("date", as_index=False)[["value"]].mean()


def build_daily_table(tasks: list[FetchTask], raw_dir: Path) -> pd.DataFrame:
    """Long table of every cached response: date, query, mode, value, norm.

    Missing responses are logged and simply absent, so downstream features for those dates
    come out as NaN rather than zero (no coverage is not the same as no articles).
    """
    frames = []
    missing = []
    for task in tasks:
        path = raw_dir / task.cache_name
        if not path.exists():
            missing.append(task.cache_name)
            continue
        points = parse_timeline(json.loads(path.read_text(encoding="utf-8")))
        daily = daily_from_points(points, task.mode)
        if task.mode == TONE_MODE:
            daily["norm"] = float("nan")
        frames.append(daily.assign(query=task.query.name, mode=task.mode))
    if missing:
        logger.warning(
            "GDELT: %d of %d responses not cached yet: %s",
            len(missing),
            len(tasks),
            ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""),
        )
    if not frames:
        return pd.DataFrame(columns=["date", "query", "mode", "value", "norm"])
    table = pd.concat(frames, ignore_index=True)
    table = table[["date", "query", "mode", "value", "norm"]]
    table = table.sort_values(["query", "mode", "date"]).reset_index(drop=True)
    logger.info(
        "GDELT daily table: %d rows, %d query/mode series",
        len(table),
        table.groupby(["query", "mode"]).ngroups,
    )
    return table


def retry_policy(settings: GdeltSettings) -> RetryPolicy:
    """Backoff settings for GDELT: effectively unlimited attempts, bounded by a deadline."""
    return RetryPolicy(
        min_interval_seconds=settings.min_interval_seconds,
        backoff_initial_seconds=settings.backoff_initial_seconds,
        backoff_max_seconds=settings.backoff_max_seconds,
        max_attempts=1_000_000,
        timeout_seconds=120.0,
    )


def deadline_after(hours: float) -> float:
    """A deadline `hours` from now on the monotonic clock used by `PoliteClient`."""
    return time.monotonic() + hours * 3600
