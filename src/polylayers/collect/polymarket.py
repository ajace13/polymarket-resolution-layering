"""Polymarket FOMC markets: event metadata (Gamma API), prices and trades (Data API v2).

Discovery combines events tagged `fed-rates` with a fixed list of older event slugs that
predate the tag. Each event is matched to the FOMC meeting whose decision date is closest
to the event's end date. For every order-book market we collect the YES token's price
series and the market's full trade feed.

Data API v2 serves trades keyset-paginated over a fixed three-year window. Trades older
than that window come from an optional seed cache of the retired v1 API
(data/raw/polymarket/trades_legacy/), which held complete histories for every market that
opened before the v2 window.
"""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from polylayers.fomc_calendar import FomcMeeting
from polylayers.http import PoliteClient, RetryPolicy

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com/v2"
EVENT_TAG = "fed-rates"
FOMC_SLUG_PREFIXES = ("fed-interest-rates-", "fed-decision-in-")
MATCH_TOLERANCE_DAYS = 7
PRICE_BUCKET_SECONDS = 10_800
TRADES_PAGE_SIZE = 1_000
V2_TRADE_WINDOW = timedelta(days=3 * 365)
RESOLVED_PRICE_TOLERANCE = 0.01

# Events from before Polymarket tagged rate markets with `fed-rates`.
UNTAGGED_EVENT_SLUGS = (
    "fed-interest-rates-january-2022",
    "fed-interest-rates-march-2022",
    "fed-interest-rates-may-2022",
    "fed-interest-rates-june-2022",
    "fed-interest-rates-july-2022",
    "fed-interest-rates-september-2022",
    "fed-interest-rates-november-2022",
    "fed-interest-rates-december-2022",
    "fed-interest-rates-january-2023",
    "fed-interest-rates-february-2023",
    "fed-interest-rates-march-2023",
    "fed-interest-rates-may-2023",
    "fed-interest-rates-june-2023",
    "fed-interest-rates-july-2023",
    "fed-interest-rates-september-2023",
    "fed-interest-rates-october-2023",
    "fed-interest-rates-november-2023",
    "fed-interest-rates-december-2023",
    "fed-interest-rates-january-2024",
    "fed-interest-rates-march-2024",
    "fed-interest-rates-april-2024",
)

TRADE_COLUMNS = [
    "market_id",
    "timestamp",
    "wallet",
    "side",
    "size",
    "price",
    "outcome_index",
    "transaction_hash",
]

GAMMA_POLICY = RetryPolicy(min_interval_seconds=0.3, backoff_initial_seconds=2.0)
DATA_API_POLICY = RetryPolicy(min_interval_seconds=0.2, backoff_initial_seconds=2.0)


@dataclass(frozen=True)
class MatchedEvent:
    """A Polymarket event paired with the FOMC meeting it resolves on."""

    event: dict[str, Any]
    meeting: FomcMeeting

    @property
    def total_volume(self) -> float:
        return sum(_as_float(market.get("volumeNum")) for market in self.event.get("markets", []))


# --------------------------------------------------------------------------- events


def fetch_tagged_events(client: PoliteClient) -> list[dict[str, Any]]:
    """All events carrying the `fed-rates` tag (open and closed), paging through Gamma."""
    events: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = {"tag_slug": EVENT_TAG, "limit": 100, "offset": offset}
        response = client.get(f"{GAMMA_URL}/events", params=params)
        response.raise_for_status()
        page = response.json()
        events.extend(page)
        if len(page) < 100:
            return events
        offset += 100


def fetch_event_by_slug(client: PoliteClient, slug: str, cache_dir: Path) -> dict[str, Any] | None:
    """One event by slug, cached on disk once closed (closed events never change)."""
    cache_path = cache_dir / f"{slug}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    response = client.get(f"{GAMMA_URL}/events", params={"slug": slug})
    response.raise_for_status()
    matches = response.json()
    if not matches:
        logger.warning("Gamma has no event with slug %s", slug)
        return None
    event = matches[0]
    if event.get("closed"):
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(event), encoding="utf-8")
    return event


def discover_events(client: PoliteClient, raw_dir: Path) -> list[dict[str, Any]]:
    """Tagged events plus untagged legacy slugs plus any cached events, deduplicated by id.

    Fresh API responses take precedence over cached copies of the same event.
    """
    events_dir = raw_dir / "events"
    cached = [json.loads(path.read_text(encoding="utf-8")) for path in events_dir.glob("*.json")]
    untagged = [fetch_event_by_slug(client, slug, events_dir) for slug in UNTAGGED_EVENT_SLUGS]
    tagged = fetch_tagged_events(client)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "events_tagged_snapshot.json").write_text(json.dumps(tagged), encoding="utf-8")

    by_id: dict[str, dict[str, Any]] = {}
    for event in [*cached, *(event for event in untagged if event), *tagged]:
        by_id[str(event["id"])] = event
    logger.info(
        "Polymarket events: %d tagged, %d untagged slugs, %d cached -> %d unique",
        len(tagged),
        sum(event is not None for event in untagged),
        len(cached),
        len(by_id),
    )
    return list(by_id.values())


def is_fomc_event(event: dict[str, Any]) -> bool:
    return str(event.get("slug", "")).startswith(FOMC_SLUG_PREFIXES)


def match_events_to_meetings(
    events: list[dict[str, Any]], meetings: list[FomcMeeting]
) -> list[MatchedEvent]:
    """Closed FOMC events whose end date is within tolerance of a meeting's decision date.

    When several events resolve on the same meeting, the one with the most volume is kept.
    """
    fomc_events = [event for event in events if is_fomc_event(event)]
    closed = [event for event in fomc_events if event.get("closed")]
    matched: dict[str, list[MatchedEvent]] = {}
    unmatched = []
    for event in closed:
        meeting = nearest_meeting(_parse_date(event.get("endDate")), meetings)
        if meeting is None:
            unmatched.append(event.get("slug"))
            continue
        matched.setdefault(meeting.meeting_id, []).append(MatchedEvent(event, meeting))

    selected = []
    for candidates in matched.values():
        candidates.sort(key=lambda candidate: candidate.total_volume, reverse=True)
        selected.append(candidates[0])
        for dropped in candidates[1:]:
            logger.warning(
                "Meeting %s has several events; dropping %s in favor of %s",
                dropped.meeting.meeting_id,
                dropped.event.get("slug"),
                candidates[0].event.get("slug"),
            )
    logger.info(
        "FOMC events: %d by slug, %d closed, %d matched to in-scope meetings, %d outside scope",
        len(fomc_events),
        len(closed),
        len(selected),
        len(unmatched),
    )
    return sorted(selected, key=lambda matched_event: matched_event.meeting)


def nearest_meeting(end_date: date | None, meetings: list[FomcMeeting]) -> FomcMeeting | None:
    """Meeting with the closest decision date, if within the matching tolerance."""
    if end_date is None or not meetings:
        return None
    best = min(meetings, key=lambda meeting: abs((meeting.decision_date - end_date).days))
    if abs((best.decision_date - end_date).days) > MATCH_TOLERANCE_DAYS:
        return None
    return best


# --------------------------------------------------------------------------- markets


def build_markets_table(matched_events: list[MatchedEvent]) -> pd.DataFrame:
    """One row per resolved order-book market, with the reasons for every market dropped."""
    rows = []
    drop_reasons: dict[str, int] = {}
    for matched in matched_events:
        for market in matched.event.get("markets", []):
            row, reason = parse_market(matched, market)
            if row is None:
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                continue
            rows.append(row)
    for reason, count in sorted(drop_reasons.items()):
        logger.info("Dropped %d markets: %s", count, reason)
    markets = pd.DataFrame(rows).sort_values(["decision_date", "market_id"]).reset_index(drop=True)
    yes_per_meeting = markets.groupby("meeting_id")["label"].sum()
    irregular = yes_per_meeting[yes_per_meeting != 1]
    if not irregular.empty:
        logger.warning("Meetings without exactly one YES market: %s", irregular.to_dict())
    logger.info(
        "Markets table: %d markets across %d meetings (YES rate %.3f)",
        len(markets),
        markets["meeting_id"].nunique(),
        markets["label"].mean(),
    )
    return markets


def parse_market(
    matched: MatchedEvent, market: dict[str, Any]
) -> tuple[dict[str, Any] | None, str]:
    """Flatten one Gamma market; returns (None, reason) when it cannot be modeled."""
    if not market.get("enableOrderBook"):
        return None, "no order book (pre-CLOB automated market maker)"
    outcomes = _parse_json_list(market.get("outcomes"))
    prices = [_as_float(price) for price in _parse_json_list(market.get("outcomePrices"))]
    token_ids = _parse_json_list(market.get("clobTokenIds"))
    if outcomes[:2] != ["Yes", "No"] or len(prices) < 2 or len(token_ids) < 2:
        return None, "unexpected outcome structure"
    yes_price = prices[0]
    if abs(yes_price - 1.0) <= RESOLVED_PRICE_TOLERANCE:
        label = 1
    elif abs(yes_price) <= RESOLVED_PRICE_TOLERANCE:
        label = 0
    else:
        return None, "not resolved to YES or NO"
    event = matched.event
    row = {
        "market_id": str(market["id"]),
        "condition_id": market.get("conditionId"),
        "yes_token_id": str(token_ids[0]),
        "question": market.get("question", ""),
        "event_id": str(event["id"]),
        "event_slug": event.get("slug"),
        "meeting_id": matched.meeting.meeting_id,
        "meeting_label": matched.meeting.label,
        "decision_date": pd.Timestamp(matched.meeting.decision_date),
        "created_at": pd.to_datetime(market.get("createdAt"), utc=True),
        "end_date": pd.to_datetime(event.get("endDate"), utc=True),
        "label": label,
        "final_volume_usd": _as_float(market.get("volumeNum")),
    }
    return row, ""


# --------------------------------------------------------------------------- prices


def fetch_price_points(client: PoliteClient, token_id: str, cache_dir: Path) -> list[dict]:
    """Whole-life price series of one outcome token (3-hour grain), cached on disk."""
    cache_path = cache_dir / f"{token_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    params: dict[str, Any] = {
        "token_id": token_id,
        "interval": "max",
        "bucket_seconds": PRICE_BUCKET_SECONDS,
    }
    points: list[dict] = []
    while True:
        response = client.get(f"{DATA_API_URL}/prices-history", params=params)
        response.raise_for_status()
        payload = response.json()
        points.extend(payload.get("data") or [])
        cursor = (payload.get("pagination") or {}).get("next_cursor")
        if not cursor:
            break
        params = {"token_id": token_id, "cursor": cursor}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(points), encoding="utf-8")
    return points


def build_prices_table(
    markets: pd.DataFrame, client: PoliteClient, raw_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Price points per market plus the on-chain settlement point (payout) per market.

    Each point summarizes a window starting at `timestamp` and lasting `resolution_seconds`,
    so it only becomes usable at `available_at` = timestamp + resolution_seconds. The
    settlement point (resolution_seconds == 0) is the payout, not a price, and is returned
    separately to validate labels.
    """
    cache_dir = raw_dir / "prices"
    price_frames = []
    settlements = []
    for market in markets.itertuples():
        points = pd.DataFrame(fetch_price_points(client, market.yes_token_id, cache_dir))
        if points.empty:
            logger.warning("No price history for market %s", market.market_id)
            continue
        is_settlement = points["resolution_seconds"] == 0
        if is_settlement.any():
            payout = points.loc[is_settlement, "price"].iloc[-1]
            settlements.append({"market_id": market.market_id, "payout": payout})
        trading = points.loc[~is_settlement].copy()
        trading["market_id"] = market.market_id
        price_frames.append(trading)
    prices = pd.concat(price_frames, ignore_index=True)
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="s", utc=True)
    prices["available_at"] = prices["timestamp"] + pd.to_timedelta(
        prices["resolution_seconds"], unit="s"
    )
    prices = prices[["market_id", "timestamp", "available_at", "price", "resolution_seconds"]]
    prices = prices.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    logger.info("Prices: %d points across %d markets", len(prices), prices["market_id"].nunique())
    return prices, pd.DataFrame(settlements, columns=["market_id", "payout"])


def validate_labels(markets: pd.DataFrame, settlements: pd.DataFrame) -> None:
    """Log any market whose Gamma label disagrees with its on-chain settlement payout."""
    joined = markets[["market_id", "label"]].merge(settlements, on="market_id", how="left")
    missing = joined["payout"].isna().sum()
    mismatched = joined.dropna(subset=["payout"])
    mismatched = mismatched[(mismatched["payout"] >= 0.5).astype(int) != mismatched["label"]]
    logger.info(
        "Label check: %d markets with an on-chain payout, %d without, %d disagreeing",
        len(joined) - missing,
        missing,
        len(mismatched),
    )
    if not mismatched.empty:
        logger.warning("Label/payout disagreements: %s", mismatched.to_dict("records"))


# --------------------------------------------------------------------------- trades


def fetch_trades_v2(client: PoliteClient, condition_id: str) -> pd.DataFrame:
    """Every trade the v2 feed serves for one market (taker side), walking the cursor."""
    params: dict[str, Any] = {"condition": condition_id, "limit": TRADES_PAGE_SIZE}
    rows: list[dict] = []
    while True:
        response = client.get(f"{DATA_API_URL}/trades", params=params)
        response.raise_for_status()
        payload = response.json()
        rows.extend(payload.get("data") or [])
        cursor = (payload.get("pagination") or {}).get("next_cursor")
        if not cursor:
            break
        params = {"condition": condition_id, "cursor": cursor}
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS[1:])
    frame = frame.rename(columns={"proxy_wallet": "wallet"})
    return frame[TRADE_COLUMNS[1:]]


def load_legacy_trades(legacy_dir: Path, market_id: str) -> pd.DataFrame | None:
    """Trades for one market from the v1 seed cache, if present."""
    path = legacy_dir / f"{market_id}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def merge_trade_sources(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Union of trade frames, dropping fills that appear in more than one source."""
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=TRADE_COLUMNS[1:])
    combined = pd.concat(non_empty, ignore_index=True)
    combined["size"] = combined["size"].astype(float).round(6)
    combined["price"] = combined["price"].astype(float).round(6)
    key = ["transaction_hash", "wallet", "side", "size", "price", "outcome_index", "timestamp"]
    return combined.drop_duplicates(subset=key).sort_values("timestamp").reset_index(drop=True)


def build_trades_table(
    markets: pd.DataFrame, client: PoliteClient, raw_dir: Path, now: datetime | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full trade history per market (v2 feed + v1 seed) and a per-market coverage table."""
    v2_dir = raw_dir / "trades_v2"
    legacy_dir = raw_dir / "trades_legacy"
    v2_dir.mkdir(parents=True, exist_ok=True)
    v2_window_start = (now or datetime.now(UTC)) - V2_TRADE_WINDOW
    trade_frames = []
    coverage_rows = []
    for position, market in enumerate(markets.itertuples(), start=1):
        cache_path = v2_dir / f"{market.market_id}.parquet"
        if cache_path.exists():
            v2_trades = pd.read_parquet(cache_path)
        else:
            v2_trades = fetch_trades_v2(client, market.condition_id)
            _write_parquet_atomically(v2_trades, cache_path)
            logger.info(
                "Trades %d/%d: market %s -> %d rows",
                position,
                len(markets),
                market.market_id,
                len(v2_trades),
            )
        legacy_trades = load_legacy_trades(legacy_dir, market.market_id)
        merged = merge_trade_sources([v2_trades, legacy_trades]).assign(market_id=market.market_id)
        trade_frames.append(merged[TRADE_COLUMNS])
        opened_in_window = market.created_at >= pd.Timestamp(v2_window_start)
        coverage_rows.append(
            {
                "market_id": market.market_id,
                "n_trades": len(merged),
                "n_v2": len(v2_trades),
                "n_legacy": 0 if legacy_trades is None else len(legacy_trades),
                "first_trade_at": _epoch_to_utc(merged["timestamp"].min()),
                "last_trade_at": _epoch_to_utc(merged["timestamp"].max()),
                "history_complete": bool(opened_in_window or legacy_trades is not None),
            }
        )
    trades = pd.concat(trade_frames, ignore_index=True)
    trades["timestamp"] = pd.to_datetime(trades["timestamp"].astype("int64"), unit="s", utc=True)
    coverage = pd.DataFrame(coverage_rows)
    logger.info(
        "Trades: %d rows across %d markets; %d markets with complete history",
        len(trades),
        trades["market_id"].nunique(),
        int(coverage["history_complete"].sum()),
    )
    return trades, coverage


# --------------------------------------------------------------------------- helpers


def _parse_json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _as_float(value: Any) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError):
        return None


def _epoch_to_utc(value: Any) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    return pd.Timestamp(int(value), unit="s", tz="UTC")


def _write_parquet_atomically(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
