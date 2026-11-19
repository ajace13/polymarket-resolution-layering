import json
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from conftest import FakeResponse, FakeSession
from polylayers.collect import polymarket
from polylayers.fomc_calendar import FomcMeeting
from polylayers.http import PoliteClient, RetryPolicy

MARCH = FomcMeeting(decision_date=date(2024, 3, 20), start_date=date(2024, 3, 19))
MAY = FomcMeeting(decision_date=date(2024, 5, 1), start_date=date(2024, 4, 30))


def make_client(payloads, clock):
    outcomes = [FakeResponse(200, json.dumps(payload)) for payload in payloads]
    policy = RetryPolicy(min_interval_seconds=0.0)
    session = FakeSession(outcomes)
    return PoliteClient(policy, session=session, clock=clock, sleep=clock.sleep), session


def gamma_market(market_id, yes_price, order_book=True, volume=100.0, outcomes=("Yes", "No")):
    return {
        "id": market_id,
        "question": f"Market {market_id}?",
        "conditionId": f"0x{market_id}",
        "clobTokenIds": json.dumps([f"{market_id}1", f"{market_id}2"]),
        "outcomes": json.dumps(list(outcomes)),
        "outcomePrices": json.dumps([str(yes_price), str(1 - yes_price)]),
        "enableOrderBook": order_book,
        "volumeNum": volume,
        "createdAt": "2024-02-01T00:00:00Z",
    }


def gamma_event(event_id, slug, end_date, markets, closed=True):
    return {"id": event_id, "slug": slug, "endDate": end_date, "closed": closed, "markets": markets}


def test_is_fomc_event_uses_slug_prefixes():
    assert polymarket.is_fomc_event({"slug": "fed-decision-in-march-885"})
    assert polymarket.is_fomc_event({"slug": "fed-interest-rates-may-2023"})
    assert not polymarket.is_fomc_event({"slug": "how-many-fed-rate-cuts-in-2025"})


def test_nearest_meeting_respects_tolerance():
    assert polymarket.nearest_meeting(date(2024, 3, 21), [MARCH, MAY]) == MARCH
    assert polymarket.nearest_meeting(date(2024, 4, 5), [MARCH, MAY]) is None
    assert polymarket.nearest_meeting(None, [MARCH]) is None


def test_match_keeps_closed_in_scope_events_and_the_largest_duplicate():
    small, large = [gamma_market("a", 1, volume=5)], [gamma_market("b", 1, volume=50)]
    events = [
        gamma_event("1", "fed-decision-in-march", "2024-03-20T00:00:00Z", small),
        gamma_event("2", "fed-decision-in-march-2", "2024-03-21", large),
        gamma_event("3", "fed-decision-in-may", "2024-05-01", [gamma_market("c", 1)], closed=False),
        gamma_event("4", "fed-decision-in-july", "2024-07-31", [gamma_market("d", 1)]),
        gamma_event("5", "bitcoin-above-100k", "2024-03-20", [gamma_market("e", 1)]),
    ]
    matched = polymarket.match_events_to_meetings(events, [MARCH, MAY])
    assert [m.event["id"] for m in matched] == ["2"]
    assert matched[0].meeting == MARCH


def test_parse_market_labels_resolved_markets_and_explains_drops():
    event = gamma_event("1", "fed-decision-in-march", "2024-03-20", [])
    matched = polymarket.MatchedEvent(event, MARCH)
    row, _ = polymarket.parse_market(matched, gamma_market("a", 1.0))
    assert row["label"] == 1 and row["yes_token_id"] == "a1" and row["meeting_id"] == "2024-03-20"
    row, _ = polymarket.parse_market(matched, gamma_market("b", 0.0))
    assert row["label"] == 0
    _, reason = polymarket.parse_market(matched, gamma_market("c", 0.5))
    assert reason == "not resolved to YES or NO"
    assert polymarket.parse_market(matched, gamma_market("d", 1, order_book=False))[0] is None
    assert polymarket.parse_market(matched, gamma_market("e", 1, outcomes=("No", "Yes")))[0] is None


def test_build_markets_table_counts_one_yes_per_meeting():
    event = gamma_event(
        "1",
        "fed-decision-in-march",
        "2024-03-20",
        [gamma_market("a", 1), gamma_market("b", 0), gamma_market("c", 0.5)],
    )
    markets = polymarket.build_markets_table([polymarket.MatchedEvent(event, MARCH)])
    assert markets["market_id"].tolist() == ["a", "b"]
    assert markets.groupby("meeting_id")["label"].sum().tolist() == [1]


def test_fetch_trades_walks_the_cursor_and_keeps_slim_columns(fake_clock):
    row = {
        "timestamp": 1710000000,
        "proxy_wallet": "0xabc",
        "side": "BUY",
        "size": 5.0,
        "price": 0.4,
        "outcome_index": 0,
        "transaction_hash": "0xt1",
        "name": "someone",
        "bio": "private-ish",
    }
    pages = [
        {"data": [row], "pagination": {"has_more": True, "next_cursor": "c1"}},
        {"data": [{**row, "transaction_hash": "0xt2"}], "pagination": {"next_cursor": None}},
    ]
    client, session = make_client(pages, fake_clock)
    trades = polymarket.fetch_trades_v2(client, "0xcond")
    assert len(trades) == 2
    assert "name" not in trades.columns and "bio" not in trades.columns
    assert session.calls[1]["params"] == {"condition": "0xcond", "cursor": "c1"}


def test_merge_trade_sources_drops_fills_seen_in_both_sources():
    trade = {
        "timestamp": 1,
        "wallet": "0xa",
        "side": "BUY",
        "size": 1.0000001,
        "price": 0.5,
        "outcome_index": 0,
        "transaction_hash": "0xt",
    }
    v2 = pd.DataFrame([trade, {**trade, "transaction_hash": "0xu", "timestamp": 2}])
    legacy = pd.DataFrame([{**trade, "size": 1.0}])
    merged = polymarket.merge_trade_sources([v2, legacy, None])
    assert merged["transaction_hash"].tolist() == ["0xt", "0xu"]


def test_build_prices_table_separates_settlement_and_sets_availability(tmp_path, fake_clock):
    markets = pd.DataFrame([{"market_id": "a", "yes_token_id": "a1"}])
    points = [
        {"timestamp": 1710000000, "price": 0.4, "resolution_seconds": 10800},
        {"timestamp": 1710043200, "price": 0.6, "resolution_seconds": 10800},
        {"timestamp": 1710054000, "price": 1.0, "resolution_seconds": 0},
    ]
    client, _ = make_client([{"data": points, "pagination": {"next_cursor": None}}], fake_clock)
    prices, settlements = polymarket.build_prices_table(markets, client, tmp_path)
    assert prices["price"].tolist() == [0.4, 0.6]
    assert prices.loc[0, "available_at"] == pd.Timestamp(1710000000 + 10800, unit="s", tz="UTC")
    assert settlements.to_dict("records") == [{"market_id": "a", "payout": 1.0}]
    assert (tmp_path / "prices" / "a1.json").exists()


def test_validate_labels_reports_disagreements(caplog):
    markets = pd.DataFrame({"market_id": ["a", "b"], "label": [1, 0]})
    settlements = pd.DataFrame({"market_id": ["a", "b"], "payout": [1.0, 1.0]})
    with caplog.at_level("WARNING"):
        polymarket.validate_labels(markets, settlements)
    assert "disagreements" in caplog.text


def test_build_trades_table_uses_caches_and_flags_coverage(tmp_path, fake_clock):
    now = datetime(2026, 11, 19, tzinfo=UTC)
    markets = pd.DataFrame(
        {
            "market_id": ["old", "new"],
            "condition_id": ["0xold", "0xnew"],
            "created_at": pd.to_datetime(["2023-01-01", "2025-01-01"], utc=True),
        }
    )
    trade = {
        "timestamp": 1700000000,
        "wallet": "0xa",
        "side": "BUY",
        "size": 1.0,
        "price": 0.5,
        "outcome_index": 0,
        "transaction_hash": "0xt",
    }
    (tmp_path / "trades_v2").mkdir()
    (tmp_path / "trades_legacy").mkdir()
    pd.DataFrame([trade]).to_parquet(tmp_path / "trades_v2" / "new.parquet")
    pd.DataFrame(columns=list(trade)).to_parquet(tmp_path / "trades_v2" / "old.parquet")
    client, session = make_client([], fake_clock)
    trades, coverage = polymarket.build_trades_table(markets, client, tmp_path, now=now)
    assert session.calls == []
    assert trades["market_id"].tolist() == ["new"]
    complete = dict(zip(coverage["market_id"], coverage["history_complete"], strict=True))
    assert complete == {"old": False, "new": True}


@pytest.mark.parametrize("value", ['["1", "0"]', ["1", "0"]])
def test_parse_json_list_accepts_strings_and_lists(value):
    assert polymarket._parse_json_list(value) == ["1", "0"]
