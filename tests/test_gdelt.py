import json
from datetime import date

import pandas as pd
import pytest

from conftest import FakeResponse, FakeSession
from polylayers.collect import gdelt
from polylayers.http import PoliteClient, RetryPolicy

THROTTLE_TEXT = "Please limit requests to one every 5 seconds or contact kalev.leetaru5@gmail.com"


def timeline_payload(points):
    return {"query_details": {}, "timeline": [{"series": "Article Count", "data": points}]}


def make_client(outcomes, clock):
    policy = RetryPolicy(
        min_interval_seconds=0.0,
        backoff_initial_seconds=60,
        backoff_max_seconds=900,
        max_attempts=1_000,
    )
    return PoliteClient(policy, session=FakeSession(list(outcomes)), clock=clock, sleep=clock.sleep)


def two_tasks():
    query = gdelt.GdeltQuery("fed", "theme:ECON_INTEREST_RATES", (gdelt.VOLUME_MODE,))
    return [
        gdelt.FetchTask(query, gdelt.VOLUME_MODE, date(2023, 1, 1), date(2023, 12, 31)),
        gdelt.FetchTask(query, gdelt.VOLUME_MODE, date(2024, 1, 1), date(2024, 12, 31)),
    ]


def test_yearly_windows_clip_first_and_last_year():
    windows = gdelt.yearly_windows(date(2022, 10, 1), date(2026, 10, 28))
    assert windows[0] == (date(2022, 10, 1), date(2022, 12, 31))
    assert windows[1] == (date(2023, 1, 1), date(2023, 12, 31))
    assert windows[-1] == (date(2026, 1, 1), date(2026, 10, 28))
    assert len(windows) == 5


def test_yearly_windows_reject_reversed_bounds():
    with pytest.raises(ValueError):
        gdelt.yearly_windows(date(2024, 1, 2), date(2024, 1, 1))


def test_build_tasks_covers_every_query_mode_and_window_once():
    tasks = gdelt.build_tasks(gdelt.QUERIES, date(2022, 10, 1), date(2026, 10, 28))
    modes_per_window = sum(len(query.modes) for query in gdelt.QUERIES)
    assert len(tasks) == modes_per_window * 5
    assert len({task.cache_name for task in tasks}) == len(tasks)


def test_request_params_use_full_day_bounds():
    params = two_tasks()[0].request_params()
    assert params["startdatetime"] == "20230101000000"
    assert params["enddatetime"] == "20231231235959"
    assert params["format"] == "json"


def test_parse_payload_rejects_plain_text_and_json_without_timeline():
    assert gdelt.parse_payload(THROTTLE_TEXT) is None
    assert gdelt.parse_payload(json.dumps({"error": "bad query"})) is None
    assert gdelt.parse_payload(json.dumps(timeline_payload([]))) is not None


def test_parse_timeline_reads_daily_points_with_and_without_norm():
    with_norm = gdelt.parse_timeline(
        timeline_payload(
            [
                {"date": "20230101T000000Z", "value": 5, "norm": 1000},
                {"date": "20230102T000000Z", "value": 7, "norm": 1100},
            ]
        )
    )
    assert with_norm["value"].tolist() == [5.0, 7.0]
    assert with_norm["norm"].tolist() == [1000.0, 1100.0]
    tone = gdelt.parse_timeline(timeline_payload([{"date": "20230101T000000Z", "value": -1.5}]))
    assert tone["norm"].isna().all()


def test_parse_timeline_handles_empty_timeline():
    assert gdelt.parse_timeline({"timeline": []}).empty


def test_daily_aggregation_sums_counts_and_averages_tone():
    points = gdelt.parse_timeline(
        timeline_payload(
            [
                {"date": "20230101T000000Z", "value": 2, "norm": 100},
                {"date": "20230101T120000Z", "value": 3, "norm": 150},
                {"date": "20230102T000000Z", "value": 4, "norm": 200},
            ]
        )
    )
    volume = gdelt.daily_from_points(points, gdelt.VOLUME_MODE)
    assert volume["value"].tolist() == [5.0, 4.0]
    assert volume["norm"].tolist() == [250.0, 200.0]
    tone = gdelt.daily_from_points(points, gdelt.TONE_MODE)
    assert tone["value"].tolist() == [2.5, 4.0]


def test_fetch_skips_cached_tasks_and_caches_new_responses(tmp_path, fake_clock):
    tasks = two_tasks()
    (tmp_path / tasks[0].cache_name).write_text(json.dumps(timeline_payload([])))
    payload = timeline_payload([{"date": "20240101T000000Z", "value": 9, "norm": 900}])
    client = make_client([FakeResponse(200, json.dumps(payload))], fake_clock)
    summary = gdelt.fetch_timelines(tasks, tmp_path, client, deadline=1e9)
    assert (summary.cached_before, summary.fetched) == (1, 1)
    assert summary.complete
    assert json.loads((tmp_path / tasks[1].cache_name).read_text()) == payload


def test_fetch_records_rejections_without_caching(tmp_path, fake_clock):
    tasks = two_tasks()[:1]
    client = make_client([FakeResponse(200, "Invalid query")], fake_clock)
    summary = gdelt.fetch_timelines(tasks, tmp_path, client, deadline=1e9)
    assert summary.rejected == [tasks[0].cache_name]
    assert not (tmp_path / tasks[0].cache_name).exists()


def test_fetch_stops_at_deadline_and_reports_remaining(tmp_path, fake_clock):
    tasks = two_tasks()
    client = make_client([FakeResponse(429, THROTTLE_TEXT)] * 50, fake_clock)
    summary = gdelt.fetch_timelines(tasks, tmp_path, client, deadline=500.0)
    assert summary.remaining == [task.cache_name for task in tasks]
    assert not summary.complete
    assert list(tmp_path.iterdir()) == []


def test_build_daily_table_combines_cached_responses(tmp_path):
    tasks = two_tasks()
    payload = timeline_payload([{"date": "20230105T000000Z", "value": 3, "norm": 300}])
    (tmp_path / tasks[0].cache_name).write_text(json.dumps(payload))
    table = gdelt.build_daily_table(tasks, tmp_path)
    assert table.columns.tolist() == ["date", "query", "mode", "value", "norm"]
    assert len(table) == 1
    assert table.loc[0, "date"] == pd.Timestamp("2023-01-05")
    assert table.loc[0, "query"] == "fed"
