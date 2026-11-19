import pytest
import requests

from conftest import FakeResponse, FakeSession
from polylayers.http import DeadlineReachedError, PoliteClient, RetriesExhaustedError, RetryPolicy

URL = "https://example.test/api"


def make_client(outcomes, clock, **policy_overrides):
    policy = RetryPolicy(
        min_interval_seconds=policy_overrides.pop("min_interval_seconds", 0.0),
        backoff_initial_seconds=policy_overrides.pop("backoff_initial_seconds", 2.0),
        backoff_max_seconds=policy_overrides.pop("backoff_max_seconds", 60.0),
        max_attempts=policy_overrides.pop("max_attempts", 5),
    )
    session = FakeSession(list(outcomes))
    return PoliteClient(policy, session=session, clock=clock, sleep=clock.sleep), session


def test_backoff_doubles_and_caps():
    policy = RetryPolicy(backoff_initial_seconds=60, backoff_max_seconds=900)
    delays = [policy.backoff_seconds(attempt) for attempt in range(1, 7)]
    assert delays == [60, 120, 240, 480, 900, 900]


def test_retries_throttled_requests_then_returns_success(fake_clock):
    outcomes = [FakeResponse(429), FakeResponse(503), FakeResponse(200, "ok")]
    client, session = make_client(outcomes, fake_clock)
    response = client.get(URL)
    assert response.text == "ok"
    assert len(session.calls) == 3
    assert fake_clock.sleeps == [2.0, 4.0]


def test_connection_errors_are_retried(fake_clock):
    outcomes = [requests.ConnectionError("reset"), FakeResponse(200, "ok")]
    client, _ = make_client(outcomes, fake_clock)
    assert client.get(URL).text == "ok"


def test_non_retryable_status_is_returned_to_caller(fake_clock):
    client, session = make_client([FakeResponse(404, "missing")], fake_clock)
    assert client.get(URL).status_code == 404
    assert len(session.calls) == 1


def test_gives_up_after_max_attempts(fake_clock):
    client, _ = make_client([FakeResponse(429)] * 3, fake_clock, max_attempts=3)
    with pytest.raises(RetriesExhaustedError):
        client.get(URL)


def test_stops_when_next_retry_would_pass_deadline(fake_clock):
    client, session = make_client([FakeResponse(429)] * 10, fake_clock, max_attempts=100)
    with pytest.raises(DeadlineReachedError):
        client.get(URL, deadline=10.0)
    # Retries at t=2 and t=6 fit inside the deadline; the next (8s wait) would not.
    assert len(session.calls) == 3


def test_min_interval_spaces_consecutive_requests(fake_clock):
    outcomes = [FakeResponse(200), FakeResponse(200)]
    client, _ = make_client(outcomes, fake_clock, min_interval_seconds=10.0)
    client.get(URL)
    fake_clock.now += 3.0
    client.get(URL)
    assert fake_clock.sleeps == [7.0]


def test_retry_after_header_extends_the_backoff(fake_clock):
    outcomes = [FakeResponse(429, headers={"Retry-After": "30"}), FakeResponse(200, "ok")]
    client, _ = make_client(outcomes, fake_clock)
    assert client.get(URL).text == "ok"
    assert fake_clock.sleeps == [30.0]
