"""Test doubles shared across the suite: no test touches the network or really sleeps."""

import json
from dataclasses import dataclass, field

import pytest


@dataclass
class FakeResponse:
    status_code: int
    text: str = ""
    headers: dict = field(default_factory=dict)

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@dataclass
class FakeSession:
    """Returns (or raises) pre-programmed outcomes in order and records each call."""

    outcomes: list
    calls: list = field(default_factory=list)

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClock:
    """A manual clock: `sleep` advances time instantly and records the requested delay."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()
